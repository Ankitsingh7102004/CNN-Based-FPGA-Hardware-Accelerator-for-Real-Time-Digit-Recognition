`timescale 1ns/1ps
// =============================================================================
// mnist_fpga_top.v  -  DE10-Standard top level (Phase C: inference from stored
//                      MNIST test images, no camera yet)
//
// Controls
//   KEY0 : reset (hold, then release)
//   KEY1 : next image      (runs inference automatically)
//   KEY2 : previous image  (runs inference automatically)
//   KEY3 : re-run the current image
//
// Display
//   HEX0 : predicted digit by the FPGA  ("-" while running)
//   HEX2 : true label of the shown test image
//   HEX5,HEX4 : image number 00..99
//   LEDR0 : busy (inference running)
//   LEDR1 : prediction correct    LEDR2 : prediction wrong
//
// Needs in the Quartus project folder: cnn_core_v1.v, quant_params.vh and all
// *.mem files from results/step4_export (test_images.mem, test_labels.mem, ...)
// =============================================================================

// ---------------------------------------------------------------------------
// Push-button conditioner: 2-FF synchronizer + falling-edge pulse + lockout
// ---------------------------------------------------------------------------
module btn_pulse (
    input  wire clk,
    input  wire rst_n,
    input  wire key_n,          // active-low button
    output reg  pulse           // 1 clock when pressed
);
    reg [1:0]  s;
    reg        prev;
    reg [15:0] lock;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s <= 2'b11; prev <= 1'b1; lock <= 16'd0; pulse <= 1'b0;
        end else begin
            s     <= {s[0], key_n};
            prev  <= s[1];
            pulse <= 1'b0;
            if (lock != 16'd0) lock <= lock - 1'b1;
            else if (prev == 1'b1 && s[1] == 1'b0) begin
                pulse <= 1'b1;
                lock  <= 16'hFFFF;        // ~1.3 ms lockout against bounce
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
module mnist_fpga_top (
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    output wire [9:0]  LEDR,
    output wire [6:0]  HEX0,
    output wire [6:0]  HEX1,
    output wire [6:0]  HEX2,
    output wire [6:0]  HEX3,
    output wire [6:0]  HEX4,
    output wire [6:0]  HEX5
);
    // Terasic 7-segment displays are active-low (0 lights a segment).
    // If your digits look inverted, change this to 0.
    localparam ACTIVE_LOW = 1;

    localparam N_IMG = 100;

    wire clk = CLOCK_50;

    // ---------------- reset ----------------
    reg [1:0] rst_sync;
    always @(posedge clk or negedge KEY[0]) begin
        if (!KEY[0]) rst_sync <= 2'b00;
        else         rst_sync <= {rst_sync[0], 1'b1};
    end
    wire rst_n = rst_sync[1];

    // ---------------- buttons ----------------
    wire p_next, p_prev, p_rerun;
    btn_pulse b1 (.clk(clk), .rst_n(rst_n), .key_n(KEY[1]), .pulse(p_next));
    btn_pulse b2 (.clk(clk), .rst_n(rst_n), .key_n(KEY[2]), .pulse(p_prev));
    btn_pulse b3 (.clk(clk), .rst_n(rst_n), .key_n(KEY[3]), .pulse(p_rerun));

    // ---------------- test-image and label ROMs ----------------
    reg [7:0] img_rom [0:78399];              // 100 images x 784 pixels
    reg [7:0] lab_rom [0:N_IMG-1];
    initial begin
        $readmemh("test_images.mem", img_rom);
        $readmemh("test_labels.mem", lab_rom);
    end

    reg  [16:0] img_addr;
    reg  [7:0]  img_q;
    reg  [6:0]  idx;                           // current image 0..99
    reg  [7:0]  lab_q;
    always @(posedge clk) begin
        img_q <= img_rom[img_addr];
        lab_q <= lab_rom[idx];
    end

    // ---------------- CNN core ----------------
    reg         pix_wr_en;
    reg  [9:0]  pix_wr_addr;
    reg  [7:0]  pix_wr_data;
    reg         start;
    wire        busy, done;
    wire [3:0]  pred;
    wire [319:0] logits_unused;

    cnn_core_v1 core (
        .clk(clk), .rst_n(rst_n),
        .pix_wr_en(pix_wr_en), .pix_wr_addr(pix_wr_addr), .pix_wr_data(pix_wr_data),
        .start(start), .busy(busy), .done(done),
        .pred(pred), .logits_flat(logits_unused)
    );

    // ---------------- image feeder + run control ----------------
    localparam [2:0] T_COPY = 3'd0, T_FLUSH = 3'd1, T_START = 3'd2,
                     T_WAIT = 3'd3, T_IDLE  = 3'd4;

    reg [2:0]  tstate;
    reg [16:0] base;                           // idx * 784
    reg [9:0]  cnt;
    reg [3:0]  fl;
    reg        v0, v1;
    reg [9:0]  a0, a1;
    reg [3:0]  pred_r;
    reg        have_result;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tstate <= T_COPY;                  // auto-run image 0 after reset
            idx <= 7'd0; base <= 17'd0; cnt <= 10'd0; fl <= 4'd0;
            v0 <= 1'b0; a0 <= 10'd0; img_addr <= 17'd0;
            start <= 1'b0; pred_r <= 4'd0; have_result <= 1'b0;
        end else begin
            v0    <= 1'b0;
            start <= 1'b0;

            case (tstate)
                T_COPY: begin                  // stream 784 pixels into the core
                    img_addr <= base + cnt;
                    v0       <= 1'b1;
                    a0       <= cnt;
                    if (cnt == 10'd783) begin
                        fl <= 4'd0; tstate <= T_FLUSH;
                    end else cnt <= cnt + 1'b1;
                end

                T_FLUSH: begin                 // let the copy pipeline drain
                    fl <= fl + 1'b1;
                    if (fl == 4'd6) begin
                        start  <= 1'b1;
                        tstate <= T_START;
                    end
                end

                T_START: tstate <= T_WAIT;

                T_WAIT: begin
                    if (done) begin
                        pred_r      <= pred;
                        have_result <= 1'b1;
                        tstate      <= T_IDLE;
                    end
                end

                T_IDLE: begin
                    if (p_next) begin
                        idx  <= (idx == N_IMG-1) ? 7'd0 : idx + 1'b1;
                        base <= (idx == N_IMG-1) ? 17'd0 : base + 17'd784;
                        cnt <= 10'd0; have_result <= 1'b0; tstate <= T_COPY;
                    end else if (p_prev) begin
                        idx  <= (idx == 7'd0) ? N_IMG-1 : idx - 1'b1;
                        base <= (idx == 7'd0) ? (N_IMG-1)*784 : base - 17'd784;
                        cnt <= 10'd0; have_result <= 1'b0; tstate <= T_COPY;
                    end else if (p_rerun) begin
                        cnt <= 10'd0; have_result <= 1'b0; tstate <= T_COPY;
                    end
                end

                default: tstate <= T_IDLE;
            endcase
        end
    end

    // pixel-write pipeline: address -> ROM read -> core write port
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; a1 <= 10'd0;
            pix_wr_en <= 1'b0; pix_wr_addr <= 10'd0; pix_wr_data <= 8'd0;
        end else begin
            v1          <= v0;
            a1          <= a0;
            pix_wr_en   <= v1;
            pix_wr_addr <= a1;
            pix_wr_data <= img_q;
        end
    end

    // ---------------- display ----------------
    function [6:0] seg;                        // active-low, bit0 = segment a
        input [3:0] d;
        case (d)
            4'd0: seg = 7'b1000000;
            4'd1: seg = 7'b1111001;
            4'd2: seg = 7'b0100100;
            4'd3: seg = 7'b0110000;
            4'd4: seg = 7'b0011001;
            4'd5: seg = 7'b0010010;
            4'd6: seg = 7'b0000010;
            4'd7: seg = 7'b1111000;
            4'd8: seg = 7'b0000000;
            4'd9: seg = 7'b0010000;
            default: seg = 7'b1111111;
        endcase
    endfunction

    localparam [6:0] SEG_DASH  = 7'b0111111;   // only segment g lit
    localparam [6:0] SEG_BLANK = 7'b1111111;

    wire running   = (tstate != T_IDLE);
    wire show_pred = have_result && !running;
    wire [3:0] idx_tens = idx / 10;
    wire [3:0] idx_ones = idx % 10;

    wire [6:0] h0 = show_pred ? seg(pred_r) : SEG_DASH;
    wire [6:0] h2 = seg(lab_q[3:0]);
    wire [6:0] h4 = seg(idx_ones);
    wire [6:0] h5 = seg(idx_tens);

    assign HEX0 = ACTIVE_LOW ? h0 : ~h0;
    assign HEX1 = ACTIVE_LOW ? SEG_BLANK : ~SEG_BLANK;
    assign HEX2 = ACTIVE_LOW ? h2 : ~h2;
    assign HEX3 = ACTIVE_LOW ? SEG_BLANK : ~SEG_BLANK;
    assign HEX4 = ACTIVE_LOW ? h4 : ~h4;
    assign HEX5 = ACTIVE_LOW ? h5 : ~h5;

    assign LEDR[0]   = running;
    assign LEDR[1]   = show_pred && (pred_r == lab_q[3:0]);
    assign LEDR[2]   = show_pred && (pred_r != lab_q[3:0]);
    assign LEDR[9:3] = 7'd0;

endmodule
