`timescale 1ns/1ps
// =============================================================================
// cnn_core_v1.v  -  MNIST CNN accelerator, version 1 (serial: ONE MAC unit)
//
// Network (from your QAT_int8 model):
//   28x28 pixel -> LUT -> int8
//   conv1 3x3 (1->8)  + ReLU + maxpool2x2 -> 8x14x14   -> requant int8
//   conv2 3x3 (8->16) + ReLU + maxpool2x2 -> 16x7x7    -> requant int8
//   fc1   784 -> 32   + ReLU                            -> requant int8
//   fc2   32  -> 10   (int32 logits, no requant)        -> argmax
//
// Integer recipe per output value (identical to export_int.py):
//   acc  = sum(int8_in * int8_w) + int32_bias
//   acc  = max(acc, 0); (conv layers: max over the 2x2 pool window)
//   out  = clamp((acc * M + 2^(S-1)) >>> S, -128, 127)
//
// Files needed in the simulation/Quartus folder (all from results/step4_export):
//   quant_params.vh, input_lut.mem, conv1_w.mem, conv1_b.mem, conv2_w.mem,
//   conv2_b.mem, fc1_w.mem, fc1_b.mem, fc2_w.mem, fc2_b.mem
//
// Usage:
//   1. Write the 784 pixels (raster order, 0..255) through the pix_wr_* port.
//   2. Pulse `start` for one clock.
//   3. Wait for `done` (1-clock pulse). `pred` = digit, logits_flat = 10 x int32.
// =============================================================================
module cnn_core_v1 (
    input  wire         clk,
    input  wire         rst_n,

    // pixel write port (raw 8-bit grayscale, MNIST polarity: white digit on black)
    input  wire         pix_wr_en,
    input  wire [9:0]   pix_wr_addr,
    input  wire [7:0]   pix_wr_data,

    input  wire         start,
    output reg          busy,
    output reg          done,          // 1-cycle pulse
    output reg  [3:0]   pred,
    output wire [319:0] logits_flat    // logit i = logits_flat[32*i +: 32]
);

`include "quant_params.vh"

    // ------------------------------------------------------------------
    // Feature-map RAM layout (one 8-bit RAM, four regions)
    // ------------------------------------------------------------------
    //   0    .. 783   : q0  input after LUT      (1  x 28 x 28)
    //   784  .. 2351  : q1  after conv1+pool     (8  x 14 x 14)
    //   2352 .. 3135  : q2  after conv2+pool     (16 x  7 x  7)  = fc1 input
    //   3136 .. 3167  : q3  after fc1            (32)            = fc2 input

    // FSM states
    localparam [3:0] S_IDLE   = 4'd0,
                     S_LOAD   = 4'd1,
                     S_LDWAIT = 4'd2,
                     S_LAYER  = 4'd3,
                     S_RUN    = 4'd4,
                     S_WAIT   = 4'd5,
                     S_ELEM   = 4'd6,
                     S_REQ1   = 4'd7,
                     S_REQ2   = 4'd8,
                     S_ARG    = 4'd9,
                     S_FIN    = 4'd10;

    reg [3:0] state;

    // ------------------------------------------------------------------
    // Memories
    // ------------------------------------------------------------------
    reg        [7:0]  pix_mem [0:783];
    reg signed [7:0]  lut_mem [0:255];
    reg signed [7:0]  w1_mem  [0:71];
    reg signed [7:0]  w2_mem  [0:1151];
    reg signed [7:0]  w3_mem  [0:25087];
    reg signed [7:0]  w4_mem  [0:319];
    reg signed [31:0] b1_mem  [0:7];
    reg signed [31:0] b2_mem  [0:15];
    reg signed [31:0] b3_mem  [0:31];
    reg signed [31:0] b4_mem  [0:9];
    reg signed [7:0]  fm_mem  [0:4095];

    initial begin
        $readmemh("input_lut.mem", lut_mem);
        $readmemh("conv1_w.mem",   w1_mem);
        $readmemh("conv2_w.mem",   w2_mem);
        $readmemh("fc1_w.mem",     w3_mem);
        $readmemh("fc2_w.mem",     w4_mem);
        $readmemh("conv1_b.mem",   b1_mem);
        $readmemh("conv2_b.mem",   b2_mem);
        $readmemh("fc1_b.mem",     b3_mem);
        $readmemh("fc2_b.mem",     b4_mem);
    end

    // ------------------------------------------------------------------
    // Layer parameters (combinational, selected by `layer`)
    // ------------------------------------------------------------------
    reg [1:0]  layer;
    integer    L_H, L_PH, L_OC, L_K, L_IN, L_OUT, L_S;
    reg [15:0] L_M;

    always @(*) begin
        case (layer)
            2'd0: begin L_H=28; L_PH=14; L_OC=8;  L_K=9;   L_IN=0;    L_OUT=784;
                        L_M=M_CONV1; L_S=SHIFT_CONV1; end
            2'd1: begin L_H=14; L_PH=7;  L_OC=16; L_K=72;  L_IN=784;  L_OUT=2352;
                        L_M=M_CONV2; L_S=SHIFT_CONV2; end
            2'd2: begin L_H=0;  L_PH=0;  L_OC=32; L_K=784; L_IN=2352; L_OUT=3136;
                        L_M=M_FC1;   L_S=SHIFT_FC1;   end
            default: begin L_H=0; L_PH=0; L_OC=10; L_K=32; L_IN=3136; L_OUT=0;
                        L_M=16'd0;   L_S=1;           end
        endcase
    end

    // ------------------------------------------------------------------
    // Loop counters
    // ------------------------------------------------------------------
    reg [5:0]  oc;          // output channel / neuron
    reg [3:0]  py, px;      // pooled output position
    reg        dy, dx;      // position inside the 2x2 pool window
    reg [9:0]  t;           // flat tap index 0..K-1
    reg [1:0]  ky, kx;      // kernel position (conv only)
    reg [3:0]  ic;          // input channel (conv only)
    reg [14:0] w_base;      // oc * K
    reg [11:0] out_idx;     // running output index inside the output region

    // input address generation for the current tap
    integer y_i, x_i, iy_i, ix_i, in_addr_i;
    reg     in_ok;
    always @(*) begin
        y_i  = 2*py + dy;
        x_i  = 2*px + dx;
        iy_i = y_i + ky - 1;
        ix_i = x_i + kx - 1;
        if (layer[1]) begin
            in_ok     = 1'b1;
            in_addr_i = L_IN + t;
        end else begin
            in_ok     = (iy_i >= 0) && (iy_i < L_H) && (ix_i >= 0) && (ix_i < L_H);
            in_addr_i = L_IN + ic*L_H*L_H + iy_i*L_H + ix_i;
        end
    end

    // ------------------------------------------------------------------
    // Input load pipeline: pix_mem -> LUT -> fm_mem[q0]
    // ------------------------------------------------------------------
    reg [9:0]  pix_rd_addr, ld_cnt, ld_a0, ld_a1, ld_a2;
    reg [7:0]  pix_q;
    reg signed [7:0] lut_q;
    reg        ld_v0, ld_v1, ld_v2;
    reg [2:0]  wcnt;

    always @(posedge clk) begin
        if (pix_wr_en) pix_mem[pix_wr_addr] <= pix_wr_data;
        pix_q <= pix_mem[pix_rd_addr];
        lut_q <= lut_mem[pix_q];
    end

    // ------------------------------------------------------------------
    // Feature-map RAM (1 write port, 1 registered read port)
    // ------------------------------------------------------------------
    reg         req_we;
    reg  [11:0] req_wa;
    reg  [7:0]  req_wd;
    reg  [11:0] fm_rd_addr;
    reg signed [7:0] fm_q;

    wire        fm_we = ld_v2 | req_we;
    wire [11:0] fm_wa = ld_v2 ? {2'b00, ld_a2} : req_wa;
    wire [7:0]  fm_wd = ld_v2 ? lut_q          : req_wd;

    always @(posedge clk) begin
        if (fm_we) fm_mem[fm_wa] <= fm_wd;
        fm_q <= fm_mem[fm_rd_addr];
    end

    // ------------------------------------------------------------------
    // Weight / bias ROMs (registered reads)
    // ------------------------------------------------------------------
    reg [14:0] w_addr;
    reg signed [7:0]  w1_q, w2_q, w3_q, w4_q;
    reg signed [31:0] b1_q, b2_q, b3_q, b4_q;

    always @(posedge clk) begin
        w1_q <= w1_mem[w_addr[6:0]];
        w2_q <= w2_mem[w_addr[10:0]];
        w3_q <= w3_mem[w_addr[14:0]];
        w4_q <= w4_mem[w_addr[8:0]];
        b1_q <= b1_mem[oc[2:0]];
        b2_q <= b2_mem[oc[3:0]];
        b3_q <= b3_mem[oc[4:0]];
        b4_q <= b4_mem[oc[3:0]];
    end

    reg signed [7:0]  w_sel;
    reg signed [31:0] bias_sel;
    always @(*) begin
        case (layer)
            2'd0:    begin w_sel = w1_q; bias_sel = b1_q; end
            2'd1:    begin w_sel = w2_q; bias_sel = b2_q; end
            2'd2:    begin w_sel = w3_q; bias_sel = b3_q; end
            default: begin w_sel = w4_q; bias_sel = b4_q; end
        endcase
    end

    // ------------------------------------------------------------------
    // MAC pipeline:  stage A (addresses issued) -> stage B (data back)
    //                -> stage C (multiply-accumulate)
    // ------------------------------------------------------------------
    reg a_v, a_inb, a_first, a_last;
    reg b_v, b_inb, b_first, b_last;
    reg signed [31:0] acc;
    reg acc_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            b_v <= 1'b0; b_inb <= 1'b0; b_first <= 1'b0; b_last <= 1'b0;
            acc <= 32'sd0; acc_done <= 1'b0;
            ld_v1 <= 1'b0; ld_v2 <= 1'b0; ld_a1 <= 10'd0; ld_a2 <= 10'd0;
        end else begin
            ld_v1 <= ld_v0; ld_a1 <= ld_a0;
            ld_v2 <= ld_v1; ld_a2 <= ld_a1;
            b_v <= a_v; b_inb <= a_inb; b_first <= a_first; b_last <= a_last;
            acc_done <= 1'b0;
            if (b_v) begin
                acc <= (b_first ? 32'sd0 : acc) + (b_inb ? (fm_q * w_sel) : 32'sd0);
                if (b_last) acc_done <= 1'b1;
            end
        end
    end

    // ------------------------------------------------------------------
    // Result processing
    // ------------------------------------------------------------------
    wire signed [31:0] elem_sum = acc + bias_sel;
    wire signed [31:0] relu_sum = elem_sum[31] ? 32'sd0 : elem_sum;

    reg  signed [31:0] pmax;
    reg  signed [63:0] req_prod;
    wire signed [16:0] m_s     = {1'b0, L_M};
    wire signed [63:0] shifted = (req_prod + (64'sd1 <<< (L_S - 1))) >>> L_S;

    reg  signed [31:0] logit [0:9];
    reg  [3:0]         arg_i, best_i;
    reg  signed [31:0] best_v;

    assign logits_flat = {logit[9], logit[8], logit[7], logit[6], logit[5],
                          logit[4], logit[3], logit[2], logit[1], logit[0]};

    // ------------------------------------------------------------------
    // Main FSM
    // ------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; pred <= 4'd0; layer <= 2'd0;
            oc <= 6'd0; py <= 4'd0; px <= 4'd0; dy <= 1'b0; dx <= 1'b0;
            t <= 10'd0; kx <= 2'd0; ky <= 2'd0; ic <= 4'd0;
            w_base <= 15'd0; out_idx <= 12'd0; pmax <= 32'sd0; req_prod <= 64'sd0;
            req_we <= 1'b0; req_wa <= 12'd0; req_wd <= 8'd0;
            fm_rd_addr <= 12'd0; w_addr <= 15'd0;
            a_v <= 1'b0; a_inb <= 1'b0; a_first <= 1'b0; a_last <= 1'b0;
            pix_rd_addr <= 10'd0; ld_v0 <= 1'b0; ld_a0 <= 10'd0; ld_cnt <= 10'd0;
            wcnt <= 3'd0; arg_i <= 4'd0; best_v <= 32'sd0; best_i <= 4'd0;
        end else begin
            // one-cycle pulses default to 0
            done   <= 1'b0;
            a_v    <= 1'b0;
            ld_v0  <= 1'b0;
            req_we <= 1'b0;

            case (state)
                // ---------------------------------------------------------
                S_IDLE: begin
                    if (start) begin
                        busy   <= 1'b1;
                        ld_cnt <= 10'd0;
                        state  <= S_LOAD;
                    end
                end

                // read pixels, apply LUT, store as int8 in q0 region
                S_LOAD: begin
                    pix_rd_addr <= ld_cnt;
                    ld_v0       <= 1'b1;
                    ld_a0       <= ld_cnt;
                    if (ld_cnt == 10'd783) begin
                        wcnt  <= 3'd0;
                        state <= S_LDWAIT;
                    end else begin
                        ld_cnt <= ld_cnt + 1'b1;
                    end
                end

                S_LDWAIT: begin           // let the load pipeline flush
                    wcnt <= wcnt + 1'b1;
                    if (wcnt == 3'd4) begin
                        layer <= 2'd0;
                        state <= S_LAYER;
                    end
                end

                // reset loop counters at the start of every layer
                S_LAYER: begin
                    oc <= 6'd0; py <= 4'd0; px <= 4'd0; dy <= 1'b0; dx <= 1'b0;
                    t <= 10'd0; kx <= 2'd0; ky <= 2'd0; ic <= 4'd0;
                    w_base <= 15'd0; out_idx <= 12'd0;
                    state <= S_RUN;
                end

                // issue K taps, one per clock
                S_RUN: begin
                    fm_rd_addr <= in_ok ? in_addr_i[11:0] : 12'd0;
                    w_addr     <= w_base + t;
                    a_v        <= 1'b1;
                    a_inb      <= in_ok;
                    a_first    <= (t == 0);
                    a_last     <= (t == L_K - 1);
                    if (t == L_K - 1) begin
                        t <= 10'd0; kx <= 2'd0; ky <= 2'd0; ic <= 4'd0;
                        state <= S_WAIT;
                    end else begin
                        t <= t + 1'b1;
                        if (kx == 2'd2) begin
                            kx <= 2'd0;
                            if (ky == 2'd2) begin
                                ky <= 2'd0;
                                ic <= ic + 1'b1;
                            end else ky <= ky + 1'b1;
                        end else kx <= kx + 1'b1;
                    end
                end

                // wait until the accumulator holds the final sum
                S_WAIT: begin
                    if (acc_done) state <= S_ELEM;
                end

                // bias add + ReLU + pooling max
                S_ELEM: begin
                    if (layer == 2'd3) begin
                        logit[oc] <= elem_sum;               // fc2: raw int32 logits
                        if (oc == L_OC - 1) begin
                            oc <= 6'd0; arg_i <= 4'd0; state <= S_ARG;
                        end else begin
                            oc <= oc + 1'b1; w_base <= w_base + L_K; state <= S_RUN;
                        end
                    end else begin
                        if (layer[1] || (dy == 1'b0 && dx == 1'b0)) pmax <= relu_sum;
                        else if (relu_sum > pmax)                   pmax <= relu_sum;

                        if (layer[1] || (dy && dx)) begin
                            state <= S_REQ1;                 // pooled value complete
                        end else begin
                            if (dx) begin dx <= 1'b0; dy <= 1'b1; end
                            else          dx <= 1'b1;
                            state <= S_RUN;                  // next pool-window position
                        end
                    end
                end

                // requantize: multiply ...
                S_REQ1: begin
                    req_prod <= pmax * m_s;
                    state    <= S_REQ2;
                end

                // ... shift, clamp to int8, write, advance to next output
                S_REQ2: begin
                    req_we <= 1'b1;
                    req_wa <= L_OUT[11:0] + out_idx;
                    if (shifted > 64'sd127)        req_wd <= 8'd127;
                    else if (shifted < -64'sd128)  req_wd <= 8'h80;
                    else                           req_wd <= shifted[7:0];

                    out_idx <= out_idx + 1'b1;
                    dy <= 1'b0; dx <= 1'b0;

                    if (!layer[1]) begin                      // conv layers
                        if (px == L_PH - 1) begin
                            px <= 4'd0;
                            if (py == L_PH - 1) begin
                                py <= 4'd0;
                                if (oc == L_OC - 1) begin
                                    layer <= layer + 1'b1; state <= S_LAYER;
                                end else begin
                                    oc <= oc + 1'b1; w_base <= w_base + L_K; state <= S_RUN;
                                end
                            end else begin
                                py <= py + 1'b1; state <= S_RUN;
                            end
                        end else begin
                            px <= px + 1'b1; state <= S_RUN;
                        end
                    end else begin                            // fc1
                        if (oc == L_OC - 1) begin
                            layer <= layer + 1'b1; state <= S_LAYER;
                        end else begin
                            oc <= oc + 1'b1; w_base <= w_base + L_K; state <= S_RUN;
                        end
                    end
                end

                // argmax over the 10 logits (first maximum wins, like numpy)
                S_ARG: begin
                    if (arg_i == 4'd0) begin
                        best_v <= logit[0]; best_i <= 4'd0; arg_i <= 4'd1;
                    end else begin
                        if (logit[arg_i] > best_v) begin
                            best_v <= logit[arg_i]; best_i <= arg_i;
                        end
                        if (arg_i == 4'd9) state <= S_FIN;
                        arg_i <= arg_i + 1'b1;
                    end
                end

                S_FIN: begin
                    pred  <= best_i;
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
