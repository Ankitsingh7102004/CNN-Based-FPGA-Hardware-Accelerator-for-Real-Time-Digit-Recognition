`timescale 1ns/1ps
// =============================================================================
// tb_cnn_v1.v - checks cnn_core_v1 against the exported integer-model vectors.
//
// Needs (in the folder you run from): test_images.mem, expected_pred.mem,
// expected_logits.mem, test_labels.mem plus everything cnn_core_v1 needs.
//
// Plusargs:
//   +n=<count>   number of images to test (default 100, max 100)
//   +dump        write a waveform file (use with +n=1, files get huge otherwise)
// =============================================================================
module tb_cnn_v1;

    reg clk   = 1'b0;
    reg rst_n = 1'b0;
    always #10 clk = ~clk;                       // 50 MHz

    reg         pix_wr_en   = 1'b0;
    reg  [9:0]  pix_wr_addr = 10'd0;
    reg  [7:0]  pix_wr_data = 8'd0;
    reg         start       = 1'b0;
    wire        busy, done;
    wire [3:0]  pred;
    wire [319:0] logits_flat;

    cnn_core_v1 dut (
        .clk(clk), .rst_n(rst_n),
        .pix_wr_en(pix_wr_en), .pix_wr_addr(pix_wr_addr), .pix_wr_data(pix_wr_data),
        .start(start), .busy(busy), .done(done),
        .pred(pred), .logits_flat(logits_flat)
    );

    localparam MAXN = 100;
    reg [7:0]  img_mem   [0:MAXN*784-1];
    reg [7:0]  exp_pred  [0:MAXN-1];
    reg [7:0]  lab_mem   [0:MAXN-1];
    reg [31:0] exp_logit [0:MAXN*10-1];

    integer n_imgs, i, j, k, cycles, total_cycles;
    integer pred_ok, logit_ok, label_ok, logit_bad;

    initial begin
        if (!$value$plusargs("n=%d", n_imgs)) n_imgs = MAXN;
        if (n_imgs > MAXN) n_imgs = MAXN;
        if ($test$plusargs("dump")) begin
            $dumpfile("cnn_v1.vcd");
            $dumpvars(0, tb_cnn_v1);
        end

        $readmemh("test_images.mem",    img_mem);
        $readmemh("expected_pred.mem",  exp_pred);
        $readmemh("test_labels.mem",    lab_mem);
        $readmemh("expected_logits.mem", exp_logit);

        pred_ok = 0; logit_ok = 0; label_ok = 0; total_cycles = 0;

        #100 rst_n = 1'b1;
        #100;

        for (i = 0; i < n_imgs; i = i + 1) begin
            // load image
            for (j = 0; j < 784; j = j + 1) begin
                @(posedge clk);
                pix_wr_en   <= 1'b1;
                pix_wr_addr <= j[9:0];
                pix_wr_data <= img_mem[i*784 + j];
            end
            @(posedge clk);
            pix_wr_en <= 1'b0;

            // start and wait for done
            @(posedge clk); start <= 1'b1;
            @(posedge clk); start <= 1'b0;
            cycles = 0;
            while (!done) begin
                @(posedge clk);
                cycles = cycles + 1;
                if (cycles > 3000000) begin
                    $display("TIMEOUT on image %0d", i);
                    $finish;
                end
            end
            total_cycles = total_cycles + cycles;

            // compare
            logit_bad = 0;
            for (k = 0; k < 10; k = k + 1)
                if (logits_flat[32*k +: 32] !== exp_logit[i*10 + k]) logit_bad = logit_bad + 1;

            if (pred === exp_pred[i][3:0]) pred_ok  = pred_ok + 1;
            if (logit_bad == 0)            logit_ok = logit_ok + 1;
            if (pred === lab_mem[i][3:0])  label_ok = label_ok + 1;

            if (logit_bad != 0 || pred !== exp_pred[i][3:0]) begin
                $display("MISMATCH image %0d: pred=%0d expected=%0d, %0d/10 logits differ",
                         i, pred, exp_pred[i], logit_bad);
                if (logit_bad != 0)
                    for (k = 0; k < 10; k = k + 1)
                        $display("   logit[%0d]: got %0d, expected %0d", k,
                                 $signed(logits_flat[32*k +: 32]), $signed(exp_logit[i*10 + k]));
            end
        end

        $display("");
        $display("==================================================");
        $display("Images tested            : %0d", n_imgs);
        $display("Predictions matching     : %0d / %0d", pred_ok,  n_imgs);
        $display("Logits bit-exact         : %0d / %0d", logit_ok, n_imgs);
        $display("Correct vs true labels   : %0d / %0d", label_ok, n_imgs);
        $display("Avg cycles per image     : %0d (excludes pixel upload)", total_cycles / n_imgs);
        $display("Time per image @ 50 MHz  : %0d us", (total_cycles / n_imgs) * 20 / 1000);
        if (logit_ok == n_imgs) $display("RESULT: PASS - hardware matches the integer model exactly");
        else                    $display("RESULT: FAIL");
        $display("==================================================");
        $finish;
    end

endmodule
