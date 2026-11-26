`timescale 1ns/1ps
// Testbench: loads NS input vectors from inputs.hex, runs each inference and
// writes
//   rtl_results.txt : "<sample> <class> <logit0> ... <logitK-1> <cycles>"
//   rtl_trace.txt   : "<sample> <layer> <o> <v> <y>" for every requantized neuron
// run_sim.py compares both files against the golden model.
module tb;
    parameter integer P        = 4;
    parameter integer NL       = 2;
    parameter integer MAX_DIM  = 32;
    parameter integer W_DEPTH  = 160;
    parameter integer B_DEPTH  = 20;
    parameter integer IN0      = 32;
    parameter integer OUT_LAST = 4;
    parameter integer NS       = 10;
    parameter integer TIMEOUT  = 100000;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg              rst_n   = 1'b0;
    reg              in_we   = 1'b0;
    reg  [7:0]       in_addr = 8'd0;
    reg  signed [7:0] in_data = 8'sd0;
    reg              start   = 1'b0;
    wire             busy, done;
    wire [7:0]       class_out;

    mlp_accel #(
        .P(P), .NL(NL), .MAX_DIM(MAX_DIM), .W_DEPTH(W_DEPTH), .B_DEPTH(B_DEPTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .in_we(in_we), .in_addr(in_addr), .in_data(in_data),
        .start(start), .busy(busy), .done(done), .class_out(class_out)
    );

    reg [7:0] inmem [0:NS*IN0-1];
    integer fres, ftr, s, i, cyc, cur_sample;
    reg signed [7:0] logit;

    initial begin
        $readmemh("inputs.hex", inmem);
        fres = $fopen("rtl_results.txt", "w");
        ftr  = $fopen("rtl_trace.txt", "w");
        cur_sample = 0;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        for (s = 0; s < NS; s = s + 1) begin
            cur_sample = s;
            for (i = 0; i < IN0; i = i + 1) begin
                @(negedge clk);
                in_we   = 1'b1;
                in_addr = i;
                in_data = inmem[s*IN0 + i];
            end
            @(negedge clk);
            in_we = 1'b0;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            cyc   = 1;
            while (!done) begin
                @(negedge clk);
                cyc = cyc + 1;
                if (cyc > TIMEOUT) begin
                    $display("ERROR: timeout on sample %0d", s);
                    $finish;
                end
            end
            $fwrite(fres, "%0d %0d", s, class_out);
            for (i = 0; i < OUT_LAST; i = i + 1) begin
                logit = dut.src_is_a ? dut.bufA[i] : dut.bufB[i];
                $fwrite(fres, " %0d", logit);
            end
            $fwrite(fres, " %0d\n", cyc);
        end
        $fclose(fres);
        $fclose(ftr);
        $finish;
    end

    // trace every requantized output (values are stable at the falling edge)
    always @(negedge clk) begin
        if (dut.wr_en)
            $fwrite(ftr, "%0d %0d %0d %0d %0d\n",
                    cur_sample, dut.req_layer, dut.req_o, dut.req_vt, dut.req_y);
    end
endmodule
