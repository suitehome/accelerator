`timescale 1ns/1ps
// ============================================================================
// mlp_accel : INT8 MLP inference accelerator, output-stationary, P MAC lanes
//             with a PIPELINED requantization unit.
//
//   * One MAC array (P lanes) is reused for every layer by the FSM.
//   * Each MAC cycle reads one input activation (broadcast to all lanes) and
//     one weight ROM word holding P int8 weights -> no bandwidth bottleneck.
//   * After IN_DIM MAC cycles, the P accumulators are issued one per cycle
//     into a 5-stage requantization pipeline (one shared unit):
//        R1  v   = acc + bias                      (33 bit)
//        R2  p   = v * M0                          (50 bit, DSP)
//        R3  p   registered again (DSP output register)
//        R4  s   = p + 2^(n-1)                     (round half up)
//        R5  y   = clamp(s >>> n), write to buffer
//     The next group's MAC starts right after the last issue (the
//     accumulators were already captured). At a layer boundary the FSM waits
//     PIPE = 4 cycles (S_DRAIN) so every result is written before the next
//     layer reads it.
//   * Hidden layers: ReLU + clamp [0,127]. Last layer: clamp [-128,127].
//   * argmax with strict '>' : on a tie the first index wins.
//
// The arithmetic is identical to rtl/mlp_accel_comb.v (single-cycle
// requantization); only the timing changes: +PIPE cycles per layer.
//
// Memory files (written by golden/qmlp.py:export_hw):
//   W_FILE  : W_DEPTH words of 8*P bits, contiguous over all layers
//   B_FILE  : B_DEPTH words of 32 bits (int32 bias)
//   CFG_FILE: NL words of 96 bits, fields documented in export_hw()
// ============================================================================
module mlp_accel #(
    parameter integer P        = 4,
    parameter integer NL       = 2,
    parameter integer MAX_DIM  = 32,
    parameter integer W_DEPTH  = 160,
    parameter integer B_DEPTH  = 20,
    parameter         W_FILE   = "wrom.hex",
    parameter         B_FILE   = "brom.hex",
    parameter         CFG_FILE = "cfg.hex"
)(
    input  wire              clk,
    input  wire              rst_n,
    // input activation load port (write while idle)
    input  wire              in_we,
    input  wire [7:0]        in_addr,
    input  wire signed [7:0] in_data,
    // control
    input  wire              start,
    output reg               busy,
    output reg               done,        // one-cycle pulse
    output reg  [7:0]        class_out
);
    localparam integer PIPE = 4;          // cycles from the last issue to the last buffer write

    // ---------------------------------------------------------------- ROMs
    reg [8*P-1:0] wrom [0:W_DEPTH-1];
    reg [31:0]    brom [0:B_DEPTH-1];
    reg [95:0]    cfg  [0:NL-1];
    initial begin
        $readmemh(W_FILE,   wrom);
        $readmemh(B_FILE,   brom);
        $readmemh(CFG_FILE, cfg);
    end

    // ------------------------------------------------- activation buffers
    reg signed [7:0] bufA [0:MAX_DIM-1];
    reg signed [7:0] bufB [0:MAX_DIM-1];

    // ------------------------------------------------------------ FSM state
    localparam [2:0] S_IDLE = 3'd0, S_MAC = 3'd1, S_REQ = 3'd2, S_DRAIN = 3'd3, S_ARGMAX = 3'd4;
    reg [2:0]  state;
    reg [7:0]  layer;
    reg [7:0]  grp;
    reg [7:0]  idx;          // input index (MAC), lane index (REQ), output index (ARGMAX)
    reg [15:0] w_ptr;        // weight ROM address, walks the ROM linearly
    reg        src_is_a;     // 1: read bufA / write bufB, 0: the reverse
    reg [2:0]  drain_cnt;

    // ------------------------------------------------- current layer config
    wire [95:0] cw      = cfg[layer];
    wire [7:0]  in_dim  = cw[7:0];
    wire [7:0]  out_dim = cw[15:8];
    wire [7:0]  n_grp   = cw[23:16];
    wire [15:0] b_base  = cw[55:40];
    wire [15:0] m0      = cw[71:56];
    wire [7:0]  nsh     = cw[79:72];
    wire        relu    = cw[80];

    // ------------------------------------------------------------ MAC array
    wire signed [7:0] xval  = src_is_a ? bufA[idx] : bufB[idx];
    wire [8*P-1:0]    wword = wrom[w_ptr];
    wire              mac_en = (state == S_MAC);
    wire              first_in = (idx == 8'd0);

    wire [32*P-1:0] acc_flat;
    genvar p;
    generate
        for (p = 0; p < P; p = p + 1) begin : lane
            wire signed [7:0]  w    = wword[8*p +: 8];
            wire signed [15:0] prod = w * xval;
            reg  signed [31:0] acc;
            always @(posedge clk) begin
                if (mac_en)
                    acc <= (first_in ? 32'sd0 : acc) + prod;   // clear on first input
            end
            assign acc_flat[32*p +: 32] = acc;
        end
    endgenerate

    // ------------------------------------------------ requant: issue (R1)
    wire [7:0]         o_idx    = grp * P + idx;
    wire               o_valid  = (o_idx < out_dim);
    wire signed [31:0] acc_sel  = acc_flat[32*idx +: 32];
    wire signed [31:0] bias     = brom[b_base + o_idx];
    wire signed [32:0] req_v    = acc_sel + bias;                // 33 bit: no overflow
    wire               issue    = (state == S_REQ) && o_valid;
    wire               grp_last_out = (idx == P - 1) || (o_idx == out_dim - 8'd1);

    // pipeline registers; control travels with the data
    reg               r1_vld, r2_vld, r3_vld, r4_vld;
    reg signed [32:0] r1_v;
    reg signed [49:0] r2_p, r3_p, r4_s;
    reg [7:0]         r1_o, r2_o, r3_o, r4_o;
    reg [7:0]         r1_n, r2_n, r3_n, r4_n;
    reg               r1_relu, r2_relu, r3_relu, r4_relu;
    reg               r1_tob, r2_tob, r3_tob, r4_tob;      // 1: write bufB
    reg [15:0]        r1_m0;
    // debug only (trace in the testbench), removed by synthesis
    reg [7:0]         r1_l, r2_l, r3_l, r4_l;
    reg signed [32:0] r2_v, r3_v, r4_v;

    wire signed [16:0] m0_s = {1'b0, r1_m0};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r1_vld <= 1'b0; r2_vld <= 1'b0; r3_vld <= 1'b0; r4_vld <= 1'b0;
        end else begin
            r1_vld <= issue;  r2_vld <= r1_vld; r3_vld <= r2_vld; r4_vld <= r3_vld;
        end
    end
    always @(posedge clk) begin
        // R1: accumulator + bias
        r1_v <= req_v;  r1_o <= o_idx;  r1_n <= nsh;  r1_relu <= relu;
        r1_tob <= src_is_a;  r1_m0 <= m0;  r1_l <= layer;
        // R2: multiply by M0
        r2_p <= r1_v * m0_s;  r2_o <= r1_o;  r2_n <= r1_n;  r2_relu <= r1_relu;
        r2_tob <= r1_tob;  r2_l <= r1_l;  r2_v <= r1_v;
        // R3: product register (absorbed into the DSP output register)
        r3_p <= r2_p;  r3_o <= r2_o;  r3_n <= r2_n;  r3_relu <= r2_relu;
        r3_tob <= r2_tob;  r3_l <= r2_l;  r3_v <= r2_v;
        // R4: add the rounding constant 2^(n-1)
        r4_s <= r3_p + (50'sd1 <<< (r3_n - 8'd1));
        r4_o <= r3_o;  r4_n <= r3_n;  r4_relu <= r3_relu;
        r4_tob <= r3_tob;  r4_l <= r3_l;  r4_v <= r3_v;
    end

    // R5: arithmetic shift, saturate, write
    wire signed [49:0] shifted = r4_s >>> r4_n;
    wire signed [49:0] lo      = r4_relu ? 50'sd0 : -50'sd128;
    wire signed [7:0]  req_y   = (shifted > 50'sd127) ? 8'sd127 :
                                 (shifted < lo)       ? lo[7:0] :
                                                        shifted[7:0];
    wire               wr_en   = r4_vld;
    // aliases used by the testbench trace
    wire [7:0]         req_layer = r4_l;
    wire [7:0]         req_o     = r4_o;
    wire signed [32:0] req_vt    = r4_v;

    // ------------------------------------------------ buffer write ports
    always @(posedge clk) begin
        if (wr_en && !r4_tob)
            bufA[r4_o] <= req_y;
        else if (in_we && state == S_IDLE)
            bufA[in_addr] <= in_data;
        if (wr_en && r4_tob)
            bufB[r4_o] <= req_y;
    end

    // ------------------------------------------------------------- argmax
    reg signed [7:0] best_v;
    reg [7:0]        best_i;
    wire             take = first_in || (xval > best_v);          // strict >
    wire [7:0]       nxt_best_i = take ? idx : best_i;

    // ----------------------------------------------------------- main FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            layer     <= 8'd0;
            grp       <= 8'd0;
            idx       <= 8'd0;
            w_ptr     <= 16'd0;
            src_is_a  <= 1'b1;
            drain_cnt <= 3'd0;
            busy      <= 1'b0;
            done      <= 1'b0;
            class_out <= 8'd0;
            best_v    <= 8'sd0;
            best_i    <= 8'd0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        state    <= S_MAC;
                        layer    <= 8'd0;
                        grp      <= 8'd0;
                        idx      <= 8'd0;
                        w_ptr    <= 16'd0;
                        src_is_a <= 1'b1;
                        busy     <= 1'b1;
                    end
                end

                S_MAC: begin
                    w_ptr <= w_ptr + 16'd1;
                    if (idx == in_dim - 8'd1) begin
                        idx   <= 8'd0;
                        state <= S_REQ;
                    end else begin
                        idx <= idx + 8'd1;
                    end
                end

                S_REQ: begin                                      // issue one output per cycle
                    if (grp_last_out) begin
                        idx <= 8'd0;
                        if (grp == n_grp - 8'd1) begin            // last group of the layer
                            drain_cnt <= PIPE[2:0];
                            state     <= S_DRAIN;
                        end else begin
                            grp   <= grp + 8'd1;
                            state <= S_MAC;
                        end
                    end else begin
                        idx <= idx + 8'd1;
                    end
                end

                S_DRAIN: begin                                    // wait for the last writes
                    if (drain_cnt == 3'd1) begin
                        grp      <= 8'd0;
                        src_is_a <= ~src_is_a;
                        if (layer == NL - 1)
                            state <= S_ARGMAX;
                        else begin
                            layer <= layer + 8'd1;
                            state <= S_MAC;
                        end
                    end
                    drain_cnt <= drain_cnt - 3'd1;
                end

                S_ARGMAX: begin                                   // src now holds logits
                    if (take) begin
                        best_v <= xval;
                        best_i <= idx;
                    end
                    if (idx == out_dim - 8'd1) begin
                        class_out <= nxt_best_i;
                        done      <= 1'b1;
                        busy      <= 1'b0;
                        idx       <= 8'd0;
                        state     <= S_IDLE;
                    end else begin
                        idx <= idx + 8'd1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
