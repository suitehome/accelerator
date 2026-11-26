`timescale 1ns/1ps
// ============================================================================
// mlp_accel : INT8 MLP inference accelerator, output-stationary, P MAC lanes
//
//   * One MAC array (P lanes) is reused for every layer by the FSM.
//   * Each MAC cycle reads one input activation (broadcast to all lanes) and
//     one weight ROM word holding P int8 weights -> no bandwidth bottleneck.
//   * After IN_DIM MAC cycles, the P accumulators are requantized serially
//     through ONE requant unit (area vs. latency trade-off, see README).
//   * Hidden layers: ReLU + clamp [0,127]. Last layer: clamp [-128,127].
//   * argmax with strict '>' : on a tie the first index wins.
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
    localparam [2:0] S_IDLE = 3'd0, S_MAC = 3'd1, S_REQ = 3'd2, S_ARGMAX = 3'd3;
    reg [2:0]  state;
    reg [7:0]  layer;
    reg [7:0]  grp;
    reg [7:0]  idx;          // input index (MAC), lane index (REQ), output index (ARGMAX)
    reg [15:0] w_ptr;        // weight ROM address, walks the ROM linearly
    reg        src_is_a;     // 1: read bufA / write bufB, 0: the reverse

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

    // ------------------------------------------------------ requant (serial)
    wire [7:0]         o_idx   = grp * P + idx;
    wire               o_valid = (o_idx < out_dim);
    wire signed [31:0] acc_sel = acc_flat[32*idx +: 32];
    wire signed [31:0] bias    = brom[b_base + o_idx];
    wire signed [32:0] req_v   = acc_sel + bias;                 // 33 bit: no overflow
    wire signed [16:0] m0_s    = {1'b0, m0};                     // force positive
    wire signed [49:0] prod_m  = req_v * m0_s;
    wire signed [49:0] rnd     = 50'sd1 <<< (nsh - 8'd1);        // 2^(n-1)
    wire signed [49:0] shifted = (prod_m + rnd) >>> nsh;         // round half up
    wire signed [49:0] lo      = relu ? 50'sd0 : -50'sd128;
    wire signed [7:0]  req_y   = (shifted > 50'sd127) ? 8'sd127 :
                                 (shifted < lo)       ? lo[7:0] :
                                                        shifted[7:0];
    wire               req_fire = (state == S_REQ) && o_valid;   // debug / trace
    wire               grp_last_out = (idx == P - 1) || (o_idx == out_dim - 8'd1);

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
            busy      <= 1'b0;
            done      <= 1'b0;
            class_out <= 8'd0;
            best_v    <= 8'sd0;
            best_i    <= 8'd0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (in_we)
                        bufA[in_addr] <= in_data;
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

                S_REQ: begin
                    if (o_valid) begin
                        if (src_is_a) bufB[o_idx] <= req_y;
                        else          bufA[o_idx] <= req_y;
                    end
                    if (grp_last_out) begin
                        idx <= 8'd0;
                        if (grp == n_grp - 8'd1) begin            // layer finished
                            grp      <= 8'd0;
                            src_is_a <= ~src_is_a;
                            if (layer == NL - 1)
                                state <= S_ARGMAX;
                            else begin
                                layer <= layer + 8'd1;
                                state <= S_MAC;
                            end
                        end else begin
                            grp   <= grp + 8'd1;
                            state <= S_MAC;
                        end
                    end else begin
                        idx <= idx + 8'd1;
                    end
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
