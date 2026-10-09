module accumulator (
    input wire clk,
    input wire reset,
    input wire [7:0] value,
    output reg [7:0] total
);
    wire [7:0] next_total;
    adder #(.WIDTH(8)) update_sum (.a(total), .b(value), .sum(next_total));
    always @(posedge clk) begin
        if (reset)
            total <= 8'd0;
        else
            total <= next_total;
    end
endmodule
