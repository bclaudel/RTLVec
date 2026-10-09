module select_word #(parameter WIDTH = 8) (
    input wire select,
    input wire [WIDTH-1:0] a,
    input wire [WIDTH-1:0] b,
    output wire [WIDTH-1:0] value
);
    assign value = select ? a : b;
endmodule
