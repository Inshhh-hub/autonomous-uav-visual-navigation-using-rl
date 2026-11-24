function out = sectorMinDepth(depth, rows, cols)
%SECTORMINDEPTH  Nearest obstacle distance in each cell of a rows x cols image grid.
%   Output is a column vector ordered row-major: index = (r-1)*cols + c,
%   r = 1 is the top row and c = 1 the left column.

[H, W] = size(depth);
re = round(linspace(0, H, rows + 1));
ce = round(linspace(0, W, cols + 1));
out = zeros(rows * cols, 1);
for r = 1:rows
    for c = 1:cols
        block = depth(re(r)+1:re(r+1), ce(c)+1:ce(c+1));
        out((r - 1) * cols + c) = min(block(:));
    end
end
end
