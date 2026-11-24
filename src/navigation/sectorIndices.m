function idx = sectorIndices(rows, cols, rowGroup, colGroup)
%SECTORINDICES  Linear sector indices (row-major, 1-based) of a group of image sectors.
%
%   rowGroup: "mid" | "top" | "bottom"
%   colGroup: "center" | "left" | "right" | "center_left" | "center_right"
%   With the default 3 x 5 grid, the middle row is row 2 and "center" spans
%   columns 2-4 (the part of the view the UAV flies into when going straight).

mid = floor(rows / 2) + 1;
switch rowGroup
    case "mid"
        r = mid;
    case "top"
        r = 1:mid-1;
    case "bottom"
        r = mid+1:rows;
    otherwise
        error("sectorIndices:rowGroup", "Unknown row group %s.", rowGroup);
end
if isempty(r)
    r = mid;
end

c = floor(cols / 2) + 1;
half = max(1, floor(cols / 4));
switch colGroup
    case "center"
        span = [c - half, c + half];
    case "left"
        span = [c - 2*half, c];
    case "right"
        span = [c, c + 2*half];
    case "center_left"
        span = [c - half, c];
    case "center_right"
        span = [c, c + half];
    otherwise
        error("sectorIndices:colGroup", "Unknown column group %s.", colGroup);
end
cc = max(1, span(1)):min(cols, span(2));
[R, C] = ndgrid(r, cc);
idx = (R(:) - 1) * cols + C(:);
end
