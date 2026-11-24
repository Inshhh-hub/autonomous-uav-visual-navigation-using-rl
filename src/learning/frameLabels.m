function [occ, distN, risk] = frameLabels(ds, idx)
%FRAMELABELS  Automatically generated detector labels for frame indices idx.
%   occ   S x B  - sector has an obstacle closer than the occupancy threshold
%   distN S x B  - nearest obstacle distance / maxDepth
%   risk  1 x B  - crash within the risk horizon
d = ds.sectorDist(:, idx(:));
occ = single(d < ds.occupancyThreshold);
distN = single(min(max(d / ds.maxDepth, 0), 1));
risk = single(ds.risk(idx(:)'));
end
