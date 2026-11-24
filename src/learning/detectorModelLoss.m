function [loss, grad, stats, Y] = detectorModelLoss(net, X, occ, distN, risk, w, needGrad)
%DETECTORMODELLOSS  Multi-task loss of the CNN obstacle detector.
%   occ   - S x B occupancy labels (0/1)
%   distN - S x B nearest-obstacle distance / maxDepth
%   risk  - 1 x B crash-within-horizon labels (0/1)
%   loss = BCE(occupancy) + distWeight * SmoothL1(distance) + riskWeight * weightedBCE(risk)
if nargin < 7
    needGrad = true;
end
Y = stripdims(forward(net, X));                  % (2S+1) x B
nS = size(occ, 1);
zo = Y(1:nS, :);
zd = Y(nS+1:2*nS, :);
zr = Y(end, :);

occLoss = mean(bceWithLogits(zo, occ), "all");
err = sigmoid(zd) - distN;
distLoss = mean(smoothL1(err, 0.05), "all");
riskLoss = mean(weightedBce(zr, risk, w.riskPosWeight), "all");
loss = occLoss + w.distWeight * distLoss + w.riskWeight * riskLoss;

if needGrad
    grad = dlgradient(loss, net.Learnables);
else
    grad = [];
end
stats = struct("loss", double(gather(extractdata(loss))), ...
    "occLoss", double(gather(extractdata(occLoss))), ...
    "distLoss", double(gather(extractdata(distLoss))), ...
    "riskLoss", double(gather(extractdata(riskLoss))));
end

function l = bceWithLogits(z, y)
l = max(z, 0) - z .* y + log(1 + exp(-abs(z)));
end

function l = weightedBce(z, y, posWeight)
% PyTorch-style BCEWithLogits with pos_weight
logw = 1 + (posWeight - 1) .* y;
l = (1 - y) .* z + logw .* (log(1 + exp(-abs(z))) + max(-z, 0));
end

function l = smoothL1(e, beta)
a = abs(e);
small = a < beta;
l = (0.5 * e.^2 / beta) .* small + (a - 0.5 * beta) .* (~small);
end
