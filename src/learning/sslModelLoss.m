function [loss, gEnc, gW, gInv, stats] = sslModelLoss(encNet, keyNet, W, invNet, X1, X2, Xn, Y, curlWeight, invWeight, needGrad)
%SSLMODELLOSS  Self-supervised losses for the CNN encoder (call through dlfeval for gradients).
%
%   CURL contrastive loss  : anchors from encNet(X1), keys from the momentum
%                            encoder keyNet(X2) (two augmentations of the same
%                            stacks); logits = z_a' * W * z_k; InfoNCE with the
%                            matching pair on the diagonal.
%   Inverse dynamics loss  : predict the executed action (one-hot Y) from
%                            [z_t; z_t+1].
if nargin < 11
    needGrad = true;
end
za = forward(encNet, X1);                         % D x B, "CB"
zk = predict(keyNet, X2);
zk = dlarray(extractdata(zk));                    % constant keys (no gradient)
logits = stripdims(za)' * (W * zk);              % B x B
logits = logits - max(logits, [], 2);
B = size(logits, 1);
I = eye(B, "like", extractdata(logits));
lse = log(sum(exp(logits), 2));
pos = sum(logits .* I, 2);
curlLoss = mean(lse - pos);

zn = forward(encNet, Xn);
logitsI = stripdims(forward(invNet, cat(1, za, zn)));   % nActions x B
m = max(logitsI, [], 1);
lseI = m + log(sum(exp(logitsI - m), 1));
invLoss = -mean(sum(Y .* (logitsI - lseI), 1));

loss = curlWeight * curlLoss + invWeight * invLoss;
if needGrad
    [gEnc, gW, gInv] = dlgradient(loss, encNet.Learnables, W, invNet.Learnables);
else
    gEnc = [];
    gW = [];
    gInv = [];
end

[~, am] = max(extractdata(logits), [], 2);
[~, ai] = max(extractdata(logitsI), [], 1);
[~, yi] = max(Y, [], 1);
stats = struct( ...
    "loss", double(gather(extractdata(loss))), ...
    "curlLoss", double(gather(extractdata(curlLoss))), ...
    "curlAcc", mean(double(gather(am(:))) == (1:B)'), ...
    "invLoss", double(gather(extractdata(invLoss))), ...
    "invAcc", mean(double(gather(ai(:))) == double(gather(yi(:)))));
end
