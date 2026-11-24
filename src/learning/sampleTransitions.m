function [X, Xn, A] = sampleTransitions(ds, batchSize, rs, split)
%SAMPLETRANSITIONS  Random (stack_t, stack_t+1, action_t) triples for self-supervised learning.
if nargin < 4
    split = "train";
end
if split == "train"
    pool = ds.transTrain;
else
    pool = ds.transVal;
end
idx = pool(randi(rs, numel(pool), batchSize, 1));
X = stackFrames(ds, idx);
Xn = stackFrames(ds, idx + 1);
A = ds.actions(idx);
end
