function X = stackFrames(ds, idx)
%STACKFRAMES  Rebuild H x W x K x B single frame stacks (values in [0,1]) for frame indices idx.
idx = idx(:);
k = ds.frameStack;
offs = -(k - 1):0;
cand = max(idx + offs, ds.epStart(idx));            % B x k, never cross an episode start
[H, W, ~] = size(ds.frames);
X = ds.frames(:, :, reshape(cand', [], 1));          % oldest -> newest for each sample
X = reshape(single(X) / 255, H, W, k, numel(idx));
end
