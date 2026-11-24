function X = augmentBatch(X, pad, photometric, rs)
%AUGMENTBATCH  Data augmentation for self-supervised learning (H x W x K x B, values in [0,1]).
%   * photometric: random per-sample contrast / brightness jitter
%   * random shift: replicate-pad by `pad` pixels and crop back at a random offset
%     (the DrQ / CURL augmentation)
[H, W, K, B] = size(X);
if photometric
    c = 1 + (rand(rs, 1, 1, 1, B) * 2 - 1) * 0.2;
    b = (rand(rs, 1, 1, 1, B) * 2 - 1) * 0.1;
    m = mean(X, [1 2 3]);
    X = min(max((X - m) .* c + m + b, 0), 1);
end
if pad > 0
    ri = [ones(1, pad), 1:H, H * ones(1, pad)];
    ci = [ones(1, pad), 1:W, W * ones(1, pad)];
    Xp = X(ri, ci, :, :);
    shift = randi(rs, [0, 2 * pad], 2, B);
    out = zeros(H, W, K, B, "like", X);
    for i = 1:B
        out(:, :, :, i) = Xp(shift(1, i) + (1:H), shift(2, i) + (1:W), :, i);
    end
    X = out;
end
end
