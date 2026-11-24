function detNet = trainDetector(cfg)
%TRAINDETECTOR  Step 3 - train the CNN obstacle detector on automatically labelled frames.
%
%   The convolutional layers start from the self-supervised encoder (if
%   cfg.paths.encoder exists). Saves the network with the lowest validation
%   loss plus its metrics (occupancy accuracy/F1, distance MAE, risk AUC).

d = cfg.detector;
rng(cfg.seed);
if ~isfile(cfg.paths.rollouts)
    error("trainDetector:noData", "%s not found - run collectData(cfg) first.", cfg.paths.rollouts);
end
data = load(cfg.paths.rollouts);
ds = makeDataset(data, cfg.env.camera.frameStack, d.valFraction, cfg.seed);
clear data

detNet = buildDetectorNetwork(cfg);
encLayers = ["conv1" "conv2" "conv3" "conv4"];
if isfile(cfg.paths.encoder)
    S = load(cfg.paths.encoder, "encoder");
    [detNet, n] = copyLearnables(detNet, S.encoder);
    fprintf("[detector] %d parameter tensors initialised from the self-supervised encoder\n", n);
else
    fprintf("[detector] no self-supervised encoder found - training the CNN from scratch\n");
end
useGPU = cfg.useGPU && canUseGPU;
photometric = cfg.env.camera.obsMode == "gray";
rs = RandStream("mt19937ar", "Seed", cfg.seed + 21);
w = struct("distWeight", d.distWeight, "riskWeight", d.riskWeight, "riskPosWeight", d.riskPosWeight);
fprintf("[detector] %d train / %d val frames, risk-positive rate %.1f%%\n", ...
    numel(ds.trainIdx), numel(ds.valIdx), 100 * mean(ds.risk));

avg = []; sq = [];
history = zeros(d.iterations, 4);
best = inf;
ensureFolder(cfg.paths.detector);
for it = 1:d.iterations
    idx = ds.trainIdx(randi(rs, numel(ds.trainIdx), d.batchSize, 1));
    X = augmentBatch(stackFrames(ds, idx), 0, photometric, rs);
    [occ, distN, risk] = frameLabels(ds, idx);
    if useGPU
        X = gpuArray(X); occ = gpuArray(occ); distN = gpuArray(distN); risk = gpuArray(risk);
    end
    [~, grad, st] = dlfeval(@detectorModelLoss, detNet, dlarray(X, "SSCB"), occ, distN, risk, w, true);
    if d.freezeEncoder
        frozen = ismember(string(grad.Layer), encLayers);
        grad.Value(frozen) = cellfun(@(g) 0 * g, grad.Value(frozen), "UniformOutput", false);
    end
    [detNet, avg, sq] = adamupdate(detNet, grad, avg, sq, it, d.learnRate);
    history(it, :) = [st.loss, st.occLoss, st.distLoss, st.riskLoss];

    if mod(it, d.logEvery) == 0
        fprintf("[detector] it %5d  loss %.4f  occ %.4f  dist %.4f  risk %.4f\n", ...
            it, mean(history(it-d.logEvery+1:it, :), 1));
    end
    if mod(it, d.evalEvery) == 0 || it == d.iterations
        metrics = evaluateDetector(detNet, ds, w, useGPU);
        fprintf("[detector] validation  loss %.4f  occ acc %.3f  occ F1 %.3f  dist MAE %.2f m  risk AUC %.3f\n", ...
            metrics.loss, metrics.occAccuracy, metrics.occF1, metrics.distMAE, metrics.riskAUC);
        if metrics.loss < best
            best = metrics.loss;
            detector = gatherNetwork(detNet); %#ok<NASGU>
            detHistory = history(1:it, :); %#ok<NASGU>
            save(cfg.paths.detector, "detector", "metrics", "detHistory");
        end
    end
end
S = load(cfg.paths.detector, "detector", "metrics");
detNet = S.detector;
fprintf("[detector] saved to %s (best validation loss %.4f)\n", cfg.paths.detector, best);
end

function m = evaluateDetector(net, ds, w, useGPU)
idx = ds.valIdx(1:min(numel(ds.valIdx), 4000));
bs = 256;
occP = []; occY = []; distP = []; distY = []; riskP = []; riskY = [];
lossSum = 0;
for i = 1:bs:numel(idx)
    b = idx(i:min(i + bs - 1, numel(idx)));
    X = stackFrames(ds, b);
    [occ, distN, risk] = frameLabels(ds, b);
    if useGPU
        X = gpuArray(X);
    end
    [~, ~, st, Y] = detectorModelLoss(net, dlarray(X, "SSCB"), occ, distN, risk, w, false);
    Y = double(gather(extractdata(Y)));
    nS = size(occ, 1);
    lossSum = lossSum + st.loss * numel(b);
    occP = [occP, 1 ./ (1 + exp(-Y(1:nS, :)))]; %#ok<AGROW>
    distP = [distP, 1 ./ (1 + exp(-Y(nS+1:2*nS, :)))]; %#ok<AGROW>
    riskP = [riskP, 1 ./ (1 + exp(-Y(end, :)))]; %#ok<AGROW>
    occY = [occY, double(occ)]; %#ok<AGROW>
    distY = [distY, double(distN)]; %#ok<AGROW>
    riskY = [riskY, double(risk)]; %#ok<AGROW>
end
pred = occP > 0.5;
truth = occY > 0.5;
tp = sum(pred & truth, "all");
fp = sum(pred & ~truth, "all");
fn = sum(~pred & truth, "all");
m = struct();
m.loss = lossSum / numel(idx);
m.occAccuracy = mean(pred == truth, "all");
m.occF1 = 2 * tp / max(2 * tp + fp + fn, 1);
m.distMAE = mean(abs(distP - distY), "all") * ds.maxDepth;
m.riskAUC = rocAuc(riskY > 0.5, riskP);
end
