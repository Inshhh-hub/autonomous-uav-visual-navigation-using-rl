function encNet = pretrainSSL(cfg)
%PRETRAINSSL  Step 2 - self-supervised pre-training of the CNN encoder.
%
%   CURL (contrastive learning with a momentum key encoder) + inverse
%   dynamics, trained on unlabelled rollouts with a custom training loop.
%   Saves the encoder (with the lowest validation loss) to cfg.paths.encoder.

s = cfg.ssl;
rng(cfg.seed);
if ~isfile(cfg.paths.rollouts)
    error("pretrainSSL:noData", "%s not found - run collectData(cfg) first.", cfg.paths.rollouts);
end
data = load(cfg.paths.rollouts);
ds = makeDataset(data, cfg.env.camera.frameStack, s.valFraction, cfg.seed);
clear data
fprintf("[ssl] %d frames, %d train / %d val transitions\n", ds.numFrames, numel(ds.transTrain), numel(ds.transVal));

D = cfg.model.featureDim;
nA = ds.nActions;
encNet = buildEncoderNetwork(cfg);
keyNet = encNet;
invNet = buildInverseNetwork(D, nA);
useGPU = cfg.useGPU && canUseGPU;
W0 = eye(D, "single");
if useGPU
    W0 = gpuArray(W0);
end
W = dlarray(W0);
photometric = cfg.env.camera.obsMode == "gray";
rs = RandStream("mt19937ar", "Seed", cfg.seed + 11);

avgE = []; sqE = []; avgW = []; sqW = []; avgI = []; sqI = [];
history = zeros(s.iterations, 5);
bestVal = inf;
ensureFolder(cfg.paths.encoder);
for it = 1:s.iterations
    [X, Xn, A] = sampleTransitions(ds, s.batchSize, rs, "train");
    Y = oneHot(A, nA);
    X1 = toDl(augmentBatch(X, s.shiftPad, photometric, rs), useGPU);
    X2 = toDl(augmentBatch(X, s.shiftPad, photometric, rs), useGPU);
    Xn1 = toDl(augmentBatch(Xn, s.shiftPad, photometric, rs), useGPU);
    if useGPU
        Y = gpuArray(Y);
    end
    [~, gE, gW, gI, st] = dlfeval(@sslModelLoss, encNet, keyNet, W, invNet, X1, X2, Xn1, Y, ...
        s.curlWeight, s.inverseWeight, true);
    [encNet, avgE, sqE] = adamupdate(encNet, gE, avgE, sqE, it, s.learnRate);
    [W, avgW, sqW] = adamupdate(W, gW, avgW, sqW, it, s.learnRate);
    [invNet, avgI, sqI] = adamupdate(invNet, gI, avgI, sqI, it, s.learnRate);
    keyNet = momentumUpdate(keyNet, encNet, s.momentumTau);
    history(it, :) = [st.loss, st.curlLoss, st.curlAcc, st.invLoss, st.invAcc];

    if mod(it, s.logEvery) == 0
        fprintf("[ssl] it %5d  loss %.3f  curl %.3f (acc %.2f)  inverse %.3f (acc %.2f)\n", ...
            it, mean(history(it-s.logEvery+1:it, :), 1));
    end
    if mod(it, s.evalEvery) == 0 || it == s.iterations
        val = evaluateSSL(encNet, keyNet, W, invNet, ds, s, photometric, useGPU, nA);
        fprintf("[ssl] validation  loss %.3f  curl acc %.2f  inverse acc %.2f\n", val.loss, val.curlAcc, val.invAcc);
        if val.loss < bestVal
            bestVal = val.loss;
            encoder = gatherNetwork(encNet); %#ok<NASGU>
            sslHistory = history(1:it, :); %#ok<NASGU>
            saveCfg = cfg; %#ok<NASGU>
            save(cfg.paths.encoder, "encoder", "sslHistory", "val", "saveCfg");
        end
    end
end
S = load(cfg.paths.encoder, "encoder");
encNet = S.encoder;
fprintf("[ssl] best validation loss %.4f - encoder saved to %s\n", bestVal, cfg.paths.encoder);
end

function X = toDl(X, useGPU)
if useGPU
    X = gpuArray(X);
end
X = dlarray(X, "SSCB");
end

function keyNet = momentumUpdate(keyNet, encNet, tau)
K = keyNet.Learnables.Value;
E = encNet.Learnables.Value;
vals = cellfun(@(k, e) (1 - tau) * k + tau * e, K, E, "UniformOutput", false);
keyNet = setLearnableValues(keyNet, vals);
end

function val = evaluateSSL(encNet, keyNet, W, invNet, ds, s, photometric, useGPU, nA)
rs = RandStream("mt19937ar", "Seed", 123);
nBatches = max(1, min(20, floor(numel(ds.transVal) / s.batchSize)));
acc = zeros(nBatches, 3);
for b = 1:nBatches
    [X, Xn, A] = sampleTransitions(ds, s.batchSize, rs, "val");
    Y = oneHot(A, nA);
    if useGPU
        Y = gpuArray(Y);
    end
    [~, ~, ~, ~, st] = sslModelLoss(encNet, keyNet, W, invNet, ...
        toDl(augmentBatch(X, s.shiftPad, photometric, rs), useGPU), ...
        toDl(augmentBatch(X, s.shiftPad, photometric, rs), useGPU), ...
        toDl(augmentBatch(Xn, s.shiftPad, photometric, rs), useGPU), Y, ...
        s.curlWeight, s.inverseWeight, false);
    acc(b, :) = [st.loss, st.curlAcc, st.invAcc];
end
m = mean(acc, 1);
val = struct("loss", m(1), "curlAcc", m(2), "invAcc", m(3));
end
