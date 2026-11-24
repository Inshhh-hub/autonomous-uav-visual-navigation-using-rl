function tests = testLearning
%TESTLEARNING  Tests of the networks, losses, data pipeline and DQN agent (needs RL + DL Toolboxes).
tests = functiontests(localfunctions);
end

function setupOnce(tc)
addpath(fileparts(fileparts(mfilename("fullpath"))));
setupProject();
tc.TestData.tmp = string(tempname);
mkdir(tc.TestData.tmp);
end

function teardownOnce(tc)
if isfolder(tc.TestData.tmp)
    rmdir(tc.TestData.tmp, "s");
end
end

function cfg = tinyConfig(tc)
cfg = quickConfig();
cfg.env.maxSteps = 40;
cfg.model.featureDim = 64;
cfg.model.hiddenDim = 64;
cfg.collect.steps = 300;
cfg.ssl.iterations = 4;
cfg.ssl.batchSize = 8;
cfg.ssl.logEvery = 2;
cfg.ssl.evalEvery = 2;
cfg.detector.iterations = 4;
cfg.detector.batchSize = 8;
cfg.detector.logEvery = 2;
cfg.detector.evalEvery = 2;
cfg.dqn.maxEpisodes = 2;
cfg.dqn.bufferLength = 500;
cfg.dqn.miniBatchSize = 8;
cfg.dqn.plots = "none";
cfg.eval.episodes = 1;
cfg.paths.rollouts = fullfile(tc.TestData.tmp, "rollouts.mat");
cfg.paths.encoder = fullfile(tc.TestData.tmp, "encoder.mat");
cfg.paths.detector = fullfile(tc.TestData.tmp, "detector.mat");
cfg.paths.agent = fullfile(tc.TestData.tmp, "agent.mat");
cfg.paths.results = fullfile(tc.TestData.tmp, "results");
end

function X = randomStack(cfg, B)
s = cfg.env.camera.imgSize;
X = dlarray(rand(s, s, cfg.env.camera.frameStack, B, "single"), "SSCB");
end

function testEncoderAndDetectorShapes(tc)
cfg = tinyConfig(tc);
enc = buildEncoderNetwork(cfg);
Z = predict(enc, randomStack(cfg, 5));
verifyEqual(tc, size(Z), [cfg.model.featureDim 5]);
verifyLessThanOrEqual(tc, max(abs(extractdata(Z)), [], "all"), 1);
det = buildDetectorNetwork(cfg);
Y = predict(det, randomStack(cfg, 5));
verifyEqual(tc, size(Y), [31 5]);
[det2, n] = copyLearnables(det, enc);
verifyEqual(tc, n, 8);                                    % 4 conv layers x (weights, bias)
L = det2.Learnables;
E = enc.Learnables;
verifyEqual(tc, extractdata(L.Value{1}), extractdata(E.Value{1}));
end

function testCriticNetwork(tc)
cfg = tinyConfig(tc);
for dueling = [true false]
    cfg.model.dueling = dueling;
    net = buildCriticNetwork(cfg, 41, 8);
    img = randomStack(cfg, 4);
    st = dlarray(rand(41, 4, "single"), "CB");
    if string(net.InputNames{1}) == "image"
        Q = predict(net, img, st);
    else
        Q = predict(net, st, img);
    end
    verifyEqual(tc, size(Q), [8 4]);
end
end

function testSelfSupervisedLoss(tc)
cfg = tinyConfig(tc);
enc = buildEncoderNetwork(cfg);
inv = buildInverseNetwork(cfg.model.featureDim, 8);
W = dlarray(eye(cfg.model.featureDim, "single"));
Y = oneHot([1 3 5 8 2 2], 8);
[loss, gE, gW, gI] = dlfeval(@sslModelLoss, enc, enc, W, inv, randomStack(cfg, 6), ...
    randomStack(cfg, 6), randomStack(cfg, 6), Y, 1, 1, true);
verifyTrue(tc, isfinite(extractdata(loss)));
verifyEqual(tc, height(gE), height(enc.Learnables));
verifyEqual(tc, size(gW), size(W));
verifyEqual(tc, height(gI), height(inv.Learnables));
end

function testDetectorLoss(tc)
cfg = tinyConfig(tc);
det = buildDetectorNetwork(cfg);
occ = single(rand(15, 6) > 0.5);
distN = rand(15, 6, "single");
risk = single(rand(1, 6) > 0.5);
w = struct("distWeight", 2, "riskWeight", 1, "riskPosWeight", 5);
[loss, grad] = dlfeval(@detectorModelLoss, det, randomStack(cfg, 6), occ, distN, risk, w, true);
verifyTrue(tc, isfinite(extractdata(loss)));
verifyEqual(tc, height(grad), height(det.Learnables));
end

function testAugmentation(tc)
rs = RandStream("mt19937ar", "Seed", 0);
X = rand(16, 16, 3, 5, "single");
Y = augmentBatch(X, 3, true, rs);
verifySize(tc, Y, size(X));
verifyGreaterThanOrEqual(tc, min(Y, [], "all"), 0);
verifyLessThanOrEqual(tc, max(Y, [], "all"), 1);
end

function testAgentCreationAndAction(tc)
cfg = tinyConfig(tc);
env = UAVNavEnv(cfg, PerceptionSource="oracle", UseFeatures=true, ShieldEnabled=true, Seed=0);
agent = createDQNAgent(cfg, getObservationInfo(env), getActionInfo(env));
obs = env.resetWithSeed(0);
a = getAction(agent, obs);
if iscell(a)
    a = a{1};
end
verifyTrue(tc, any(double(a) == 1:8));
end

function testDataPipelineAndTinyTraining(tc)
cfg = tinyConfig(tc);
collectData(cfg);
verifyTrue(tc, isfile(cfg.paths.rollouts));
data = load(cfg.paths.rollouts);
ds = makeDataset(data, cfg.env.camera.frameStack, 0.2, 0);
[X, Xn, A] = sampleTransitions(ds, 6, RandStream("mt19937ar", "Seed", 1));
verifySize(tc, X, [cfg.env.camera.imgSize cfg.env.camera.imgSize cfg.env.camera.frameStack 6]);
verifyEqual(tc, X(:, :, 2:end, :), Xn(:, :, 1:end-1, :));  % consecutive stacks overlap
verifyTrue(tc, all(A >= 1 & A <= 8));

pretrainSSL(cfg);
verifyTrue(tc, isfile(cfg.paths.encoder));
trainDetector(cfg);
verifyTrue(tc, isfile(cfg.paths.detector));
[agent, stats] = trainDQN(cfg);
verifyTrue(tc, isfile(cfg.paths.agent));
verifyNotEmpty(tc, stats.EpisodeReward);
results = evaluatePolicies(cfg, Agent=agent, Detector=load(cfg.paths.detector).detector, ...
    Policies=["dqn_shield" "reactive"], Episodes=1, RecordEpisodes=0);
verifyEqual(tc, height(results), 2);
close all
end
