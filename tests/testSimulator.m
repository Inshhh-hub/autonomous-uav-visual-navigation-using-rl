function tests = testSimulator
%TESTSIMULATOR  Unit tests of the MATLAB UAV simulator.   >> runtests("tests")
tests = functiontests(localfunctions);
end

function setupOnce(~)
addpath(fileparts(fileparts(mfilename("fullpath"))));
setupProject();
end

function env = placedEnv(cfg, pos, yaw, goal, pillars)
env = UAVNavEnv(cfg, Seed=0);
env.resetWithSeed(0);
w = emptyWorld(cfg.env.arenaSize, cfg.env.ceiling, cfg.env.wallHeight);
if nargin >= 5
    w.pillars = pillars;
    w.pillarAlbedo = 0.7 * ones(size(pillars, 1), 1);
end
env.World = w;
env.Pos = pos(:);
env.Vel = zeros(3, 1);
env.Yaw = yaw;
env.YawRate = 0;
env.Goal = goal(:);
env.PrevDist = norm(env.Goal - env.Pos);
end

function testResetShapes(tc)
cfg = defaultConfig();
env = UAVNavEnv(cfg, Seed=0);
obs = env.resetWithSeed(0);
s = cfg.env.camera.imgSize;
k = cfg.env.camera.frameStack;
verifySize(tc, obs{1}, [s s k]);
verifySize(tc, obs{2}, [10 1]);
verifyGreaterThanOrEqual(tc, min(obs{1}, [], "all"), 0);
verifyLessThanOrEqual(tc, max(obs{1}, [], "all"), 1);
verifyEqual(tc, obs{1}(:, :, 1), obs{1}(:, :, end));      % stack filled with the first frame
verifySize(tc, env.LastInfo.sectorDistTrue, [15 1]);
end

function testValidateEnvironment(tc)
env = UAVNavEnv(defaultConfig(), Seed=1);
validateEnvironment(env);
verifyTrue(tc, true);
end

function testFrameStackSlides(tc)
env = UAVNavEnv(defaultConfig(), Seed=0);
obs = env.resetWithSeed(1);
nxt = env.step(1);
verifyEqual(tc, nxt{1}(:, :, 1:end-1), obs{1}(:, :, 2:end));
end

function testSeededResetIsDeterministic(tc)
cfg = defaultConfig();
a = UAVNavEnv(cfg);
b = UAVNavEnv(cfg);
oa = a.resetWithSeed(7);
ob = b.resetWithSeed(7);
verifyEqual(tc, oa{1}, ob{1});
for act = [1 2 4 6 8 3]
    [oa, ra] = a.step(act);
    [ob, rb] = b.step(act);
    verifyEqual(tc, oa{1}, ob{1});
    verifyEqual(tc, ra, rb);
end
end

function testStartAndGoalAreFree(tc)
cfg = defaultConfig();
env = UAVNavEnv(cfg);
for seed = 1:20
    env.resetWithSeed(seed);
    verifyGreaterThan(tc, worldClearance(env.World, env.Pos), cfg.env.uavRadius);
    verifyGreaterThanOrEqual(tc, norm(env.Goal(1:2) - env.Start(1:2)), cfg.env.minStartGoalDist - 1e-9);
end
end

function testCameraDepthToPillar(tc)
cam = makeCamera(64, 90, 45);
w = emptyWorld(50, 8, 10);
w.pillars = [20 25 1 10];
w.pillarAlbedo = 0.7;
depth = renderCamera(w, cam, [10 25 3], 0);
verifyEqual(tc, depth(32:33, 32:33), 9 * ones(2), AbsTol=0.05);
verifyGreaterThan(tc, depth(32, 1), 9.5);                  % far left of the image misses the pillar
end

function testCameraGroundDepth(tc)
cam = makeCamera(64, 90, 45);
w = emptyWorld(50, 8, 10);
z = 2;
depth = renderCamera(w, cam, [25 25 z], 0);
dz = reshape(cam.dirs(:, 3), 64, 64);
expected = z ./ -dz(64, 32:33);
verifyEqual(tc, depth(64, 32:33), expected, RelTol=1e-6);
end

function testSectorMinDepth(tc)
depth = 9 * ones(6, 10);
depth(1, 1) = 1;            % top-left sector
depth(6, 10) = 2;           % bottom-right sector
out = sectorMinDepth(depth, 3, 5);
verifyEqual(tc, out(1), 1);
verifyEqual(tc, out(15), 2);
verifyEqual(tc, out(8), 9);
end

function testCollisionWithPillar(tc)
cfg = defaultConfig();
env = placedEnv(cfg, [10 25 3], 0, [40 25 3], [13 25 0.5 10]);
isDone = false;
for k = 1:30
    [~, reward, isDone, info] = env.step(1);       % forward
    if isDone
        break
    end
end
verifyTrue(tc, isDone);
verifyTrue(tc, info.collision);
verifyEqual(tc, info.obstacle, "pillar");
verifyLessThan(tc, reward, cfg.env.reward.collision + 1);
verifyError(tc, @() env.step(1), "UAVNavEnv:episodeOver");
end

function testGoalReached(tc)
cfg = defaultConfig();
env = placedEnv(cfg, [10 25 3], 0, [14 25 3]);
total = 0;
for k = 1:30
    [~, reward, isDone, info] = env.step(1);
    total = total + reward;
    if isDone
        break
    end
end
verifyTrue(tc, info.reached);
verifyFalse(tc, info.collision);
verifyGreaterThan(tc, total, cfg.env.reward.goal);
end

function testTimeLimitTruncates(tc)
cfg = defaultConfig();
cfg.env.maxSteps = 5;
env = placedEnv(cfg, [25 25 3], 0, [45 45 3]);
for k = 1:5
    [~, ~, isDone, info] = env.step(8);             % hover
end
verifyTrue(tc, info.truncated);
verifyFalse(tc, isDone);
end

function testCeilingIsAGeofence(tc)
cfg = defaultConfig();
env = placedEnv(cfg, [25 25 7], 0, [45 45 3]);
for k = 1:40
    [~, ~, isDone, info] = env.step(6);             % ascend
    if isDone
        break
    end
end
verifyTrue(tc, info.collision);
verifyEqual(tc, info.obstacle, "ceiling");
end

function testMoversStayInsideArena(tc)
cfg = defaultConfig();
rs = RandStream("mt19937ar", "Seed", 3);
w = generateWorld(rs, cfg.env, 15, 6, [5; 5; 3], [40; 40; 3]);
verifyEqual(tc, size(w.moverPos, 1), 6);
for k = 1:500
    w = stepMovers(w, 0.1, rs, 0.05, [1 6]);
end
r = w.moverRadius;
verifyTrue(tc, all(w.moverPos(:, 1) >= r - 1e-9 & w.moverPos(:, 1) <= w.size - r + 1e-9));
verifyTrue(tc, all(w.moverPos(:, 3) >= 1 - 1e-9 & w.moverPos(:, 3) <= 6 + 1e-9));
end

function testDepthObservationMode(tc)
cfg = defaultConfig();
cfg.env.camera.obsMode = "depth";
env = UAVNavEnv(cfg, Seed=0);
obs = env.resetWithSeed(0);
expected = min(env.Depth / cfg.env.camera.maxDepth, 1);
verifyLessThan(tc, mean(abs(double(obs{1}(:, :, end)) - expected), "all"), 0.05);
end

function testEpisodeLogCountsOutcomes(tc)
cfg = defaultConfig();
cfg.env.maxSteps = 3;
env = UAVNavEnv(cfg, Seed=0);
env.resetWithSeed(1);
for k = 1:3
    env.step(8);
end
env.resetWithSeed(2);                                  % previous episode timed out
verifyEqual(tc, numel(env.EpisodeLog), 1);
verifyEqual(tc, env.EpisodeLog(1).outcome, "timeout");
end
