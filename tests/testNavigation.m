function tests = testNavigation
%TESTNAVIGATION  Unit tests of the safety shield, sector geometry and reactive planner.
tests = functiontests(localfunctions);
end

function setupOnce(~)
addpath(fileparts(fileparts(mfilename("fullpath"))));
setupProject();
end

function sh = makeShield(varargin)
cfg = defaultConfig();
for k = 1:2:numel(varargin)
    cfg.shield.(varargin{k}) = varargin{k + 1};
end
sh = SafetyShield(cfg, uavActions());
end

function idx = act(name)
names = [uavActions().name];
idx = find(names == name);
end

function testSectorGroups(tc)
verifyEqual(tc, sort(sectorIndices(3, 5, "mid", "center"))', [7 8 9]);
verifyEqual(tc, sort(sectorIndices(3, 5, "mid", "left"))', [6 7 8]);
verifyEqual(tc, sort(sectorIndices(3, 5, "top", "center"))', [2 3 4]);
reqs = actionSectorRequirements(3, 5, uavActions());
clr = actionClearances(reqs, (0:14)');
verifyEqual(tc, clr(act("forward")), 6);
verifyEqual(tc, clr(act("hover")), inf);
end

function testShieldAllowsEverythingInFreeSpace(tc)
sh = makeShield();
[mask, info] = sh.computeMask(20 * ones(15, 1), 0, 3, 0);
verifyTrue(tc, all(mask));
verifyFalse(tc, info.fallback);
end

function testShieldBlocksForwardWhenObstacleAhead(tc)
sh = makeShield();
d = 20 * ones(15, 1);
d(8) = 1.5;                                  % centre sector, 1.5 m away
mask = sh.computeMask(d, 0, 3, 0);
verifyFalse(tc, mask(act("forward")));
verifyTrue(tc, mask(act("hover")));
end

function testShieldGeofence(tc)
sh = makeShield();
mask = sh.computeMask(20 * ones(15, 1), 0, 7.8, 0);
verifyFalse(tc, mask(act("ascend")));
verifyTrue(tc, mask(act("descend")));
sh.reset();
mask = sh.computeMask(20 * ones(15, 1), 0, 0.5, 0);
verifyFalse(tc, mask(act("descend")));
verifyTrue(tc, mask(act("ascend")));
end

function testShieldTimeToCollisionAndThreat(tc)
sh = makeShield("closingSmoothing", 0);
d = 20 * ones(15, 1);
d(7:9) = 6;
mask = sh.computeMask(d, 0, 3, 0);
verifyTrue(tc, mask(act("forward")));         % 6 m away and static
d(7:9) = 5;                                  % 1 m closer in 0.2 s -> TTC = 1 s
[mask, info] = sh.computeMask(d, 0, 3, 0);
verifyFalse(tc, mask(act("forward")));
verifyTrue(tc, info.threat);
verifyFalse(tc, mask(act("hover")));          % approaching obstacle: hovering is unsafe
end

function testShieldLearnedRisk(tc)
sh = makeShield();
mask = sh.computeMask(20 * ones(15, 1), 0.95, 3, 0);
verifyFalse(tc, mask(act("forward")));
verifyTrue(tc, mask(act("turn_left")));
end

function testShieldFallbackKeepsOneAction(tc)
sh = makeShield();
sh.computeMask(0.3 * ones(15, 1), 0, 3, 0);
mask = sh.computeMask(0.1 * ones(15, 1), 0, 3, 0);
verifyGreaterThanOrEqual(tc, nnz(mask), 1);
end

function testReactivePlanner(tc)
cfg = defaultConfig();
planner = makeReactivePlanner(cfg, uavActions());
names = [uavActions().name];
a = reactiveAct(planner, 20 * ones(15, 1), pi / 2, 0);
verifyTrue(tc, any(names(a) == ["turn_left" "forward_left"]));
a = reactiveAct(planner, 20 * ones(15, 1), 0, 0);
verifyEqual(tc, names(a), "forward");
d = 20 * ones(15, 1);
d(7:9) = 1;                                  % wall straight ahead
a = reactiveAct(planner, d, 0, 0);
verifyNotEqual(tc, names(a), "forward");
end

function testShieldSubstitutesUnsafeAction(tc)
cfg = defaultConfig();
env = UAVNavEnv(cfg, ShieldEnabled=true, Seed=0);
env.resetWithSeed(3);
w = emptyWorld(cfg.env.arenaSize, cfg.env.ceiling, cfg.env.wallHeight);
w.pillars = [12 25 1 10];
w.pillarAlbedo = 0.7;
env.World = w;
env.Pos = [10; 25; 3];
env.Yaw = 0;
env.Vel = zeros(3, 1);
env.Shield.reset();
env.Frames = repmat(env.renderFrame(), 1, 1, cfg.env.camera.frameStack);
env.updatePerception();
verifyFalse(tc, env.Mask(act("forward")));
[~, ~, ~, info] = env.step(act("forward"));
verifyTrue(tc, info.shielded);
verifyNotEqual(tc, info.executedAction, act("forward"));
end

function testReactiveBaselineRuns(tc)
cfg = defaultConfig();
cfg.env.maxSteps = 60;
[env, policy] = makeEnvForPolicy(cfg, "reactive_shield@oracle");
stats = runEpisode(env, policy, 5);
verifyTrue(tc, isfield(stats, "success"));
s = summarizeEpisodes([stats, runEpisode(env, policy, 6)]);
verifyEqual(tc, s.episodes, 2);
end

function testRocAuc(tc)
verifyEqual(tc, rocAuc([0 0 1 1], [0.1 0.2 0.8 0.9]), 1);
verifyEqual(tc, rocAuc([0 1 0 1], [0.5 0.5 0.5 0.5]), 0.5, AbsTol=1e-12);
end
