function file = collectData(cfg)
%COLLECTDATA  Step 1 - fly exploration policies and record automatically labelled frames.
%
%   Policies (mixed per episode, cfg.collect.policyMix):
%     random   - sticky random actions
%     reactive - reactive planner with action noise (mostly safe flight)
%     greedy   - flies straight at the goal ignoring obstacles (produces crashes)
%   Labels (no manual annotation):
%     sectorDist - nearest obstacle per image sector from the depth buffer
%     risk       - 1 if a crash happens within cfg.collect.riskHorizon steps
%     actions    - executed action (for the inverse-dynamics SSL loss)

c = cfg.collect;
env = UAVNavEnv(cfg, PerceptionSource="oracle", ShieldEnabled=false, Seed=cfg.seed);
reactive = makeReactivePlanner(cfg, env.Actions);
greedy = makeReactivePlanner(cfg, env.Actions, IgnoreObstacles=true);
rs = RandStream("mt19937ar", "Seed", cfg.seed + 1);
mix = c.policyMix / sum(c.policyMix);
policyNames = ["random" "reactive" "greedy"];
nA = numel(env.Actions);
H = cfg.env.camera.imgSize;
nS = env.NumSectors;

% pre-allocate generously, trim at the end
cap = c.steps + 2000;
frames = zeros(H, H, cap, "uint8");
states = zeros(10, cap, "single");
actions = zeros(cap, 1, "int16");
episode = zeros(cap, 1, "int32");
sectorDist = zeros(nS, cap, "single");
risk = zeros(cap, 1, "uint8");

n = 0;            % frames written
total = 0;        % transitions
ep = 0;
outcomes = struct("goal", 0, "collision", 0, "timeout", 0);
tStart = tic;
while total < c.steps
    ep = ep + 1;
    kind = policyNames(find(rand(rs) <= cumsum(mix), 1));
    env.setDifficulty(rand(rs));
    obs = env.resetWithSeed(cfg.seed * 100003 + ep);
    info = env.LastInfo;
    epFirst = n + 1;
    push(obs, info);
    prev = randi(rs, nA);
    while true
        switch kind
            case "random"
                if rand(rs) < c.randomRepeat
                    a = prev;
                else
                    a = randi(rs, nA);
                end
            case "reactive"
                a = reactiveAct(reactive, info.sectorDistTrue, info.goalBearing, info.goalDz);
                if rand(rs) < c.reactiveNoise
                    a = randi(rs, nA);
                end
            otherwise
                a = reactiveAct(greedy, info.sectorDistTrue, info.goalBearing, info.goalDz);
        end
        prev = a;
        actions(n) = a;                                % action taken in the current frame
        [obs, ~, isDone, info] = env.step(a);
        total = total + 1;
        push(obs, info);
        if isDone || info.truncated || n >= cap
            break
        end
    end
    actions(n) = -1;                                   % last frame of an episode has no action
    if info.collision
        risk(max(epFirst, n - c.riskHorizon):n) = 1;
        outcomes.collision = outcomes.collision + 1;
    elseif info.reached
        outcomes.goal = outcomes.goal + 1;
    else
        outcomes.timeout = outcomes.timeout + 1;
    end
    if mod(ep, 20) == 0
        fprintf("[collect] episodes=%d steps=%d/%d goal=%d collision=%d timeout=%d (%.0f steps/s)\n", ...
            ep, total, c.steps, outcomes.goal, outcomes.collision, outcomes.timeout, total / toc(tStart));
    end
    if n >= cap
        break
    end
end

data = struct();
data.frames = frames(:, :, 1:n);
data.states = states(:, 1:n);
data.actions = actions(1:n);
data.episode = episode(1:n);
data.sectorDist = sectorDist(:, 1:n);
data.risk = risk(1:n);
data.meta = struct("maxDepth", cfg.env.camera.maxDepth, ...
    "occupancyThreshold", cfg.env.sectors.occupancyThreshold, ...
    "rows", cfg.env.sectors.rows, "cols", cfg.env.sectors.cols, ...
    "nActions", nA, "frameStack", cfg.env.camera.frameStack, ...
    "obsMode", string(cfg.env.camera.obsMode), "imgSize", H);
file = cfg.paths.rollouts;
ensureFolder(file);
save(file, "-struct", "data", "-v7.3");
fprintf("[collect] saved %d frames from %d episodes to %s (risk-positive frames: %.1f%%)\n", ...
    n, ep, file, 100 * mean(data.risk));

    function push(o, inf0)
        % nested function: writes into the pre-allocated arrays without copying them
        n = n + 1;
        frames(:, :, n) = uint8(round(255 * o{1}(:, :, end)));
        states(:, n) = single(o{2}(1:10));
        sectorDist(:, n) = single(inf0.sectorDistTrue);
        episode(n) = ep;
    end
end
