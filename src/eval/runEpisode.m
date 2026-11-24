function [stats, rec] = runEpisode(env, policy, seed, options)
%RUNEPISODE  Run one episode with policy(env, obs) -> action index.
%
%   [stats, rec] = runEpisode(env, policy, seed, Record=true, Viewer=viewer)
%   stats: success / collision / timeout, return, steps, path length, SPL, ...
%   rec:   trajectory (positions, yaw, velocities, mover positions, actions)
%          and, when a SimViewer is given, the rendered RGB frames.
arguments
    env
    policy function_handle
    seed (1,1) double
    options.Record (1,1) logical = false
    options.Viewer = []
end
obs = env.resetWithSeed(seed);
ret = 0;
rec = struct();
if options.Record
    rec.world = env.World;
    rec.goal = env.Goal;
    rec.start = env.Start;
    rec.goalRadius = env.Cfg.env.goalRadius;
    rec.dt = env.Cfg.env.dt * env.Cfg.env.actionRepeat;
    rec.pos = env.Pos;
    rec.vel = env.Vel;
    rec.yaw = env.Yaw;
    rec.moverPos = env.World.moverPos;
    rec.actions = strings(1, 0);
    rec.frames = {};
    if ~isempty(options.Viewer)
        options.Viewer.newEpisode(env);
        rec.frames{end + 1} = options.Viewer.update(env, "");
    end
end
while true
    a = policy(env, obs);
    [obs, r, isDone, info] = env.step(a);
    ret = ret + r;
    if options.Record
        rec.pos(:, end + 1) = env.Pos;
        rec.vel(:, end + 1) = env.Vel;
        rec.yaw(end + 1) = env.Yaw;
        rec.moverPos(:, :, end + 1) = env.World.moverPos;
        rec.actions(end + 1) = env.Actions(info.executedAction).name;
        if ~isempty(options.Viewer)
            rec.frames{end + 1} = options.Viewer.update(env, env.Actions(info.executedAction).name);
        end
    end
    if isDone || info.truncated
        break
    end
end
shortest = info.startDistance;
stats = struct( ...
    "success", info.reached, ...
    "collision", info.collision, ...
    "timeout", info.truncated && ~isDone, ...
    "obstacle", info.obstacle, ...
    "episodeReturn", ret, ...
    "steps", info.steps, ...
    "pathLength", info.pathLength, ...
    "startDistance", shortest, ...
    "spl", double(info.reached) * shortest / max([info.pathLength, shortest, 1e-6]), ...
    "minClearance", info.minClearance, ...
    "interventions", info.interventions);
end
