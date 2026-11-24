function policy = makePolicy(name, cfg, env, agent)
%MAKEPOLICY  Function handle policy(env, obs) -> action for evaluation and demos.
%
%   "dqn" / "dqn_shield"            greedy action of a trained rlDQNAgent
%                                   (the shield, if any, lives in the environment)
%   "reactive" / "reactive_shield"  VFH-style reactive planner on the perceived sectors
%   "random"                        uniformly random safe action
switch name
    case {"dqn", "dqn_shield"}
        if nargin < 4 || isempty(agent)
            error("makePolicy:noAgent", "Policy %s needs a trained agent.", name);
        end
        try %#ok<TRYNC>
            agent.UseExplorationPolicy = false;
        end
        policy = @(e, obs) agentAction(agent, obs);
    case {"reactive", "reactive_shield"}
        planner = makeReactivePlanner(cfg, env.Actions);
        policy = @(e, obs) plannerAction(planner, e);
    case "random"
        rs = RandStream("mt19937ar", "Seed", cfg.seed + 7);
        policy = @(e, obs) randomAction(rs, e);
    otherwise
        error("makePolicy:unknown", "Unknown policy %s.", name);
end
end

function a = agentAction(agent, obs)
a = getAction(agent, obs);
if iscell(a)
    a = a{1};
end
a = double(a);
end

function a = plannerAction(planner, env)
[~, bearing, dz] = env.goalGeometry();
a = reactiveAct(planner, env.Percept.sectorDist, bearing, dz, env.Mask);
end

function a = randomAction(rs, env)
allowed = find(env.Mask);
a = allowed(randi(rs, numel(allowed)));
end
