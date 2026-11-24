function [env, policy] = makeEnvForPolicy(cfg, name, agent, detNet)
%MAKEENVFORPOLICY  Environment + policy handle for a policy name such as "dqn_shield@oracle".
%
%   The suffix "_shield" enables the safety shield inside the environment;
%   "@oracle" / "@detector" selects the perception source (default:
%   cfg.perception.source, falling back to oracle if no detector is given).
if nargin < 3
    agent = [];
end
if nargin < 4
    detNet = [];
end
parts = split(string(name), "@");
base = parts(1);
source = string(cfg.perception.source);
if numel(parts) > 1
    source = parts(2);
end
if source == "detector" && isempty(detNet)
    source = "oracle";
end
isDqn = startsWith(base, "dqn");
if isDqn && isempty(agent)
    error("makeEnvForPolicy:noAgent", "Policy %s needs a trained agent.", name);
end
useFeatures = false;
if isDqn
    agentObs = getObservationInfo(agent);
    useFeatures = agentObs(2).Dimension(1) > 10;
end
env = UAVNavEnv(cfg, Detector=detNet, PerceptionSource=source, UseFeatures=useFeatures, ...
    ShieldEnabled=endsWith(base, "_shield"), Seed=cfg.seed);
policy = makePolicy(base, cfg, env, agent);
end
