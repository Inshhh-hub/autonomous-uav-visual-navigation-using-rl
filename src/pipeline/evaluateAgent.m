function results = evaluateAgent(cfg, varargin)
%EVALUATEAGENT  Step 5 - load the trained agent / detector and benchmark all policies.
%   results = evaluateAgent(cfg)
%   results = evaluateAgent(cfg, Policies=["dqn_shield" "dqn_shield@oracle"], Episodes=100)
agent = [];
if isfile(cfg.paths.agent)
    S = load(cfg.paths.agent, "agent");
    agent = S.agent;
else
    fprintf("[eval] %s not found - only the baselines will be evaluated\n", cfg.paths.agent);
end
detNet = [];
if isfile(cfg.paths.detector)
    S = load(cfg.paths.detector, "detector");
    detNet = S.detector;
end
results = evaluatePolicies(cfg, Agent=agent, Detector=detNet, varargin{:});
end
