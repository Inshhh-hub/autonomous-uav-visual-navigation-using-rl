%DEMO3DSCENARIO  Fly one episode and replay it in 3D with UAV Toolbox (uavScenario).
%
%   Uses the trained DQN agent + shield if checkpoints/<experiment>_agent.mat
%   exists, otherwise the reactive planner + shield. Saves an MP4 (or AVI on
%   Linux) to results/.  Run from the project root:  >> demo3DScenario

setupProject;
cfg = defaultConfig();
agent = [];
detNet = [];
policyName = "reactive_shield@oracle";
if isfile(cfg.paths.agent)
    S = load(cfg.paths.agent, "agent");
    agent = S.agent;
    policyName = "dqn_shield";
    if isfile(cfg.paths.detector)
        S = load(cfg.paths.detector, "detector");
        detNet = S.detector;
    end
end
[env, policy] = makeEnvForPolicy(cfg, policyName, agent, detNet);
[stats, rec] = runEpisode(env, policy, cfg.eval.seedOffset + 3, Record=true);
fprintf("%s: success=%d collision=%d steps=%d path=%.1f m\n", policyName, stats.success, ...
    stats.collision, stats.steps, stats.pathLength);
replay3DScenario(rec, VideoFile=fullfile(cfg.paths.results, "replay3d_" + replace(policyName, "@", "_") + ".mp4"));
