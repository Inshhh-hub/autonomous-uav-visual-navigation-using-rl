function [agent, trainStats, env] = trainDQN(cfg)
%TRAINDQN  Step 4 - train the DQN navigation agent with Reinforcement Learning Toolbox.
%
%   Per step:  camera frames -> CNN detector -> sector distances / crash risk
%                  -> safety shield (post-posed: unsafe actions are replaced)
%                  -> state vector [+ detector features]
%              Q-network(frames, state) -> epsilon-greedy action
%   The curriculum raises the obstacle density over cfg.dqn.curriculumEpisodes.

rng(cfg.seed);
detNet = [];
source = cfg.perception.source;
if source == "detector"
    if isfile(cfg.paths.detector)
        S = load(cfg.paths.detector, "detector");
        detNet = S.detector;
    else
        warning("trainDQN:noDetector", "%s not found - falling back to oracle perception.", cfg.paths.detector);
        source = "oracle";
    end
end
env = UAVNavEnv(cfg, Detector=detNet, PerceptionSource=source, ...
    UseFeatures=cfg.perception.useFeatures, ShieldEnabled=cfg.shield.enabled, Seed=cfg.seed);
env.CurriculumEpisodes = cfg.dqn.curriculumEpisodes;
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

encNet = [];
if isfile(cfg.paths.encoder)
    S = load(cfg.paths.encoder, "encoder");
    encNet = S.encoder;
else
    fprintf("[dqn] no self-supervised encoder found - CNN starts from random weights\n");
end
agent = createDQNAgent(cfg, obsInfo, actInfo, encNet);

trainOpts = rlTrainingOptions( ...
    MaxEpisodes=cfg.dqn.maxEpisodes, ...
    MaxStepsPerEpisode=cfg.env.maxSteps, ...
    ScoreAveragingWindowLength=cfg.dqn.scoreWindow, ...
    StopTrainingCriteria="EpisodeCount", ...
    StopTrainingValue=cfg.dqn.maxEpisodes, ...
    Verbose=cfg.dqn.verbose, ...
    Plots=cfg.dqn.plots);

fprintf("[dqn] training for %d episodes (perception=%s, shield=%d, features=%d)\n", ...
    cfg.dqn.maxEpisodes, source, cfg.shield.enabled, cfg.perception.useFeatures);
trainStats = train(agent, env, trainOpts);

episodeLog = env.EpisodeLog;
ensureFolder(cfg.paths.agent);
saveCfg = cfg; %#ok<NASGU>
save(cfg.paths.agent, "agent", "trainStats", "episodeLog", "saveCfg", "source");
fprintf("[dqn] agent saved to %s\n", cfg.paths.agent);

ensureFolder(cfg.paths.results);
plotTrainingResults(trainStats, episodeLog, fullfile(cfg.paths.results, cfg.experiment + "_training.png"));
end
