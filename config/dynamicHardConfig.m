function cfg = dynamicHardConfig()
%DYNAMICHARDCONFIG  Crowded scenario with many fast obstacles crossing the flight path.

cfg = defaultConfig();
cfg.experiment = "uav_dqn_dynamic_hard";
cfg.env.nStatic = [15 35];
cfg.env.nDynamic = [3 10];
cfg.env.moverSpeed = [1.0 3.0];
cfg.env.crossingRatio = 0.7;
cfg.env.moverTurnProb = 0.04;
cfg.shield.ttcThreshold = 1.5;
cfg.dqn.maxEpisodes = 4000;
cfg.dqn.curriculumEpisodes = 2000;
cfg.paths.agent = fullfile(cfg.paths.checkpoints, cfg.experiment + "_agent.mat");
end
