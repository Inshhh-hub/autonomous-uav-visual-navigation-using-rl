function cfg = quickConfig()
%QUICKCONFIG  Tiny settings for a fast end-to-end check of the whole pipeline.
%   The numbers are far too small to learn a good policy - use them to make
%   sure everything runs on your machine before the full training.

cfg = defaultConfig();
cfg.experiment = "quick_smoke";
cfg.env.maxSteps = 120;
cfg.env.nStatic = [6 14];
cfg.env.nDynamic = [0 2];
cfg.collect.steps = 3000;
cfg.ssl.iterations = 150;
cfg.ssl.batchSize = 32;
cfg.ssl.logEvery = 25;
cfg.ssl.evalEvery = 75;
cfg.detector.iterations = 150;
cfg.detector.batchSize = 32;
cfg.detector.logEvery = 25;
cfg.detector.evalEvery = 75;
cfg.dqn.maxEpisodes = 30;
cfg.dqn.curriculumEpisodes = 20;
cfg.dqn.bufferLength = 5000;
cfg.dqn.miniBatchSize = 32;
cfg.eval.episodes = 5;

cfg.paths.rollouts = fullfile(cfg.paths.data, "quick_rollouts.mat");
cfg.paths.encoder = fullfile(cfg.paths.checkpoints, "quick_encoder_ssl.mat");
cfg.paths.detector = fullfile(cfg.paths.checkpoints, "quick_detector.mat");
cfg.paths.agent = fullfile(cfg.paths.checkpoints, cfg.experiment + "_agent.mat");
cfg.paths.results = fullfile(projectRoot(), "results", "quick");
end
