function cfg = defaultConfig()
%DEFAULTCONFIG  All parameters of the UAV visual-navigation project.
%
%   cfg = defaultConfig() returns a nested struct. Edit values here or
%   override them in your script, e.g.
%       cfg = defaultConfig();
%       cfg.dqn.maxEpisodes = 3000;
%       cfg.env.nDynamic = [2 8];

cfg = struct();
cfg.experiment = "uav_dqn";
cfg.seed = 0;
cfg.useGPU = false;                 % GPU for SSL / detector / DQN (needs Parallel Computing Toolbox)

%% ------------------------------------------------------------ environment
env = struct();
env.arenaSize = 50;                 % side length of the square arena [m]
env.ceiling = 8;                    % geofence: maximum altitude [m]
env.wallHeight = 10;
env.nStatic = [10 30];              % pillars per episode [min max]; max reached at difficulty 1
env.nDynamic = [0 5];               % moving obstacles per episode [min max]
env.pillarRadius = [0.3 1.2];
env.pillarHeight = [3 14];
env.moverRadius = [0.3 0.6];
env.moverSpeed = [0.5 2.0];
env.moverAltitude = [1 6];
env.moverTurnProb = 0.02;           % per physics step
env.crossingRatio = 0.5;            % share of movers that cut across the start->goal corridor
env.minObstacleSpacing = 2.5;
env.startGoalClearance = 3;
env.startAltitude = [1.5 4];
env.goalAltitude = [1.5 5];
env.minStartGoalDist = 20;
env.goalRadius = 1.5;
env.uavRadius = 0.35;
env.safeDistance = 2;               % proximity penalty below this clearance [m]
env.dt = 0.1;                       % physics step [s]
env.actionRepeat = 2;               % decision period = dt * actionRepeat = 0.2 s
env.maxSteps = 400;                 % decisions per episode (80 s)
env.velocityTau = 0.3;              % first-order lag of the velocity controller [s]
env.yawTau = 0.15;
env.windStd = 0.05;                 % gust added to the velocity every physics step [m/s]
env.speedScale = 1;
env.difficulty = 1;                 % 0..1 (overridden by the curriculum during training)

env.camera.imgSize = 48;            % square image, pixels
env.camera.fovDeg = 90;
env.camera.maxDepth = 20;
env.camera.obsMode = "gray";        % "gray" (monocular camera) or "depth" (depth camera)
env.camera.frameStack = 3;
env.camera.noiseStd = 0.02;
env.camera.fogDistance = 45;
env.camera.domainRandomization = true;

env.sectors.rows = 3;
env.sectors.cols = 5;
env.sectors.occupancyThreshold = 3; % sector "occupied" if an obstacle is closer [m]

env.reward.progress = 0.5;          % per metre of progress towards the goal
env.reward.step = -0.01;
env.reward.proximity = -0.3;
env.reward.collision = -10;
env.reward.goal = 10;
env.reward.shieldPenalty = -0.05;   % when the safety shield overrides the agent
cfg.env = env;

%% ----------------------------------------------------- perception & shield
cfg.perception.source = "detector"; % "detector" (CNN) or "oracle" (ground-truth sectors)
cfg.perception.useFeatures = true;  % append detector outputs to the DQN state vector

cfg.shield.enabled = true;
cfg.shield.margin = 0.4;            % extra clearance on top of the UAV radius [m]
cfg.shield.reactionTime = 0.8;      % seconds of travel that must be free [s]
cfg.shield.ttcThreshold = 1.2;      % time-to-collision limit [s]
cfg.shield.riskThreshold = 0.7;     % learned crash-risk probability that blocks fast flight
cfg.shield.minAltitude = 0.8;
cfg.shield.ceilingMargin = 0.7;
cfg.shield.closingSmoothing = 0.5;  % EMA factor of closing-speed estimates
cfg.shield.threatSpeed = 0.5;       % closing speed beyond own motion that marks a moving threat

%% ---------------------------------------------------------- data collection
cfg.collect.steps = 60000;
cfg.collect.policyMix = [0.3 0.4 0.3];   % [random reactive greedy]
cfg.collect.reactiveNoise = 0.25;
cfg.collect.randomRepeat = 0.8;          % sticky random actions
cfg.collect.riskHorizon = 5;             % frames this close to a crash get risk label 1

%% --------------------------------------------- self-supervised pretraining
cfg.ssl.iterations = 6000;
cfg.ssl.batchSize = 96;
cfg.ssl.learnRate = 3e-4;
cfg.ssl.curlWeight = 1;
cfg.ssl.inverseWeight = 1;
cfg.ssl.momentumTau = 0.05;
cfg.ssl.shiftPad = 3;
cfg.ssl.valFraction = 0.1;
cfg.ssl.logEvery = 100;
cfg.ssl.evalEvery = 500;

%% ------------------------------------------------------ obstacle detector
cfg.detector.iterations = 5000;
cfg.detector.batchSize = 96;
cfg.detector.learnRate = 3e-4;
cfg.detector.freezeEncoder = false;
cfg.detector.distWeight = 2;
cfg.detector.riskWeight = 1;
cfg.detector.riskPosWeight = 5;
cfg.detector.valFraction = 0.1;
cfg.detector.logEvery = 100;
cfg.detector.evalEvery = 500;

%% -------------------------------------------------------------- networks
cfg.model.featureDim = 256;
cfg.model.hiddenDim = 256;
cfg.model.dueling = true;           % set false if your release rejects the functionLayer

%% ------------------------------------------------------------------- DQN
cfg.dqn.maxEpisodes = 2500;
cfg.dqn.curriculumEpisodes = 1200;  % difficulty ramps 0.1 -> 1 over these episodes (0 = off)
cfg.dqn.learnRate = 1e-4;
cfg.dqn.gradientThreshold = 10;
cfg.dqn.discount = 0.99;
cfg.dqn.nStep = 3;
cfg.dqn.miniBatchSize = 64;
cfg.dqn.bufferLength = 30000;       % 48x48x3 single images: ~1.7 GB of replay memory
cfg.dqn.targetSmoothFactor = 1e-2;
cfg.dqn.targetUpdateFrequency = 1;
cfg.dqn.epsilonStart = 1.0;
cfg.dqn.epsilonMin = 0.05;
cfg.dqn.epsilonDecay = 2e-5;        % per agent step (multiplicative)
cfg.dqn.prioritizedReplay = true;
cfg.dqn.perAlpha = 0.6;
cfg.dqn.perBetaStart = 0.4;
cfg.dqn.perAnnealingSteps = 3e5;
cfg.dqn.scoreWindow = 50;
cfg.dqn.plots = "training-progress"; % or "none"
cfg.dqn.verbose = false;

%% ------------------------------------------------------------- evaluation
cfg.eval.episodes = 50;
cfg.eval.seedOffset = 10000;
cfg.eval.policies = ["dqn_shield" "dqn" "reactive_shield" "reactive" "random"];
cfg.eval.recordEpisodes = 1;        % GIFs per policy

%% ------------------------------------------------------------------ paths
root = projectRoot();
cfg.paths.data = fullfile(root, "data");
cfg.paths.checkpoints = fullfile(root, "checkpoints");
cfg.paths.results = fullfile(root, "results");
cfg.paths.rollouts = fullfile(cfg.paths.data, "rollouts.mat");
cfg.paths.encoder = fullfile(cfg.paths.checkpoints, "encoder_ssl.mat");
cfg.paths.detector = fullfile(cfg.paths.checkpoints, "detector.mat");
cfg.paths.agent = fullfile(cfg.paths.checkpoints, cfg.experiment + "_agent.mat");
end
