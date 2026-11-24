%RUNALL  Complete pipeline: data -> self-supervised CNN -> obstacle detector -> DQN -> evaluation.
%
%   Run from the project root:  >> runAll
%   Set useQuickSettings = true first to check that everything works on your
%   machine in a few minutes (the resulting agent will not be good).

setupProject;
useQuickSettings = false;
if useQuickSettings
    cfg = quickConfig();
else
    cfg = defaultConfig();          % or dynamicHardConfig()
end

collectData(cfg);                   % 1. exploration rollouts + automatic labels
pretrainSSL(cfg);                   % 2. CURL + inverse-dynamics encoder pre-training
trainDetector(cfg);                 % 3. CNN obstacle detector
[agent, trainStats] = trainDQN(cfg); % 4. DQN with safety shield and curriculum
results = evaluateAgent(cfg);       % 5. benchmark vs. baselines, GIFs in results/
