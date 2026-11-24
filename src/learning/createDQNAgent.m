function agent = createDQNAgent(cfg, obsInfo, actInfo, encNet)
%CREATEDQNAGENT  Double DQN agent with dueling critic, n-step returns and prioritised replay.
%
%   agent = createDQNAgent(cfg, obsInfo, actInfo)          % random CNN initialisation
%   agent = createDQNAgent(cfg, obsInfo, actInfo, encNet)  % CNN initialised from the
%                                                          % self-supervised encoder
nActions = numel(actInfo.Elements);
stateDim = obsInfo(2).Dimension(1);
net = buildCriticNetwork(cfg, stateDim, nActions);
if nargin >= 4 && ~isempty(encNet)
    [net, n] = copyLearnables(net, encNet);
    fprintf("[dqn] initialised %d critic parameter tensors from the self-supervised encoder\n", n);
end

device = "cpu";
if cfg.useGPU && canUseGPU
    device = "gpu";
end
critic = rlVectorQValueFunction(net, obsInfo, actInfo, ...
    ObservationInputNames=["image", "state"], UseDevice=device);

d = cfg.dqn;
opts = rlDQNAgentOptions();
opts.UseDoubleDQN = true;
opts.DiscountFactor = d.discount;
opts.MiniBatchSize = d.miniBatchSize;
opts.ExperienceBufferLength = d.bufferLength;
opts.NumStepsToLookAhead = d.nStep;
opts.TargetSmoothFactor = d.targetSmoothFactor;
opts.TargetUpdateFrequency = d.targetUpdateFrequency;
opts.SampleTime = cfg.env.dt * cfg.env.actionRepeat;
opts.CriticOptimizerOptions = rlOptimizerOptions(LearnRate=d.learnRate, GradientThreshold=d.gradientThreshold);
opts.EpsilonGreedyExploration.Epsilon = d.epsilonStart;
opts.EpsilonGreedyExploration.EpsilonMin = d.epsilonMin;
opts.EpsilonGreedyExploration.EpsilonDecay = d.epsilonDecay;
try %#ok<TRYNC> % option removed in some releases; keep the buffer between train() calls
    opts.ResetExperienceBufferBeforeTraining = false;
end

agent = rlDQNAgent(critic, opts);

if d.prioritizedReplay
    try
        buffer = rlPrioritizedReplayMemory(obsInfo, actInfo, d.bufferLength);
        buffer.PriorityExponent = d.perAlpha;
        buffer.InitialImportanceSamplingExponent = d.perBetaStart;
        buffer.NumAnnealingSteps = d.perAnnealingSteps;
        agent.ExperienceBuffer = buffer;
    catch err
        warning("uavnav:per", "Prioritized replay unavailable (%s) - using uniform replay.", err.message);
    end
end
end
