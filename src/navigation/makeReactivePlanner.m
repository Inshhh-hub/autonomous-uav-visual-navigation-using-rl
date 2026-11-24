function planner = makeReactivePlanner(cfg, actions, options)
%MAKEREACTIVEPLANNER  Vector-Field-Histogram style reactive planner (baseline).
%
%   Every action is scored by its expected progress towards the goal after a
%   short look-ahead, its heading and altitude error, and the free space in
%   the sectors it flies into. Used as
%     * a classical baseline in the evaluation,
%     * an exploration policy for data collection (IgnoreObstacles=true gives
%       a greedy goal-seeker that crashes and produces crash-risk labels),
%     * the substitute chooser of the safety shield.

arguments
    cfg struct
    actions struct
    options.IgnoreObstacles (1,1) logical = false
end
planner.actions = actions;
planner.reqs = actionSectorRequirements(cfg.env.sectors.rows, cfg.env.sectors.cols, actions);
planner.uavRadius = cfg.env.uavRadius;
planner.horizon = 1.0;          % look-ahead [s]
planner.clearanceCap = 8;       % [m]
planner.wProgress = 1.0;
planner.wHeading = 1.0;
planner.wVertical = 0.6;
planner.wClearance = 1.5;
planner.margin = 0.5;
planner.reactionTime = 0.8;
planner.ignoreObstacles = options.IgnoreObstacles;
end
