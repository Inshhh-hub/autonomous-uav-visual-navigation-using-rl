function [results, details] = evaluatePolicies(cfg, options)
%EVALUATEPOLICIES  Benchmark navigation policies on identical, seeded scenarios.
%
%   results = evaluatePolicies(cfg, Agent=agent, Detector=detNet)
%   results = evaluatePolicies(cfg, Policies=["reactive_shield" "random"])   % no training needed
%
%   Policy names: dqn_shield, dqn, reactive_shield, reactive, random.
%   Append "@oracle" to use ground-truth sector distances instead of the CNN
%   detector (perception upper bound), e.g. "dqn_shield@oracle".
%   Writes <results>/<experiment>_evaluation.csv / .mat and one GIF per policy.
arguments
    cfg struct
    options.Agent = []
    options.Detector = []
    options.Policies (1,:) string = cfg.eval.policies
    options.Episodes (1,1) double = cfg.eval.episodes
    options.RecordEpisodes (1,1) double = cfg.eval.recordEpisodes
    options.OutDir (1,1) string = cfg.paths.results
end
ensureFolder(options.OutDir);
rows = {};
details = struct();
viewer = [];
if options.RecordEpisodes > 0
    viewer = SimViewer();
end

for name = options.Policies
    if startsWith(name, "dqn") && isempty(options.Agent)
        fprintf("[eval] skipping %s - no trained agent given\n", name);
        continue
    end
    [env, policy] = makeEnvForPolicy(cfg, name, options.Agent, options.Detector);

    epStats = [];
    for i = 1:options.Episodes
        record = i <= options.RecordEpisodes;
        if record
            [st, rec] = runEpisode(env, policy, cfg.eval.seedOffset + i, Record=true, Viewer=viewer);
            gifName = fullfile(options.OutDir, replace(name, "@", "_") + "_episode" + i + ".gif");
            saveGif(rec.frames, gifName, 10);
            details.(matlab.lang.makeValidName(name + "_rec" + i)) = rmfield(rec, "frames");
        else
            st = runEpisode(env, policy, cfg.eval.seedOffset + i);
        end
        epStats = [epStats, st]; %#ok<AGROW>
    end
    s = summarizeEpisodes(epStats);
    details.(matlab.lang.makeValidName(name)) = epStats;
    fprintf("[eval] %-20s success %.2f  collision %.2f  timeout %.2f  SPL %.2f\n", ...
        name, s.successRate, s.collisionRate, s.timeoutRate, s.spl);
    rows(end + 1, :) = {name, s.successRate, s.collisionRate, s.timeoutRate, s.spl, ...
        s.meanReturn, s.meanMinClearance, s.meanStepsSuccess, s.interventionsPerEpisode}; %#ok<AGROW>
end
if ~isempty(viewer)
    viewer.closeFigure();
end

varNames = ["Policy" "SuccessRate" "CollisionRate" "TimeoutRate" "SPL" "MeanReturn" ...
    "MeanMinClearance" "MeanStepsSuccess" "ShieldInterventionsPerEp"];
if isempty(rows)
    rows = cell(0, numel(varNames));
end
results = cell2table(rows, VariableNames=varNames);
disp(results);
base = fullfile(options.OutDir, cfg.experiment + "_evaluation");
writetable(results, base + ".csv");
save(base + ".mat", "results", "details");
fprintf("[eval] results written to %s.csv\n", base);
end
