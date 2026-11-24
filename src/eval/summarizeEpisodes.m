function s = summarizeEpisodes(stats)
%SUMMARIZEEPISODES  Aggregate metrics over a struct array returned by runEpisode.
succ = [stats.success];
s = struct();
s.episodes = numel(stats);
s.successRate = mean(succ);
s.collisionRate = mean([stats.collision]);
s.timeoutRate = mean([stats.timeout]);
s.spl = mean([stats.spl]);
s.meanReturn = mean([stats.episodeReturn]);
s.meanMinClearance = mean([stats.minClearance]);
if any(succ)
    s.meanStepsSuccess = mean([stats(succ).steps]);
    s.meanPathSuccess = mean([stats(succ).pathLength]);
else
    s.meanStepsSuccess = NaN;
    s.meanPathSuccess = NaN;
end
s.interventionsPerEpisode = mean([stats.interventions]);
end
