function fig = plotTrainingResults(trainStats, episodeLog, outFile)
%PLOTTRAININGRESULTS  Reward, success / collision rates, curriculum and episode length.
fig = figure(Name="DQN training results", NumberTitle="off", Color="w", Position=[100 100 1100 650]);
tl = tiledlayout(fig, 2, 2, TileSpacing="compact", Padding="compact");
win = 50;

ax = nexttile(tl);
try
    plot(ax, trainStats.EpisodeIndex, trainStats.EpisodeReward, Color=[0.75 0.75 0.75]);
    hold(ax, "on");
    plot(ax, trainStats.EpisodeIndex, trainStats.AverageReward, LineWidth=1.6);
    legend(ax, ["episode reward" "moving average"], Location="southeast");
catch
    text(ax, 0.1, 0.5, "trainStats not available");
end
title(ax, "Episode reward");
xlabel(ax, "episode");

if ~isempty(episodeLog)
    outcome = [episodeLog.outcome];
    ax = nexttile(tl);
    plot(ax, movmean(double(outcome == "goal"), win), LineWidth=1.6);
    hold(ax, "on");
    plot(ax, movmean(double(outcome == "collision"), win), LineWidth=1.6);
    plot(ax, movmean(double(outcome == "timeout"), win), LineWidth=1.6);
    ylim(ax, [0 1]);
    legend(ax, ["success" "collision" "timeout"], Location="best");
    title(ax, sprintf("Outcomes (moving average, %d episodes)", win));
    xlabel(ax, "episode");

    ax = nexttile(tl);
    plot(ax, [episodeLog.difficulty], LineWidth=1.6);
    ylim(ax, [0 1.05]);
    title(ax, "Curriculum difficulty");
    xlabel(ax, "episode");

    ax = nexttile(tl);
    plot(ax, movmean([episodeLog.interventions] ./ max([episodeLog.steps], 1), win), LineWidth=1.6);
    title(ax, "Shield interventions per step (moving average)");
    xlabel(ax, "episode");
end
if nargin >= 3 && strlength(string(outFile)) > 0
    ensureFolder(outFile);
    exportgraphics(fig, outFile, Resolution=150);
    fprintf("[viz] saved %s\n", outFile);
end
end
