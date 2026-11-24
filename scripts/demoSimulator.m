%DEMOSIMULATOR  Watch the MATLAB UAV simulation - no training needed.
%
%   Runs the reactive planner with the collision-avoidance safety shield in a
%   few random scenarios (static pillars + moving obstacles), shows the live
%   onboard-camera / top-down-map view and saves GIFs to results/.
%   Run from the project root:  >> demoSimulator

setupProject;
cfg = defaultConfig();
env = UAVNavEnv(cfg, PerceptionSource="oracle", ShieldEnabled=true, Seed=cfg.seed);
policy = makePolicy("reactive_shield", cfg, env);
viewer = SimViewer("UAV visual navigation - reactive planner + safety shield");

for seed = [202 207 212]
    [stats, rec] = runEpisode(env, policy, seed, Record=true, Viewer=viewer);
    if stats.success
        outcome = "GOAL";
    elseif stats.collision
        outcome = "COLLISION (" + stats.obstacle + ")";
    else
        outcome = "TIMEOUT";
    end
    fprintf("seed %d: %s after %d steps, path %.1f m, min clearance %.2f m, shield interventions %d\n", ...
        seed, outcome, stats.steps, stats.pathLength, stats.minClearance, stats.interventions);
    saveGif(rec.frames, fullfile(cfg.paths.results, "demo_reactive_shield_seed" + seed + ".gif"), 10);
end
