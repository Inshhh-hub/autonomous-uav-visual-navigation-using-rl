function world = generateWorld(rs, ecfg, nStatic, nDynamic, start, goal)
%GENERATEWORLD  Random pillars and moving obstacles, keeping start and goal free.
%
%   rs    - RandStream used for all sampling (reproducible scenarios)
%   ecfg  - cfg.env
%   start, goal - 3 x 1 positions [m]

world = emptyWorld(ecfg.arenaSize, ecfg.ceiling, ecfg.wallHeight);
L = world.size;
start = start(:)';
goal = goal(:)';

% ------------------------------------------------------------------ pillars
pillars = zeros(0, 4);
attempts = 0;
while size(pillars, 1) < nStatic && attempts < 200 * max(nStatic, 1)
    attempts = attempts + 1;
    r = unif(rs, ecfg.pillarRadius);
    x = unif(rs, [r + 1, L - r - 1]);
    y = unif(rs, [r + 1, L - r - 1]);
    if hypot(x - start(1), y - start(2)) < r + ecfg.startGoalClearance
        continue
    end
    if hypot(x - goal(1), y - goal(2)) < r + ecfg.startGoalClearance
        continue
    end
    if ~isempty(pillars) && any(hypot(x - pillars(:, 1), y - pillars(:, 2)) < ...
            r + pillars(:, 3) + ecfg.minObstacleSpacing)
        continue
    end
    pillars(end + 1, :) = [x, y, r, unif(rs, ecfg.pillarHeight)]; %#ok<AGROW>
end
world.pillars = pillars;
world.pillarAlbedo = 0.45 + 0.45 * rand(rs, size(pillars, 1), 1);

% ------------------------------------------------------------------- movers
pos = zeros(0, 3);
vel = zeros(0, 3);
rad = zeros(0, 1);
seg = goal(1:2) - start(1:2);
segDir = seg / max(norm(seg), 1e-6);
nrm = [-segDir(2), segDir(1)];
tries = 0;
while size(pos, 1) < nDynamic && tries < 200 * max(nDynamic, 1)
    tries = tries + 1;
    r = unif(rs, ecfg.moverRadius);
    speed = unif(rs, ecfg.moverSpeed);
    if rand(rs) < ecfg.crossingRatio
        % spawn beside the start->goal corridor and fly across it
        frac = unif(rs, [0.3 0.85]);
        side = 2 * (rand(rs) < 0.5) - 1;
        xy = start(1:2) + frac * seg + side * unif(rs, [5 12]) * nrm;
        heading = -side * nrm + 0.25 * randn(rs, 1, 2);
        z = start(3) + frac * (goal(3) - start(3)) + 0.7 * randn(rs);
    else
        xy = [unif(rs, [r + 1, L - r - 1]), unif(rs, [r + 1, L - r - 1])];
        heading = randn(rs, 1, 2);
        z = unif(rs, ecfg.moverAltitude);
    end
    heading = heading / max(norm(heading), 1e-6);
    z = min(max(z, ecfg.moverAltitude(1)), ecfg.moverAltitude(2));
    if xy(1) < r + 0.5 || xy(1) > L - r - 0.5 || xy(2) < r + 0.5 || xy(2) > L - r - 0.5
        continue
    end
    p = [xy, z];
    if norm(p - start) < 6
        continue
    end
    if ~isempty(world.pillars) && any(pillarDistances(world, p) < r + 0.5)
        continue
    end
    pos(end + 1, :) = p; %#ok<AGROW>
    vel(end + 1, :) = [heading * speed, 0.15 * randn(rs)]; %#ok<AGROW>
    rad(end + 1, 1) = r; %#ok<AGROW>
end
world.moverPos = pos;
world.moverVel = vel;
world.moverRadius = rad;
world.moverAlbedo = 0.08 + 0.17 * rand(rs, size(pos, 1), 1);
end

function v = unif(rs, range)
v = range(1) + (range(2) - range(1)) * rand(rs);
end
