function world = stepMovers(world, dt, rs, turnProb, zRange)
%STEPMOVERS  Advance the moving obstacles by dt seconds.
%   Movers fly with constant speed, change heading at random (probability
%   turnProb per step) and bounce off the walls, altitude limits and pillars.

m = size(world.moverPos, 1);
if m == 0
    return
end
if turnProb > 0
    turn = rand(rs, m, 1) < turnProb;
    if any(turn)
        ang = (rand(rs, nnz(turn), 1) * 2 - 1) * pi / 3;
        vx = world.moverVel(turn, 1);
        vy = world.moverVel(turn, 2);
        world.moverVel(turn, 1) = cos(ang) .* vx - sin(ang) .* vy;
        world.moverVel(turn, 2) = sin(ang) .* vx + cos(ang) .* vy;
    end
end

world.moverPos = world.moverPos + world.moverVel * dt;
r = world.moverRadius(:);
lo = [r, r, repmat(zRange(1), m, 1)];
hi = [world.size - r, world.size - r, repmat(zRange(2), m, 1)];
below = world.moverPos < lo;
above = world.moverPos > hi;
world.moverPos(below) = lo(below);
world.moverVel(below) = abs(world.moverVel(below));
world.moverPos(above) = hi(above);
world.moverVel(above) = -abs(world.moverVel(above));

if ~isempty(world.pillars)
    for i = 1:m
        p = world.moverPos(i, :);
        dvec = p(1:2) - world.pillars(:, 1:2);
        dist = vecnorm(dvec, 2, 2);
        pen = world.pillars(:, 3) + r(i) - dist;
        hit = pen > 0 & (p(3) - r(i)) < world.pillars(:, 4);
        if ~any(hit)
            continue
        end
        pen(~hit) = -inf;
        [~, j] = max(pen);
        nrm = dvec(j, :) / max(dist(j), 1e-6);
        world.moverPos(i, 1:2) = world.pillars(j, 1:2) + nrm * (world.pillars(j, 3) + r(i) + 1e-3);
        vn = world.moverVel(i, 1:2) * nrm';
        if vn < 0
            world.moverVel(i, 1:2) = world.moverVel(i, 1:2) - 2 * vn * nrm;
        end
    end
end
end
