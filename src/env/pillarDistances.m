function d = pillarDistances(world, p)
%PILLARDISTANCES  Distance from point p (3-vector) to the surface of every pillar (N x 1).
if isempty(world.pillars)
    d = zeros(0, 1);
    return
end
dh = hypot(p(1) - world.pillars(:, 1), p(2) - world.pillars(:, 2)) - world.pillars(:, 3);
dz = p(3) - world.pillars(:, 4);
d = dh;
above = dz > 0;
d(above) = hypot(max(dh(above), 0), dz(above));
end
