function [d, kind] = worldClearance(world, p)
%WORLDCLEARANCE  Distance from point p to the closest surface and its type.
%   kind is "ground", "ceiling", "wall", "pillar" or "mover".

x = p(1); y = p(2); z = p(3);
d = z;
kind = "ground";
c = world.ceiling - z;
if c < d
    d = c;
    kind = "ceiling";
end
w = min([x, world.size - x, y, world.size - y]);
if w < d
    d = w;
    kind = "wall";
end
if ~isempty(world.pillars)
    dp = min(pillarDistances(world, p));
    if dp < d
        d = dp;
        kind = "pillar";
    end
end
if ~isempty(world.moverPos)
    dm = min(vecnorm(world.moverPos - [x y z], 2, 2) - world.moverRadius(:));
    if dm < d
        d = dm;
        kind = "mover";
    end
end
end
