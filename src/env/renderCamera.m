function [depth, shade, kind] = renderCamera(world, cam, pos, yaw, light)
%RENDERCAMERA  Vectorised ray casting of the UAV's forward camera.
%
%   [depth, shade, kind] = renderCamera(world, cam, pos, yaw, light)
%   depth - H x W range along each ray [m] (Inf where nothing is hit)
%   shade - H x W grayscale image in [0, 1] (Lambert shading, textures, fog)
%   kind  - H x W surface id: 0 sky, 1 ground, 2/3 wall, 4 pillar side,
%           5 pillar top, 6 moving obstacle
%
%   Every ray is intersected analytically with the ground plane, the four
%   arena walls, all pillars (vertical cylinders + top caps) and all movers
%   (spheres); the closest hit wins.

if nargin < 5
    light = [0.4; 0.3; 0.866];
end
c = cos(yaw);
s = sin(yaw);
f = cam.dirs(:, 1);
l = cam.dirs(:, 2);
dx = c * f - s * l;
dy = s * f + c * l;
dz = cam.dirs(:, 3);
dirs = [dx dy dz];
P = numel(dx);
px = pos(1); py = pos(2); pz = pos(3);

best = inf(P, 1);
kind = zeros(P, 1);
idx = ones(P, 1);

% ---- ground plane z = 0
t = inf(P, 1);
m = dz < -1e-9;
t(m) = -pz ./ dz(m);
upd = t > 0 & t < best;
best(upd) = t(upd);
kind(upd) = 1;

% ---- walls x = 0 / x = L (kind 2) and y = 0 / y = L (kind 3)
L = world.size;
axisDir = {dx, dy};
origin = [px, py];
for k = 1:2
    d = axisDir{k};
    t = inf(P, 1);
    m = d > 1e-9;
    t(m) = (L - origin(k)) ./ d(m);
    m = d < -1e-9;
    t(m) = -origin(k) ./ d(m);
    zh = pz + t .* dz;
    upd = t > 0 & zh >= 0 & zh <= world.wallHeight & t < best;
    best(upd) = t(upd);
    kind(upd) = 1 + k;
end

% ---- pillars (vertical cylinders)
if ~isempty(world.pillars)
    cx = world.pillars(:, 1)';
    cy = world.pillars(:, 2)';
    r = world.pillars(:, 3)';
    h = world.pillars(:, 4)';
    ox = px - cx;
    oy = py - cy;
    a = dx.^2 + dy.^2;                          % P x 1
    b = 2 * (dx * ox + dy * oy);                % P x N
    cc = ox.^2 + oy.^2 - r.^2;                  % 1 x N
    disc = b.^2 - 4 * a .* cc;
    t = (-b - sqrt(max(disc, 0))) ./ (2 * a);
    zh = pz + t .* dz;
    ok = disc >= 0 & a > 1e-12 & t > 1e-6 & zh >= 0 & zh <= h;
    t(~ok) = inf;
    [tj, j] = min(t, [], 2);
    upd = tj < best;
    best(upd) = tj(upd);
    kind(upd) = 4;
    idx(upd) = j(upd);

    above = pz > h;                             % top caps only visible from above
    if any(above)
        t = (h - pz) ./ dz;
        hx = px + t .* dx - cx;
        hy = py + t .* dy - cy;
        ok = dz < -1e-9 & above & t > 0 & (hx.^2 + hy.^2) <= r.^2;
        t(~ok) = inf;
        [tj, j] = min(t, [], 2);
        upd = tj < best;
        best(upd) = tj(upd);
        kind(upd) = 5;
        idx(upd) = j(upd);
    end
end

% ---- movers (spheres)
if ~isempty(world.moverPos)
    oc = [px py pz] - world.moverPos;            % M x 3
    b = 2 * (dirs * oc');                        % P x M
    cc = sum(oc.^2, 2)' - (world.moverRadius(:)').^2;
    disc = b.^2 - 4 * cc;
    t = (-b - sqrt(max(disc, 0))) / 2;
    t(disc < 0 | t <= 1e-6) = inf;
    [tj, j] = min(t, [], 2);
    upd = tj < best;
    best(upd) = tj(upd);
    kind(upd) = 6;
    idx(upd) = j(upd);
end

% ---- shading
hit = kind > 0;
hp = zeros(P, 3);
hp(hit, :) = [px py pz] + best(hit) .* dirs(hit, :);
n = zeros(P, 3);
alb = zeros(P, 1);

m = kind == 1;                                   % ground: 2 m checkerboard
n(m, 3) = 1;
alb(m) = 0.30 + 0.12 * checker(hp(m, 1) / 2, hp(m, 2) / 2);
m = kind == 2;                                   % walls: bricks
n(m, 1) = -sign(dx(m));
alb(m) = 0.55 + 0.10 * checker(hp(m, 2) / 1.5, hp(m, 3));
m = kind == 3;
n(m, 2) = -sign(dy(m));
alb(m) = 0.55 + 0.10 * checker(hp(m, 1) / 1.5, hp(m, 3));
m = kind == 4;                                   % pillar sides: vertical stripes
if any(m)
    pj = world.pillars(idx(m), :);
    nxy = (hp(m, 1:2) - pj(:, 1:2)) ./ pj(:, 3);
    n(m, 1:2) = nxy;
    stripe = mod(floor(atan2(nxy(:, 2), nxy(:, 1)) / (pi / 4)), 2);
    albP = world.pillarAlbedo(idx(m));
    alb(m) = albP(:) .* (0.8 + 0.2 * stripe);
end
m = kind == 5;
if any(m)
    n(m, 3) = 1;
    albP = world.pillarAlbedo(idx(m));
    alb(m) = 0.9 * albP(:);
end
m = kind == 6;
if any(m)
    rad = world.moverRadius(idx(m));
    n(m, :) = (hp(m, :) - world.moverPos(idx(m), :)) ./ rad(:);
    albM = world.moverAlbedo(idx(m));
    alb(m) = albM(:);
end

light = light(:) / norm(light);
lambert = cam.ambient + (1 - cam.ambient) * min(max(n * light, 0), 1);
img = alb .* lambert;
tt = best;
tt(~hit) = 0;
fog = 1 - exp(-tt / cam.fogDistance);
img = img .* (1 - fog) + 0.75 * fog;
sky = 0.70 + 0.25 * min(max(dz, 0), 1);
img(~hit) = sky(~hit);
img = min(max(img, 0), 1);

depth = reshape(best, cam.H, cam.W);
shade = reshape(img, cam.H, cam.W);
kind = reshape(kind, cam.H, cam.W);
end

function v = checker(a, b)
v = mod(floor(a) + floor(b), 2);
end
