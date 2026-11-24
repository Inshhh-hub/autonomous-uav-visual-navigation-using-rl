function cam = makeCamera(imgSize, fovDeg, fogDistance)
%MAKECAMERA  Pre-compute the ray direction of every pixel in the camera frame.
%
%   Camera frame: x forward, y left, z up. Pixels are stored column-major
%   (MATLAB order), so reshape(values, H, W) gives the image. Column 1 is the
%   UAV's left side, row 1 is the top of the image.

H = imgSize;
W = imgSize;
tanH = tand(fovDeg / 2);
tanV = tanH * H / W;
u = 1 - 2 * ((0:W-1) + 0.5) / W;      % +1 = left image edge
v = 1 - 2 * ((0:H-1) + 0.5) / H;      % +1 = top image edge
[uu, vv] = meshgrid(u, v);            % H x W
d = [ones(H*W, 1), uu(:) * tanH, vv(:) * tanV];
cam.dirs = d ./ vecnorm(d, 2, 2);     % (H*W) x 3 unit vectors
cam.H = H;
cam.W = W;
cam.fov = deg2rad(fovDeg);
cam.fogDistance = fogDistance;
cam.ambient = 0.35;
end
