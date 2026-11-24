function replay3DScenario(rec, options)
%REPLAY3DSCENARIO  Replay a recorded episode in a UAV Toolbox uavScenario (3D view).
%
%   [~, rec] = runEpisode(env, policy, seed, Record=true);
%   replay3DScenario(rec)                          % animate in a figure
%   replay3DScenario(rec, VideoFile="flight.mp4")  % also save a video
%
%   The arena, pillars and goal are added as static meshes, the UAV as a
%   quadrotor platform and every moving obstacle as its own platform whose
%   pose is updated with move() at each recorded time step.
arguments
    rec struct
    options.Speed (1,1) double = 1
    options.VideoFile (1,1) string = ""
end
if isempty(which("uavScenario"))
    error("replay3DScenario:noToolbox", "UAV Toolbox is required for the 3D replay.");
end
T = size(rec.pos, 2);
scene = uavScenario(UpdateRate=1 / rec.dt, StopTime=T * rec.dt + 1, ReferenceLocation=[0 0 0]);
L = rec.world.size;
addMesh(scene, "polygon", {[0 0; L 0; L L; 0 L], [-0.2 0]}, [0.78 0.78 0.74]);
for i = 1:size(rec.world.pillars, 1)
    p = rec.world.pillars(i, :);
    addMesh(scene, "cylinder", {[p(1) p(2) p(3)], [0 p(4)]}, [0.45 0.45 0.5]);
end
g = rec.goal(:)';
addMesh(scene, "cylinder", {[g(1) g(2) rec.goalRadius], [g(3) - 0.05, g(3) + 0.05]}, [0.1 0.75 0.25]);

uav = uavPlatform("UAV", scene, ReferenceFrame="ENU", ...
    InitialPosition=rec.pos(:, 1)', InitialOrientation=yawQuat(rec.yaw(1)));
updateMesh(uav, "quadrotor", {1.2}, [0.1 0.3 0.9], eye(4));
M = size(rec.moverPos, 1);
movers = cell(1, M);
for j = 1:M
    movers{j} = uavPlatform("Mover" + j, scene, ReferenceFrame="ENU", ...
        InitialPosition=rec.moverPos(j, :, 1));
    updateMesh(movers{j}, "cuboid", {[0.8 0.8 0.8]}, [0.85 0.2 0.2], eye(4));
end
setup(scene);

ax = show3D(scene);
hold(ax, "on");
trail = plot3(ax, rec.pos(1, 1), rec.pos(2, 1), rec.pos(3, 1), "-", Color=[1 0.55 0], LineWidth=2);
view(ax, -35, 35);
axis(ax, "equal");
title(ax, "UAV Toolbox scenario replay");

writer = [];
if strlength(options.VideoFile) > 0
    ensureFolder(options.VideoFile);
    try
        writer = VideoWriter(options.VideoFile, "MPEG-4");
    catch
        writer = VideoWriter(replace(options.VideoFile, ".mp4", ".avi"), "Motion JPEG AVI");
    end
    writer.FrameRate = round(1 / rec.dt);
    open(writer);
end
for t = 1:T
    move(uav, [rec.pos(:, t)', rec.vel(:, t)', zeros(1, 3), yawQuat(rec.yaw(t)), zeros(1, 3)]);
    for j = 1:M
        move(movers{j}, [rec.moverPos(j, :, t), zeros(1, 3), zeros(1, 3), 1, 0, 0, 0, zeros(1, 3)]);
    end
    show3D(scene, Parent=ax, FastUpdate=true);
    set(trail, XData=rec.pos(1, 1:t), YData=rec.pos(2, 1:t), ZData=rec.pos(3, 1:t));
    drawnow limitrate
    if ~isempty(writer)
        writeVideo(writer, getframe(ancestor(ax, "figure")));
    end
    pause(rec.dt / options.Speed);
end
if ~isempty(writer)
    close(writer);
    fprintf("[viz] saved %s\n", writer.Filename);
end
end

function q = yawQuat(yaw)
% quaternion [w x y z] of a rotation by yaw about the z (up) axis
q = [cos(yaw / 2), 0, 0, sin(yaw / 2)];
end
