function actions = uavActions(speedScale)
%UAVACTIONS  Discrete high-level action set (velocity / yaw-rate set-points).
%
%   Each action is a set-point for the low-level flight controller, like a
%   companion computer commanding a PX4/ArduPilot autopilot in offboard mode.
%   Fields: name, forward [m/s], vertical [m/s, + up], yawRate [rad/s, + left], speed.

if nargin < 1
    speedScale = 1;
end
names   = ["forward" "forward_left" "forward_right" "turn_left" "turn_right" "ascend" "descend" "hover"];
forward = [2.5 1.8 1.8 0.3 0.3 1.0 1.0 0.0] * speedScale;
vertical = [0 0 0 0 0 1.2 -1.2 0] * speedScale;
yawRate = [0 0.8 -0.8 1.5 -1.5 0 0 0];
speed = sqrt(forward.^2 + vertical.^2);
actions = struct("name", num2cell(names), "forward", num2cell(forward), ...
    "vertical", num2cell(vertical), "yawRate", num2cell(yawRate), "speed", num2cell(speed));
end
