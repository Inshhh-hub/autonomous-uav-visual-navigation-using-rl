function reqs = actionSectorRequirements(rows, cols, actions)
%ACTIONSECTORREQUIREMENTS  Which image sectors each action flies into, and how fast.
%
%   reqs{i} is a struct array with fields
%     sectors - sector indices swept by action i
%     speed   - speed of the UAV towards those sectors [m/s]
%   Hover has no requirements.

reqs = cell(1, numel(actions));
for i = 1:numel(actions)
    a = actions(i);
    switch a.name
        case "forward"
            groups = {"mid", "center", "forward"};
        case "forward_left"
            groups = {"mid", "left", "forward"};
        case "forward_right"
            groups = {"mid", "right", "forward"};
        case "turn_left"
            groups = {"mid", "center_left", "forward"};
        case "turn_right"
            groups = {"mid", "center_right", "forward"};
        case "ascend"
            groups = {"top", "center", "total"; "mid", "center", "forward"};
        case "descend"
            groups = {"bottom", "center", "total"; "mid", "center", "forward"};
        otherwise
            groups = cell(0, 3);
    end
    R = struct("sectors", {}, "speed", {});
    for g = 1:size(groups, 1)
        if groups{g, 3} == "forward"
            sp = abs(a.forward);
        else
            sp = a.speed;
        end
        R(end + 1).sectors = sectorIndices(rows, cols, groups{g, 1}, groups{g, 2}); %#ok<AGROW>
        R(end).speed = sp;
    end
    reqs{i} = R;
end
end
