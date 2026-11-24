function scores = reactiveScores(planner, sectorDist, goalBearing, goalDz)
%REACTIVESCORES  Score of every action for the reactive planner (higher is better).
clr = actionClearances(planner.reqs, sectorDist);
nA = numel(planner.actions);
scores = zeros(1, nA);
for i = 1:nA
    a = planner.actions(i);
    headingErr = abs(wrapAngle(goalBearing - a.yawRate * planner.horizon));
    progress = a.forward * cos(headingErr);
    verticalErr = abs(goalDz - a.vertical * planner.horizon);
    s = planner.wProgress * progress - planner.wHeading * headingErr ...
        - planner.wVertical * min(verticalErr, 3);
    if ~planner.ignoreObstacles
        s = s + planner.wClearance * min(clr(i), planner.clearanceCap) / planner.clearanceCap;
        needed = planner.uavRadius + planner.margin + a.speed * planner.reactionTime;
        if clr(i) < needed
            s = s - 100;
        end
    end
    scores(i) = s;
end
end
