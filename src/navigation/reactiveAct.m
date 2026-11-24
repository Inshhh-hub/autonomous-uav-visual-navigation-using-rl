function a = reactiveAct(planner, sectorDist, goalBearing, goalDz, mask)
%REACTIVEACT  Best action of the reactive planner, optionally restricted by a safety mask.
scores = reactiveScores(planner, sectorDist, goalBearing, goalDz);
if nargin >= 5 && ~isempty(mask) && any(mask)
    scores(~mask) = -inf;
end
[~, a] = max(scores);
end
