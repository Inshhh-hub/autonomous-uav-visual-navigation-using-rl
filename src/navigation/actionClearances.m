function clr = actionClearances(reqs, sectorDist)
%ACTIONCLEARANCES  Smallest obstacle distance in the sectors each action flies into.
%   Returns a 1 x numActions vector (Inf for actions without requirements, i.e. hover).
clr = inf(1, numel(reqs));
for i = 1:numel(reqs)
    R = reqs{i};
    for r = 1:numel(R)
        clr(i) = min(clr(i), min(sectorDist(R(r).sectors)));
    end
end
end
