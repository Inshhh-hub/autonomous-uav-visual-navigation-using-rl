function ds = makeDataset(data, frameStack, valFraction, seed)
%MAKEDATASET  Index a rollout file for self-supervised and detector training.
%
%   data: struct loaded from collectData (frames H x W x T uint8, actions,
%   episode, sectorDist, risk, meta). Frame stacks are rebuilt on the fly
%   exactly like the environment does (first frame repeated at episode start).
%   Train / validation split is done by episode.

ds = struct();
ds.frames = data.frames;
ds.actions = double(data.actions(:));
ds.sectorDist = double(data.sectorDist);           % S x T
ds.risk = double(data.risk(:))';                    % 1 x T
ds.frameStack = frameStack;
ds.maxDepth = data.meta.maxDepth;
ds.occupancyThreshold = data.meta.occupancyThreshold;
ds.nActions = data.meta.nActions;
ds.nSectors = size(ds.sectorDist, 1);

ep = double(data.episode(:));
T = numel(ep);
group = cumsum([1; diff(ep) ~= 0]);
starts = [1; find(diff(ep) ~= 0) + 1];
ds.epStart = starts(group);

nEp = numel(starts);
rs = RandStream("mt19937ar", "Seed", seed);
nVal = 0;
if nEp > 1 && valFraction > 0
    nVal = min(max(1, round(valFraction * nEp)), nEp - 1);
end
perm = randperm(rs, nEp);
isVal = ismember(group, perm(1:nVal));
hasNext = ds.actions >= 1;
ds.trainIdx = find(~isVal);
ds.valIdx = find(isVal);
ds.transTrain = find(~isVal & hasNext);
ds.transVal = find(isVal & hasNext);
if isempty(ds.valIdx)                     % tiny datasets: validate on train
    ds.valIdx = ds.trainIdx;
    ds.transVal = ds.transTrain;
end
ds.numFrames = T;
end
