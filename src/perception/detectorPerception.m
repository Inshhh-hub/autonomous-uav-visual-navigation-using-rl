function percept = detectorPerception(detNet, frames, ecfg)
%DETECTORPERCEPTION  Run the CNN obstacle detector on the current frame stack.
%
%   frames - H x W x K single image stack in [0, 1]
%   Output fields: sectorDist [m], occupancy (probability), risk (probability)
%   and features (normalised vector appended to the DQN state).

X = dlarray(single(frames), "SSCB");
Y = double(gather(extractdata(predict(detNet, X))));
Y = Y(:);
nS = (numel(Y) - 1) / 2;
occ = sigmoidLocal(Y(1:nS));
distN = sigmoidLocal(Y(nS+1:2*nS));
risk = sigmoidLocal(Y(end));
percept = struct( ...
    "sectorDist", distN * ecfg.camera.maxDepth, ...
    "occupancy", occ, ...
    "risk", risk, ...
    "features", single([distN; occ; risk]));
end

function y = sigmoidLocal(x)
y = 1 ./ (1 + exp(-x));
end
