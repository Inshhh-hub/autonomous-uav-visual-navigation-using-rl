function saveGif(frames, file, fps)
%SAVEGIF  Write a cell array of RGB frames to an animated GIF.
if nargin < 3
    fps = 10;
end
if isempty(frames)
    return
end
ensureFolder(file);
delay = 1 / fps;
for i = 1:numel(frames)
    [A, map] = rgb2ind(frames{i}, 128);
    if i == 1
        imwrite(A, map, file, "gif", LoopCount=Inf, DelayTime=delay);
    else
        imwrite(A, map, file, "gif", WriteMode="append", DelayTime=delay);
    end
end
fprintf("[viz] saved %s\n", file);
end
