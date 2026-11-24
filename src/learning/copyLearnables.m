function [dst, nCopied] = copyLearnables(dst, src)
%COPYLEARNABLES  Copy parameters from src to dst for layers with identical names and sizes.
%   Used to transfer the self-supervised encoder into the detector and critic.
L = dst.Learnables;
S = src.Learnables;
values = L.Value;
nCopied = 0;
for i = 1:height(L)
    j = find(string(S.Layer) == string(L.Layer(i)) & string(S.Parameter) == string(L.Parameter(i)), 1);
    if ~isempty(j) && isequal(size(S.Value{j}), size(L.Value{i}))
        values{i} = S.Value{j};
        nCopied = nCopied + 1;
    end
end
dst = setLearnableValues(dst, values);
end
