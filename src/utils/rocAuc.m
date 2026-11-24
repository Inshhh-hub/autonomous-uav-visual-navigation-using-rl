function auc = rocAuc(labels, scores)
%ROCAUC  Area under the ROC curve (Mann-Whitney statistic, ties averaged). No toolboxes needed.
y = logical(labels(:));
s = double(scores(:));
nPos = sum(y);
nNeg = sum(~y);
if nPos == 0 || nNeg == 0
    auc = NaN;
    return
end
[ss, order] = sort(s);
[~, ~, g] = unique(ss);
avgRank = accumarray(g, (1:numel(s))', [], @mean);
r = zeros(size(s));
r(order) = avgRank(g);
auc = (sum(r(y)) - nPos * (nPos + 1) / 2) / (nPos * nNeg);
end
