function Y = oneHot(labels, nClasses)
%ONEHOT  nClasses x B single one-hot matrix for integer labels 1..nClasses.
labels = labels(:)';
Y = zeros(nClasses, numel(labels), "single");
Y(sub2ind(size(Y), labels, 1:numel(labels))) = 1;
end
