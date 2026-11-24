function q = duelingCombine(v, a)
%DUELINGCOMBINE  Dueling aggregation Q = V + A - mean(A) (Wang et al., 2016).
%   v: 1 x B and a: nActions x B formatted dlarrays ("CB").
q = v + a - mean(a, 1);
end
