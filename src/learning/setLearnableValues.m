function net = setLearnableValues(net, values)
%SETLEARNABLEVALUES  Replace all learnable parameters of a dlnetwork.
%   values is a cell array in the row order of net.Learnables.
%   dlupdate visits the learnables in that same order.
k = 0;
net = dlupdate(@assign, net);

    function w = assign(~)
        k = k + 1;
        w = values{k};
    end
end
