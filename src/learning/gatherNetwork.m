function net = gatherNetwork(net)
%GATHERNETWORK  Move all learnables to CPU memory (so checkpoints load without a GPU).
net = dlupdate(@gather, net);
end
