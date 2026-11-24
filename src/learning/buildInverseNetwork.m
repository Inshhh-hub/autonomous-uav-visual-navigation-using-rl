function net = buildInverseNetwork(featureDim, nActions)
%BUILDINVERSENETWORK  Inverse-dynamics head: [z_t; z_t+1] -> action logits.
layers = [
    featureInputLayer(2 * featureDim, Name="z_pair")
    fullyConnectedLayer(256, Name="inv_fc1")
    reluLayer(Name="inv_relu")
    fullyConnectedLayer(nActions, Name="inv_out")];
net = dlnetwork(layers);
end
