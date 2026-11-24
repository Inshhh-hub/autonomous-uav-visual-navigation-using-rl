function net = buildCriticNetwork(cfg, stateDim, nActions)
%BUILDCRITICNETWORK  Q-network with two inputs ("image", "state") and nActions outputs.
%
%   image -> CNN encoder (same layer names as the self-supervised encoder)
%   state -> 2-layer MLP
%   concat -> hidden layer -> dueling head  Q = V + A - mean(A)
cam = cfg.env.camera;
D = cfg.model.featureDim;
Hd = cfg.model.hiddenDim;

imagePath = [
    encoderConvLayers(cam.imgSize, cam.frameStack)
    fullyConnectedLayer(D, Name="enc_fc")
    layerNormalizationLayer(Name="enc_ln")
    tanhLayer(Name="enc_tanh")];
statePath = [
    featureInputLayer(stateDim, Name="state")
    fullyConnectedLayer(128, Name="st_fc1")
    reluLayer(Name="st_relu1")
    fullyConnectedLayer(128, Name="st_fc2")
    reluLayer(Name="st_relu2")];
trunk = [
    concatenationLayer(1, 2, Name="concat")
    fullyConnectedLayer(Hd, Name="trunk_fc")
    reluLayer(Name="trunk_relu")];

lg = layerGraph(imagePath);
lg = addLayers(lg, statePath);
lg = addLayers(lg, trunk);
lg = connectLayers(lg, "enc_tanh", "concat/in1");
lg = connectLayers(lg, "st_relu2", "concat/in2");

if cfg.model.dueling
    valuePath = [
        fullyConnectedLayer(Hd, Name="val_fc1")
        reluLayer(Name="val_relu")
        fullyConnectedLayer(1, Name="val_out")];
    advPath = [
        fullyConnectedLayer(Hd, Name="adv_fc1")
        reluLayer(Name="adv_relu")
        fullyConnectedLayer(nActions, Name="adv_out")];
    combine = functionLayer(@duelingCombine, NumInputs=2, Formattable=true, Name="dueling");
    lg = addLayers(lg, valuePath);
    lg = addLayers(lg, advPath);
    lg = addLayers(lg, combine);
    lg = connectLayers(lg, "trunk_relu", "val_fc1");
    lg = connectLayers(lg, "trunk_relu", "adv_fc1");
    lg = connectLayers(lg, "val_out", "dueling/in1");
    lg = connectLayers(lg, "adv_out", "dueling/in2");
else
    head = [
        fullyConnectedLayer(Hd, Name="q_fc1")
        reluLayer(Name="q_relu")
        fullyConnectedLayer(nActions, Name="q_out")];
    lg = addLayers(lg, head);
    lg = connectLayers(lg, "trunk_relu", "q_fc1");
end
net = dlnetwork(lg);
end
