function net = buildDetectorNetwork(cfg)
%BUILDDETECTORNETWORK  CNN obstacle detector.
%
%   Output (2*S + 1) x B, S = number of image sectors (default 15):
%     rows 1..S      occupancy logits  (obstacle closer than the threshold?)
%     rows S+1..2S   distance logits   (sigmoid -> distance / maxDepth)
%     row  2S+1      crash-risk logit  (crash within the next few steps?)
cam = cfg.env.camera;
nS = cfg.env.sectors.rows * cfg.env.sectors.cols;
layers = [
    encoderConvLayers(cam.imgSize, cam.frameStack)
    fullyConnectedLayer(256, Name="det_fc1")
    reluLayer(Name="det_relu1")
    fullyConnectedLayer(128, Name="det_fc2")
    reluLayer(Name="det_relu2")
    fullyConnectedLayer(2 * nS + 1, Name="det_out")];
net = dlnetwork(layers);
end
