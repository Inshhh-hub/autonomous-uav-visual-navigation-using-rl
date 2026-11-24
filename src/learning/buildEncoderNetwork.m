function net = buildEncoderNetwork(cfg)
%BUILDENCODERNETWORK  CNN encoder: frame stack -> featureDim latent (LayerNorm + tanh).
cam = cfg.env.camera;
layers = [
    encoderConvLayers(cam.imgSize, cam.frameStack)
    fullyConnectedLayer(cfg.model.featureDim, Name="enc_fc")
    layerNormalizationLayer(Name="enc_ln")
    tanhLayer(Name="enc_tanh")];
net = dlnetwork(layers);
end
