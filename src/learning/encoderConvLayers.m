function layers = encoderConvLayers(imgSize, frameStack)
%ENCODERCONVLAYERS  Convolutional trunk shared by the encoder, detector and DQN critic.
%   The same layer names are used in every network, so weights learned by the
%   self-supervised encoder can be copied into the detector and the critic.
%   48 x 48 input -> 24 -> 12 -> 6 -> 6 (64 channels).
layers = [
    imageInputLayer([imgSize imgSize frameStack], Normalization="none", Name="image")
    convolution2dLayer(5, 32, Stride=2, Padding="same", Name="conv1")
    reluLayer(Name="relu1")
    convolution2dLayer(3, 64, Stride=2, Padding="same", Name="conv2")
    reluLayer(Name="relu2")
    convolution2dLayer(3, 64, Stride=2, Padding="same", Name="conv3")
    reluLayer(Name="relu3")
    convolution2dLayer(3, 64, Stride=1, Padding="same", Name="conv4")
    reluLayer(Name="relu4")];
end
