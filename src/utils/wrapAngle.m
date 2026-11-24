function a = wrapAngle(a)
%WRAPANGLE  Wrap angles to [-pi, pi).
a = mod(a + pi, 2*pi) - pi;
end
