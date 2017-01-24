function [ out ] = presvm( filename )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

img = imread(filename);
out = imresize(img, [32 20]);



end

