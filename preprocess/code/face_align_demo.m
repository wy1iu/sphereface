% --------------------------------------------------------
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to align the faces by similarity transformation.
% Here we only use five facial landmarks (two eyes, nose point and two mouth corners).
%
% Usage:
% cd $SPHEREFACE_ROOT/preprocess
% run code/face_align_demo.m
% --------------------------------------------------------

function face_align_demo()

clear;clc;close all;
cd('../');

load('result/dataList.mat')
%% alignment settings
imgSize     = [112, 96];
coord5point = [30.2946, 51.6963;
               65.5318, 51.5014;
               48.0252, 71.7366;
               33.5493, 92.3655;
               62.7299, 92.2041];

%% face alignment
for i = 1:length(dataList)
    fprintf('aligning the %dth image...\n', i);
    if isempty(dataList(i).facial5point)
       continue;
    end
    dataList(i).facial5point = double(dataList(i).facial5point);
    % load and crop image
    img      = imread(dataList(i).file);
    transf   = cp2tform(dataList(i).facial5point, coord5point, 'similarity');
    cropImg  = imtransform(img, transf, 'XData', [1 imgSize(2)],...
                                        'YData', [1 imgSize(1)], 'Size', imgSize);
    % save image
    [sPathStr, name, ext] = fileparts(dataList(i).file);
    tPathStr = strrep(sPathStr, '/data/', '/result/');
    tPathStr = strrep(tPathStr, dataList(i).dataset, [dataList(i).dataset '-112X96']);
    if ~exist(tPathStr, 'dir')
       mkdir(tPathStr)
    end
    imwrite(cropImg, fullfile(tPathStr, [name, '.jpg']), 'jpg');
end

end