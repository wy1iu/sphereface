% --------------------------------------------------------
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to detect the faces in training & testing datasets (CASIA & LFW).
% Face and facial landmark detection are performed by MTCNN 
% (paper: http://kpzhang93.github.io/papers/spl.pdf, 
%  code: https://github.com/kpzhang93/MTCNN_face_detection_alignment).
%
% Note:
% If you want to use this script for other dataset, please make sure
% (a) the dataset is structured as `dataset/idnetity/image`, e.g. `casia/id/001.jpg`
% (b) the folder name and image format (bmp, png, etc.) are correctly specified.
% 
% Usage:
% cd $SPHEREFACE_ROOT/preprocess
% run code/face_detect_demo.m
% --------------------------------------------------------

function face_detect_demo()

clear;clc;close all;
cd('../');

%% collect a image list of CASIA & LFW
trainList = collectData(fullfile(pwd, 'data/CASIA-WebFace'), 'CASIA-WebFace');
testList  = collectData(fullfile(pwd, 'data/lfw'), 'lfw');

%% mtcnn settings
minSize   = 20;
factor    = 0.85;
threshold = [0.6 0.7 0.9];

%% add toolbox paths
matCaffe       = fullfile(pwd, '../tools/caffe-sphereface/matlab');
pdollarToolbox = fullfile(pwd, '../tools/toolbox');
MTCNN          = fullfile(pwd, '../tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1');
addpath(genpath(matCaffe));
addpath(genpath(pdollarToolbox));
addpath(genpath(MTCNN));

%% caffe settings
gpu = 1;
if gpu
   gpu_id = 0;
   caffe.set_mode_gpu();
   caffe.set_device(gpu_id);
else
   caffe.set_mode_cpu();
end
caffe.reset_all();
modelPath = fullfile(pwd, '../tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model');
PNet = caffe.Net(fullfile(modelPath, 'det1.prototxt'), ...
                 fullfile(modelPath, 'det1.caffemodel'), 'test');
RNet = caffe.Net(fullfile(modelPath, 'det2.prototxt'), ...
                 fullfile(modelPath, 'det2.caffemodel'), 'test');
ONet = caffe.Net(fullfile(modelPath, 'det3.prototxt'), ...
                 fullfile(modelPath, 'det3.caffemodel'), 'test');

%% face and facial landmark detection
dataList = [trainList; testList];
for i = 1:length(dataList)
    fprintf('detecting the %dth image...\n', i);
    % load image
    img = imread(dataList(i).file);
    if size(img, 3)==1
       img = repmat(img, [1,1,3]);
    end
    % detection
    [bboxes, landmarks] = detect_face(img, minSize, PNet, RNet, ONet, threshold, false, factor);

    if size(bboxes, 1)>1
       % pick the face closed to the center
       center   = size(img) / 2;
       distance = sum(bsxfun(@minus, [mean(bboxes(:, [2, 4]), 2), ...
                                      mean(bboxes(:, [1, 3]), 2)], center(1:2)).^2, 2);
       [~, Ix]  = min(distance);
       dataList(i).facial5point = reshape(landmarks(:, Ix), [5, 2]);
    elseif size(bboxes, 1)==1
       dataList(i).facial5point = reshape(landmarks, [5, 2]);
    else
       dataList(i).facial5point = [];
    end
end
save result/dataList.mat dataList

end


function list = collectData(folder, name)
    subFolders = struct2cell(dir(folder))';
    subFolders = subFolders(3:end, 1);
    files      = cell(size(subFolders));
    for i = 1:length(subFolders)
        fprintf('%s --- Collecting the %dth folder (total %d) ...\n', name, i, length(subFolders));
        subList  = struct2cell(dir(fullfile(folder, subFolders{i}, '*.jpg')))';
        files{i} = fullfile(folder, subFolders{i}, subList(:, 1));
    end
    files      = vertcat(files{:});
    dataset    = cell(size(files));
    dataset(:) = {name};
    list       = cell2struct([files dataset], {'file', 'dataset'}, 2);
end