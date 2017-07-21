clear;clc;close all;
cd('../');

%% collect a image list of CASIA & LFW
trainList = collectData(fullfile(pwd, 'data/CASIA-WebFace'), 'CASIA-WebFace');
testList  = collectData(fullfile(pwd, 'data/lfw'), 'lfw');

%% mtcnn settings
minSize   = 20;
factor    = 0.709;
threshold = [0.6 0.7 0.9];

%% add toolbox paths
matCaffe       = fullfile(pwd, '../tools/caffe-sphereface/matlab');
pdollarToolbox = fullfile(pwd, '../tools/toolbox');
MTCNN          = fullfile(pwd, '../tools/MTCNN_face_detection_alignment/code/codes/MTCNNv2');
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
modelPath = fullfile(pwd, '../tools/MTCNN_face_detection_alignment/code/codes/MTCNNv2/model');
PNet = caffe.Net(fullfile(modelPath, 'det1.prototxt'), ...
                 fullfile(modelPath, 'det1.caffemodel'), 'test');
RNet = caffe.Net(fullfile(modelPath, 'det2.prototxt'), ...
                 fullfile(modelPath, 'det2.caffemodel'), 'test');
ONet = caffe.Net(fullfile(modelPath, 'det3.prototxt'), ...
                 fullfile(modelPath, 'det3.caffemodel'), 'test');
LNet = caffe.Net(fullfile(modelPath, 'det4.prototxt'), ...
                 fullfile(modelPath, 'det4.caffemodel'), 'test');

%% face and facial landmark detection
dataList = [trainList; testList];
for i = 1:length(dataList)
    fprintf('detecting the %dth image...\n', i);
    % load image
    img = imread(dataList(i).fileName);
    if size(img, 3)==1
       img = repmat(img, [1,1,3]);
    end
    % detection
    [bboxes, landmarks] = detect_face(img, minSize, PNet, RNet, ONet, LNet, threshold, false, factor);

    % pick the largest face
    if size(bboxes, 1)>1
       areas = prod(bboxes(:, 3:4), 2);
       [~, ind] = max(areas);
       dataList(i).facial5point = reshape(landmarks(:, ind), [5, 2]);
    elseif size(bboxes, 1)==1
       dataList(i).facial5point = reshape(landmarks, [5, 2]);
    else
       dataList(i).facial5point = [];
    end
end
save result/dataList.mat dataList


function list = collectData(folder, name)
    subFolder = struct2cell(dir(folder))';
    subFolder = subFolder(3:end, 1);
    fileName  = cell(size(subFolder));
    for i = 1:length(subFolder)
        fprintf('%s --- Collecting the %dth folder (total %d) ...\n', name, i, length(subFolder));
        subList     = struct2cell(dir(fullfile(folder, subFolder{i})))';
        fileName{i} = fullfile(folder, subFolder{i}, subList(3:end, 1));
    end
    fileName   = vertcat(fileName{:});
    dataset    = cell(size(fileName));
    dataset(:) = {name};
    list       = cell2struct([fileName dataset], {'fileName', 'dataset'}, 2);
end
caffe.reset_all();
