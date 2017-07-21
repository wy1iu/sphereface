% --------------------------------------------------------
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to evaluate the performance of the trained model on LFW dataset.
% We perform 10-fold cross validation, using cosine similarity as metric.
% More details about the testing protocol refers to LFW (http://vis-www.cs.umass.edu/lfw/).
% 
% Usage:
% cd $SPHEREFACE_ROOT/test
% run code/evaluation.m
% --------------------------------------------------------

function evaluation()

clear;clc;close all;
cd('../')

%% caffe setttings
matCaffe = fullfile(pwd, '../tools/caffe/matlab');
addpath(genpath(matCaffe));
gpu = 1;
if gpu
   gpu_id = 0;
   caffe.set_mode_gpu();
   caffe.set_device(gpu_id);
else
   caffe.set_mode_cpu();
end
caffe.reset_all();

model   = '../train/code/sphereface/sphereface_deploy.prototxt';
weights = '../train/result/sphereface/sphereface_model_iter_28000.caffemodel';
net     = caffe.Net(model, weights, 'test');

%% compute features and scores
pairs = parseList('data/pairs.txt', fullfile(pwd, 'data/lfw-112X96'));
for i = 1:length(pairs)
    fprintf('extracting deep features from the %dth face pair...\n', i);
    pairs(i).feature_l = extractDeepFeature(pairs(i).file_l, net);
    pairs(i).feature_r = extractDeepFeature(pairs(i).file_r, net);
    pairs(i).score     = pairs(i).feature_l' * pairs(i).feature_r / ...
                         norm(pairs(i).feature_l) / norm(pairs(i).feature_r);
end

%% 10-fold evaluation
pairs = struct2cell(pairs')';
accs  = zeros(10, 1);
for i = 1:10
    group    = cell2mat(pairs(:,4));
    scores   = cell2mat(pairs(:,7));
    positive = cell2mat(pairs(:,3));
    thr      = getThreshold(scores(group~=i), positive(group~=i), 1000);
    accs(i)  = getAccuracy(scores(group==i), positive(group==i), thr);
end
accs
fprintf('the acc is %f', mean(accs)); 

end


function pairs = parseList(list, folder)
    i    = 0;
    fid  = fopen(list);
    line = fgets(fid);
    while ischar(line)
          strings = strsplit(line, '\t');
          if length(strings) == 3
             i = i + 1;
             pairs(i).file_l   = fullfile(folder, strings{1}, [strings{1}, '_', sprintf('%04d', str2double(strings{2})), '.jpg']);
             pairs(i).file_r   = fullfile(folder, strings{1}, [strings{1}, '_', sprintf('%04d', str2double(strings{3})), '.jpg']);
             pairs(i).positive = 1;
             pairs(i).group    = ceil(i / 300);
          elseif length(strings) == 4
             i = i + 1;
             pairs(i).file_l   = fullfile(folder, strings{1}, [strings{1}, '_', sprintf('%04d', str2double(strings{2})), '.jpg']);
             pairs(i).file_r   = fullfile(folder, strings{3}, [strings{3}, '_', sprintf('%04d', str2double(strings{4})), '.jpg']);
             pairs(i).positive = -1;
             pairs(i).group    = ceil(i / 600);
          end
          line = fgets(fid);
    end
    fclose(fid);
end

function feature = extractDeepFeature(file, net)
    img     = imread(file);
    img     = (img - 127.5)/128;
    img     = permute(img, [2,1,3]);
    img     = img(:,:,[3,2,1]);
    res     = net.forward({img});
    res_    = net.forward({flip(img, 1)});
    feature = [res{1}; res_{1}];
end

function bestThreshold = getThreshold(scores, positive, thrNum)
    accs = zeros(thrNum, 1);
    thrs = (1:thrNum) / thrNum;
    for i = 1:thrNum
        accs(i) = getAccuracy(scores, positive, thrs(i));        
    end
    [~, indx]     = max(accs);
    bestThreshold = thrs(indx);
end

function acc = getAccuracy(scores, positive, threshold)
    acc = (length(find(scores(positive==1)>threshold)) + ...
           length(find(scores(positive==-1)<threshold))) / length(scores);
end
