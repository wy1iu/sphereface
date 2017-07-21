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
    img     = imread(dataList(i).fileName);
    transf  = cp2tform(dataList(i).facial5point, coord5point, 'similarity');
    cropImg = imtransform(img, transf, 'XData', [1 imgSize(2)],...
                                       'YData', [1 imgSize(1)], 'Size', imgSize);
    % save image
    [sPathStr, name, ext] = fileparts(dataList(i).fileName);
    tPathStr              = strrep(sPathStr, '/data/', '/result/');
    tPathStr              = strrep(tPathStr, dataList(i).dataset, [dataList(i).dataset '-112X96']);
    if ~exist(tPathStr, 'dir')
       mkdir(tPathStr)
    end
    imwrite(cropImg, fullfile(tPathStr, [name, '.jpg']), 'jpg');
end

%%% collect a list of aligned face images in CASIA-WebFace dataset
%listName  = fullfile(pwd, 'result/CASIA-WebFace.txt');
%fid       = fopen(listName, 'w');
%imgDir    = fullfile(pwd, 'result/CASIA-WebFace/');
%imgSubdir = dir(imgDir);
%for i = 3:length(imgSubdir)
%    fprintf('Collecting the %dth folder (total %d) ...\n', i-2, length(imgSubdir)-2);
%    imgSublist = dir(fullfile(imgDir, imgSubdir(i).name));
%    for j = 3:length(imgSublist)
%        fprintf(fid, '%s\n', fullfile(imgDir, imgSubdir(i).name, imgSublist(j).name));
%    end
%end
%fclose(fid);
