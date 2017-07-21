clear;clc;close all;
cd('../');

folder    = fullfile(pwd, 'data/CASIA-WebFace-112X96');
subFolder = struct2cell(dir(folder))';
subFolder = subFolder(3:end, 1);

% exclude the identities appearing in LFW dataset
indx = ismember(subFolder, [{'0166921'}, {'1056413'}, {'1193098'}]);
subFolder(indx) = [];

% create the list for trianing
fid       = fopen('data/CASIA-WebFace-112X96.txt', 'w');
for i = 1:length(subFolder)
    fprintf('Collecting the %dth folder (total %d) ...\n', i, length(subFolder));
    subList     = struct2cell(dir(fullfile(folder, subFolder{i})))';
    fileNames   = fullfile(folder, subFolder{i}, subList(3:end, 1));
    for j = 1:size(fileNames, 1)
        fprintf(fid, '%s %d\n', fileNames{j}, i-1);
    end
end
fclose(fid);
