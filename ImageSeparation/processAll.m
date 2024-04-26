function processAll()
%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
currentFolder = pwd; 
addpath(genpath(currentFolder));
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
%% 判断根目录下的文件夹是否存在，不存在则创建
rootFolder = fullfile(pwd, 'ProcessMultiLabel6');
if ~exist(rootFolder, 'dir')
    mkdir(rootFolder);
end
% 判断DNA、protein和unmix_composition文件夹是否存在，不存在则创建
subfolders = {'DNA', 'protein', 'unmix_composition'};
for i = 1:numel(subfolders)
    subfolderPath = fullfile(rootFolder, subfolders{i});
    if ~exist(subfolderPath, 'dir')
        mkdir(subfolderPath);
    end
    % 判断Cy、Er、Go、Mi、Nu、Ve和others文件夹是否存在，不存在则创建
    subsubfolders = {'Cy', 'Er', 'Go', 'Mi', 'Nu', 'Ve', 'others'};
    for j = 1:numel(subsubfolders)
        subsubfolderPath = fullfile(subfolderPath, subsubfolders{j});
        if ~exist(subsubfolderPath, 'dir')
            mkdir(subsubfolderPath);
        end
    end
end
%% 处理代码（保存W）
% W = getBasis();
% % 保存的具体代码ZZ
% existingFiles = dir('ZZWbasis*.mat');
% existingFileCount = numel(existingFiles);
% if existingFileCount > 0
%     newFileName = ['ZZWbasis', num2str(existingFileCount), '.mat'];
% else
%     newFileName = 'ZZWbasis.mat';
% end
% save(newFileName, 'W');
% disp(['W 矩阵已保存为 ', newFileName]);

%% 处理代码
load('Wbasis.mat');
params = struct();
params.W = W; % 假设加载的矩阵存储在变量 W 中
params.UMETHOD = 'nmf';
imgPath='./ALLdata/';
imgDataDir  = dir(imgPath);             % 遍历所有文件
for i = 1:length(imgDataDir)
    if(isequal(imgDataDir(i).name,'.')||... % 去除系统自带的两个隐文件夹
       isequal(imgDataDir(i).name,'..')||...
       ~imgDataDir(i).isdir)                % 去除遍历中不是文件夹的
           continue;
    end
%     disp(imgDataDir(i));
    imgDir = dir([imgPath imgDataDir(i).name '/*.jpg']); 
    for j =1:length(imgDir)  
        % 遍历所有图片
        disp(['处理第 ' num2str(j) ' 张图像']);
        readPath = [imgPath imgDataDir(i).name '/' imgDir(j).name];%E:/hpaData/hpaAll../UBTF_cerebral+cortex_Nucleolus3.jpg

        imgWritePath1=strrep(readPath,'ALLdata','ProcessMultiLabel6/DNA');
        imgWritePath2=strrep(readPath,'ALLdata','ProcessMultiLabel6/protein');
        imgWritePath3=strrep(readPath,'ALLdata','ProcessMultiLabel6/unmix_composition');
        processImage(readPath,imgWritePath1,params,'DNA');
        processImage(readPath,imgWritePath2,params,'protein');
        processImage(readPath,imgWritePath3,params,'unmix_composition');
    end
end

end

