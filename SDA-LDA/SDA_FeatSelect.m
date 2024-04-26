% function [idx_sda sda_feature] = SDA_FeatSelect(features, Labels)

close all, clear all, clc;
currentFolder = pwd;
addpath(genpath(currentFolder))

file_path = 'E:\Users\zhangzhen\Desktop\wenxin\PScL-HDeep-master\lib\4_featureSelectionCode\Normalized_ALL_data_origin_K.xlsx';
[num_data, txt_data, raw_data] = xlsread(file_path);
features = raw_data(2:end, 1:end-1);
features = cell2mat(features);

% 提取原始标签数据
raw_labels = raw_data(2:end, end);
% 将标签数据转换为字符向量
Labels = cell(size(raw_labels));
for i = 1:numel(raw_labels)
    if isnumeric(raw_labels{i}) || islogical(raw_labels{i})
        Labels{i} = num2str(raw_labels{i});  % 将数字转换为字符串
    elseif isdatetime(raw_labels{i})
        Labels{i} = datestr(raw_labels{i});  % 将日期转换为字符串
    else
        Labels{i} = char(raw_labels{i});      % 其他类型保持不变
    end
end

u = unique(Labels);
feat = [];
feat{length(u)} = [];
for i=1:length(u)
%     feat{i} = features( strcmp(Labels, u{i}), :);
%     feat{i} = features( Labels==u(i), :);
    idx = strcmp(Labels, u{i});
    feat{i} = features(idx, :);
end
out_path = 'E:\Users\zhangzhen\Desktop\wenxin\PScL-HDeep-master\lib\4_featureSelectionCode\';
logfilename = [out_path 'Normalized_ALL_data_origin_K.txt'];
idx_sda = ml_stepdisc(feat,logfilename);
sda_feature = features(:,idx_sda);

header = raw_data(1, :);
labels_column = raw_data(1:end, end);
header_cell = header(idx_sda);

% 串联 header_cell 和 sda_feature
sda_feature_with_header = [header_cell; num2cell(sda_feature)];

% 将 sda_feature_with_header 和 labels_column 进行横向串联
sda_feature_with_header = [sda_feature_with_header, labels_column];

% 保存带有标头备注和类别的特征数据到 Excel 文件
xlswrite('E:\Users\zhangzhen\Desktop\wenxin\PScL-HDeep-master\lib\4_featureSelectionCode\SDA_Normalized_ALL_data_origin_K.xlsx', sda_feature_with_header);







