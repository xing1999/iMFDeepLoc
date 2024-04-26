%%  清空环境变量和添加变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行 
currentFolder = pwd;
addpath(genpath(currentFolder))

%%  创建时间命名文件夹
currentDateTime = datestr(now, '()zzyyyymmdd_HHMMSS');
outputFolder = fullfile(pwd, currentDateTime);
mkdir(outputFolder);

%% 导入数据
data = 'FEC-Net_ALL_data.xlsx';
res = xlsread(data);
res2=res(:, end);
size1=size(res,1);

%% 7:3划分数据集为训练集和测试集
P = size1*0.7;
P = fix(P);
size2=size(res,2);
temp = randperm(size1);
P_train = res(temp(1:P),1:size2-1)';
T_train = res2(temp(1:P),1)';
M = size(P_train, 2);
P_test = res(temp(P:end),1:size2-1)';
T_test = res2(temp(P:end),1)';

N = size(P_test, 2);
num_class =length(unique(T_test));
num_dim = size(res, 2) - 1;             
num_res = size(res, 1);                

train_data = [P_train', T_train']; % 将P_train和T_train拼接成一个矩阵
train_data_filename = fullfile(outputFolder, 'train_data.xlsx');
% xlswrite(train_data_filename, train_data, 1, 'A1');
save(fullfile(outputFolder, 'train_data.mat'),"train_data");

test_data = [P_test', T_test']; % 将P_train和T_train拼接成一个矩阵
test_data_filename = fullfile(outputFolder, 'test_data.xlsx');
% xlswrite(test_data_filename, test_data, 1, 'A1');
save(fullfile(outputFolder, 'test_data.mat'),"test_data");

%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

save(fullfile(outputFolder, 'ps_input.mat'),"ps_input");
%%  数据平铺
%   将数据平铺成1维数据只是一种处理方式
%   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
%   但是应该始终和输入层数据结构保持一致
P_train =  double(reshape(P_train, num_dim, 1, 1, M));
P_test  =  double(reshape(P_test , num_dim, 1, 1, N));

%%  数据格式转换
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end
    
%%  建立模型
lgraph = layerGraph();                                  % 建立空白网络结构

tempLayers = [
    sequenceInputLayer([num_dim, 1, 1], "Name", "sequence")  % 建立输入层，输入数据结构为[num_dim, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                % 建立序列折叠层
lgraph = addLayers(lgraph, tempLayers);                      % 将上述网络结构加入空白结构中

tempLayers = [
    convolution2dLayer([2, 1], 16, "Name", "conv_1")    % 建立卷积层, 卷积核大小[2, 1], 32个特征图
    dropoutLayer(0.2,"Name","dropout_1")  %zz
    batchNormalizationLayer("Name","batchnorm_1")  %zz
    reluLayer("Name", "relu_1")                         % Relu 激活层
    convolution2dLayer([2, 1], 32, "Name", "conv_2")];  % 建立卷积层, 卷积核大小[2, 1], 64个特征图
lgraph = addLayers(lgraph, tempLayers);                 % 将上述网络结构加入空白结构中

tempLayers = TransposeLayer("tans_1");                  % 维度交换层, 从而在空间维度进行GAP, 而不是通道维度
lgraph = addLayers(lgraph, tempLayers);                 % 将上述网络结构加入空白结构中

tempLayers = [globalAveragePooling2dLayer("Name", "gapool")];             % 全局平均池化层
lgraph = addLayers(lgraph, tempLayers);                      % 将上述网络结构加入空白结构中

tempLayers = [globalMaxPooling2dLayer("Name", "gmpool")];     % 全局最大池化层
lgraph = addLayers(lgraph, tempLayers);                      % 将上述网络结构加入空白结构中

tempLayers = [
    concatenationLayer(1, 2, "Name", "concat")                         % 拼接 GAP 和 GMP 后的结果
    TransposeLayer("tans_2")                                           % 维度交换层, 恢复原始维度
    convolution2dLayer([1, 1], 1, "Name", "conv_3", "Padding", "same") % 建立卷积层, 通道数目变换
    sigmoidLayer("Name", "sigmoid")];                                  % sigmoid 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的空间注意力
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
    flattenLayer("Name", "flatten")                                    % 网络铺平层
    bilstmLayer(13, "Name", "BiLSTM", "OutputMode", "last")            % BiLSTM层
    dropoutLayer(0.5,"Name","dropout_5")           %zz
    batchNormalizationLayer("Name","batchnorm_5")  %zz
    fullyConnectedLayer(num_class, "Name", "fc")                       % 全连接层
    softmaxLayer("Name", "softmax")                                    % softmax激活层
    classificationLayer("Name", "classification")];                    % 分类层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize");
                                                                       % 折叠层输出 连接 反折叠层输入
lgraph = connectLayers(lgraph, "conv_2", "tans_1");                    % 卷积层输出 链接 维度变换层
lgraph = connectLayers(lgraph, "conv_2", "multiplication/in2");        % 卷积层输出 链接 点乘层(注意力)输入2
lgraph = connectLayers(lgraph, "tans_1", "gapool");                    % 维度变换层 链接 GAP
lgraph = connectLayers(lgraph, "tans_1", "gmpool");                    % 维度变换层 链接 GMP
lgraph = connectLayers(lgraph, "gapool", "concat/in2");                % GAP 链接 拼接层1
lgraph = connectLayers(lgraph, "gmpool", "concat/in1");                % GMP 链接 拼接层2
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");        % sigmoid 链接 相乘层
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");       % 点乘输出

%%  参数设置
options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MaxEpochs', 10, ...                  % 最大迭代次数
    'InitialLearnRate', 0.004, ...           % 初始学习率
    'L2Regularization', 0.6113e-02, ...         % L2正则化参数
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.003, ...        % 学习率下降因子 0.1
    'LearnRateDropPeriod', 450, ...        % 经过450次训练后 学习率为 0.001 * 0.5
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'ValidationPatience', 200, ...          % 关闭验证
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false, ...                  % 不显示详细信息
    'GradientThreshold',inf, ...
    'ValidationData',{p_test,t_test}, ...
    'ExecutionEnvironment', 'auto');        % 执行环境设为 GPU

%%  训练模型 (并保存训练过程中的数据)
% net = trainNetwork(p_train, t_train, lgraph, options);
[net,info] = trainNetwork(p_train, t_train, lgraph, options);
save(fullfile(outputFolder, 'info.mat'),"info");
auto_loss = info.TrainingLoss';
auto_accuracy = info.TrainingAccuracy';
test_loss = info.ValidationLoss';
test_accuracy = info.ValidationAccuracy';

iterations = (1:numel(auto_loss))';
data = table(iterations, auto_accuracy, auto_loss, 'VariableNames', {'迭代次数', '准确率', '损失'});
filename = fullfile(outputFolder, 'train_process_data.xlsx');
writetable(data, filename);


figure(1)
plot(info.TrainingLoss);%%画出训练的loss
hold on
saveas(gcf, fullfile(outputFolder, 'loss.png'));
figure(2)
plot(info.TrainingAccuracy);%%画出训练的准确率
saveas(gcf, fullfile(outputFolder, 'Accuracy.png'));
%%  预测模型
t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test ); 

save(fullfile(outputFolder, 't_sim1.mat'),"t_sim1");
save(fullfile(outputFolder, 't_sim2.mat'),"t_sim2");
save(fullfile(outputFolder, 'T_test.mat'),"T_test");
save(fullfile(outputFolder, 'T_train.mat'),"T_train");
%%  反归一化
T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

%%  性能评价
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  绘制网络分析图
analyzeNetwork(lgraph);
%%  绘图
figure
scatter(1:M, T_train, 'b', 'O');  % 绘制训练集真实值散点图
hold on
scatter(1:M, T_sim1, 'r', '*');        % 绘制训练集预测值散点图
hold off
legend('True Values', 'Predicted Values')
xlabel('Sample')
ylabel('Prediction Results')
string = {'Comparison of Training Set Predictions'; ['Accuracy=' num2str(error1) '%']};
title(string)
grid
saveas(gcf, fullfile(outputFolder, 'Train_Prediction_Comparison.fig'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'Train_Prediction_Comparison.png'))
% print(gcf, '-dpng', '-r300', fullfile(outputFolder, 'Train_Prediction_Comparison.jpg'))


figure
scatter(1:N, T_test, 'b', 'O');  % 绘制测试集真实值散点图
hold on
scatter(1:N, T_sim2, 'r', '*');        % 绘制测试集预测值散点图
hold off
legend('True Values', 'Predicted Values')
xlabel('Sample')
ylabel('Prediction Results')
string = {'Comparison of Test Set Predictions'; ['Accuracy=' num2str(error2) '%']};
title(string)
grid
saveas(gcf, fullfile(outputFolder, 'Test_Prediction_Comparison.fig'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'Test_Prediction_Comparison.png'))
% print(gcf, '-dpng', '-r300', fullfile(outputFolder, 'Test_Prediction_Comparison.jpg'))


%%  混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
saveas(gcf, fullfile(outputFolder, 'Confusion_Matrix_Train.fig'));
% saveas(gcf, fullfile(outputFolder, 'Confusion_Matrix_Train.jpg'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'Confusion_Matrix_Train.png'))
% print(gcf, '-dpng', '-r300', fullfile(outputFolder, 'Confusion_Matrix_Train.jpg'))
close;
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
saveas(gcf, fullfile(outputFolder, 'Confusion_Matrix_Test.fig'));
% saveas(gcf, fullfile(outputFolder, 'Confusion_Matrix_Train.jpg'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'Confusion_Matrix_Test.png'))
% print(gcf, '-dpng', '-r300', fullfile(outputFolder, 'Confusion_Matrix_Test.jpg'))
close;

%% 计算AUC和绘制AOC曲线
% class_names = {'Cytopl', 'ER', 'Gol', 'Mito', 'Nucl', 'Vesi'};
class_names = {'Cytopl', 'ER', 'Gol'};
% 绘制 ROC 曲线和计算 AUC 值
roc_data = cell(1, 3);
auc_values = zeros(1, 3);
for i = 1:3
    % 获取当前类别的预测概率和真实标签
    scores = t_sim2(:, i);
    labels = (T_test == i); % 将当前类别标签设为正类，其他类别为负类
    [X, Y, ~, AUC] = perfcurve(labels, scores, 1);
    roc_data{i} = [X, Y];
    auc_values(i) = AUC;
end

figure;
hold on;
for i = 1:3
    plot(roc_data{i}(:, 1), roc_data{i}(:, 2), 'LineWidth', 1.5);
end
plot([0, 1], [0, 1], '-', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 0.5);
% xlabel('False Positive Rate');
% ylabel('True Positive Rate');
% title('ROC Curve (Multiclass)');
xlabel('{\fontname{Arial} False Positive Rate}','FontSize',13,'Interpreter','tex');
ylabel('{\fontname{Arial} True Positive Rate}','FontSize',13,'Interpreter','tex');
title('{\fontname{Arial}\bfseries ROC Curve (Multiclass)}','FontSize',13,'Interpreter','tex');


% 更改字体
set(gca, 'FontName', 'Arial'); 
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Arial');

legend_strings = cell(1, 3);
for i = 1:3
    fprintf('%s (AUC=%.4f)\n', class_names{i}, auc_values(i));
    legend_strings{i} = sprintf('%s (AUC=%.4f)', class_names{i}, auc_values(i));
end

h_legend = legend(legend_strings, 'Location', 'southeast');
set(h_legend, 'FontName', 'Arial');
grid off;
hold off;
mean_auc = mean(auc_values);
% 显示平均 AUC 值
% fprintf('Mean AUC Value: %.4f\n', mean_auc);

saveas(gcf, fullfile(outputFolder, 'ROC_Curve_(Multiclass).fig'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'ROC_Curve_(Multiclass).png'))

%% 根据混淆矩阵计算准确率、召回率和 F1 值
C = confusionmat(T_test, T_sim2);

accuracy = sum(diag(C)) / sum(C(:));
precision  = diag(C) ./ sum(C,1)';
precision(isnan(precision)) = 0;
recall= diag(C) ./ sum(C,2);
recall(isnan(recall)) = 0;
f1 = 2 * precision .* recall ./ (precision + recall);
f1(isnan(f1)) = 0;

macro_precision = mean(precision);
macro_recall = mean(recall);
macro_f1 = mean(f1);

num_classes = size(C, 1);
macro_accuracy = 0;
for i = 1:num_classes
    true_positive = C(i, i); % 当前类别的真正例数
    false_positive = sum(C(:, i)) - true_positive; % 当前类别的假正例数
    true_negative = sum(C(:)) - sum(C(i, :)) - sum(C(:, i)) + true_positive; % 其他类别的真负例数
    false_negative = sum(C(i, :)) - true_positive; % 其他类别的假负例数
    binary_accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative);
    macro_accuracy = macro_accuracy + binary_accuracy;
end
macro_accuracy = macro_accuracy / num_classes;

% 输出混淆矩阵和性能指标
disp('Confusion matrix:');
disp(C);
disp('Accuracy:');
disp(accuracy);
disp('Precision:');
disp(precision);
disp('macro_precision:');
disp(macro_precision);
disp('Recall:');
disp(recall);
disp('macro_recall:');
disp(macro_recall);
disp('F1 score:');
disp(f1);
disp('macro_f1 score:');
disp(macro_f1);
disp('Macro Accuracy:');
disp(macro_accuracy);
disp('Macro AUC:');
disp(mean_auc);

filePath = fullfile(outputFolder, 'Evaluation_indicators.txt');
fileID = fopen(filePath, 'w');
fprintf(fileID, 'Confusion matrix:\n');
for i = 1:size(C, 1)
    for j = 1:size(C, 2)
        fprintf(fileID, '%d\t', C(i, j));
    end
    fprintf(fileID, '\n');
end
fprintf(fileID, 'Accuracy: %.6f\n', accuracy);
fprintf(fileID, 'Precision: %.6f\n', precision);
fprintf(fileID, 'Recall: %.6f\n', recall);
fprintf(fileID, 'F1 score: %.6f\n', f1);
fprintf(fileID, 'Macro Precision: %.6f\n', macro_precision);
fprintf(fileID, 'Macro Recall: %.6f\n', macro_recall);
fprintf(fileID, 'Macro F1 score: %.6f\n', macro_f1);
fprintf(fileID, 'Macro Accuracy: %.6f\n', macro_accuracy);
fprintf(fileID, 'Macro AUC: %.6f\n', mean_auc);
fclose(fileID);

%%  保存网络模型
save(fullfile(outputFolder, 'net.mat'),"net");
save(fullfile(outputFolder, 'C.mat'),"C");
save(fullfile(outputFolder, 'options.mat'),"options");
save(fullfile(outputFolder, 'data.mat'),"res");










