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
data = '(ALL=tradition+2+11+12+1+13)_Feature.xlsx';
res = xlsread(data);
res2=res(:, end);
size1=size(res,1);

%%  划分训练集和测试集
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
num_class =length(unique(T_test));  % 类别数（Excel最后一列放类别）
num_dim = size(res, 2) - 1;               % 特征维度
num_res = size(res, 1);                   % 样本数（每一行，是一个样本）


train_data = [P_train', T_train']; % 将P_train和T_train拼接成一个矩阵
train_data_filename = fullfile(outputFolder, 'train_data.xlsx');
% xlswrite(train_data_filename, train_data, 1, 'A1');
test_data = [P_test', T_test']; % 将P_train和T_train拼接成一个矩阵
test_data_filename = fullfile(outputFolder, 'test_data.xlsx');
% xlswrite(test_data_filename, test_data, 1, 'A1');

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test  = mapminmax('apply', P_test, ps_input);
t_train = ind2vec(T_train);
t_test  = ind2vec(T_test );

save(fullfile(outputFolder, 'ps_input.mat'),"ps_input");
%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  创建模型
k = 6;      % 保留主成分个数
[Xloadings, Yloadings, Xscores, Yscores, betaPLS, PLSPctVar, MSE, stats] = plsregress(p_train, t_train, k);
% https://blog.csdn.net/linping_/article/details/110193946

%%  预测拟合
t_sim1 = [ones(M, 1), p_train] * betaPLS;
t_sim2 = [ones(N, 1), p_test ] * betaPLS;

%%  数据反归一化
T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

%%  数据排序
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  性能评价
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

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
fclose(fileID);

%%  保存网络模型
save(fullfile(outputFolder, 'C.mat'),"C");
save(fullfile(outputFolder, 'data.mat'),"res");

