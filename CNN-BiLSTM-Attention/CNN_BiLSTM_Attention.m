%%  ��ջ�����������ӱ���
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ��������� 
currentFolder = pwd;
addpath(genpath(currentFolder))

%%  ����ʱ�������ļ���
currentDateTime = datestr(now, '()zzyyyymmdd_HHMMSS');
outputFolder = fullfile(pwd, currentDateTime);
mkdir(outputFolder);

%% ��������
data = 'FEC-Net_ALL_data.xlsx';
res = xlsread(data);
res2=res(:, end);
size1=size(res,1);

%% 7:3�������ݼ�Ϊѵ�����Ͳ��Լ�
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

train_data = [P_train', T_train']; % ��P_train��T_trainƴ�ӳ�һ������
train_data_filename = fullfile(outputFolder, 'train_data.xlsx');
% xlswrite(train_data_filename, train_data, 1, 'A1');
save(fullfile(outputFolder, 'train_data.mat'),"train_data");

test_data = [P_test', T_test']; % ��P_train��T_trainƴ�ӳ�һ������
test_data_filename = fullfile(outputFolder, 'test_data.xlsx');
% xlswrite(test_data_filename, test_data, 1, 'A1');
save(fullfile(outputFolder, 'test_data.mat'),"test_data");

%%  ���ݹ�һ��
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

save(fullfile(outputFolder, 'ps_input.mat'),"ps_input");
%%  ����ƽ��
%   ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
%   Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
%   ����Ӧ��ʼ�պ���������ݽṹ����һ��
P_train =  double(reshape(P_train, num_dim, 1, 1, M));
P_test  =  double(reshape(P_test , num_dim, 1, 1, N));

%%  ���ݸ�ʽת��
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end
    
%%  ����ģ��
lgraph = layerGraph();                                  % �����հ�����ṹ

tempLayers = [
    sequenceInputLayer([num_dim, 1, 1], "Name", "sequence")  % ��������㣬�������ݽṹΪ[num_dim, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                % ���������۵���
lgraph = addLayers(lgraph, tempLayers);                      % ����������ṹ����հ׽ṹ��

tempLayers = [
    convolution2dLayer([2, 1], 16, "Name", "conv_1")    % ���������, ����˴�С[2, 1], 32������ͼ
    dropoutLayer(0.2,"Name","dropout_1")  %zz
    batchNormalizationLayer("Name","batchnorm_1")  %zz
    reluLayer("Name", "relu_1")                         % Relu �����
    convolution2dLayer([2, 1], 32, "Name", "conv_2")];  % ���������, ����˴�С[2, 1], 64������ͼ
lgraph = addLayers(lgraph, tempLayers);                 % ����������ṹ����հ׽ṹ��

tempLayers = TransposeLayer("tans_1");                  % ά�Ƚ�����, �Ӷ��ڿռ�ά�Ƚ���GAP, ������ͨ��ά��
lgraph = addLayers(lgraph, tempLayers);                 % ����������ṹ����հ׽ṹ��

tempLayers = [globalAveragePooling2dLayer("Name", "gapool")];             % ȫ��ƽ���ػ���
lgraph = addLayers(lgraph, tempLayers);                      % ����������ṹ����հ׽ṹ��

tempLayers = [globalMaxPooling2dLayer("Name", "gmpool")];     % ȫ�����ػ���
lgraph = addLayers(lgraph, tempLayers);                      % ����������ṹ����հ׽ṹ��

tempLayers = [
    concatenationLayer(1, 2, "Name", "concat")                         % ƴ�� GAP �� GMP ��Ľ��
    TransposeLayer("tans_2")                                           % ά�Ƚ�����, �ָ�ԭʼά��
    convolution2dLayer([1, 1], 1, "Name", "conv_3", "Padding", "same") % ���������, ͨ����Ŀ�任
    sigmoidLayer("Name", "sigmoid")];                                  % sigmoid �����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % ��˵Ŀռ�ע����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % �������з��۵���
    flattenLayer("Name", "flatten")                                    % ������ƽ��
    bilstmLayer(13, "Name", "BiLSTM", "OutputMode", "last")            % BiLSTM��
    dropoutLayer(0.5,"Name","dropout_5")           %zz
    batchNormalizationLayer("Name","batchnorm_5")  %zz
    fullyConnectedLayer(num_class, "Name", "fc")                       % ȫ���Ӳ�
    softmaxLayer("Name", "softmax")                                    % softmax�����
    classificationLayer("Name", "classification")];                    % �����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % �۵������ ���� ���������
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize");
                                                                       % �۵������ ���� ���۵�������
lgraph = connectLayers(lgraph, "conv_2", "tans_1");                    % �������� ���� ά�ȱ任��
lgraph = connectLayers(lgraph, "conv_2", "multiplication/in2");        % �������� ���� ��˲�(ע����)����2
lgraph = connectLayers(lgraph, "tans_1", "gapool");                    % ά�ȱ任�� ���� GAP
lgraph = connectLayers(lgraph, "tans_1", "gmpool");                    % ά�ȱ任�� ���� GMP
lgraph = connectLayers(lgraph, "gapool", "concat/in2");                % GAP ���� ƴ�Ӳ�1
lgraph = connectLayers(lgraph, "gmpool", "concat/in1");                % GMP ���� ƴ�Ӳ�2
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");        % sigmoid ���� ��˲�
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");       % ������

%%  ��������
options = trainingOptions('adam', ...      % Adam �ݶ��½��㷨
    'MaxEpochs', 10, ...                  % ����������
    'InitialLearnRate', 0.004, ...           % ��ʼѧϰ��
    'L2Regularization', 0.6113e-02, ...         % L2���򻯲���
    'LearnRateSchedule', 'piecewise', ...  % ѧϰ���½�
    'LearnRateDropFactor', 0.003, ...        % ѧϰ���½����� 0.1
    'LearnRateDropPeriod', 450, ...        % ����450��ѵ���� ѧϰ��Ϊ 0.001 * 0.5
    'Shuffle', 'every-epoch', ...          % ÿ��ѵ���������ݼ�
    'ValidationPatience', 200, ...          % �ر���֤
    'Plots', 'training-progress', ...      % ��������
    'Verbose', false, ...                  % ����ʾ��ϸ��Ϣ
    'GradientThreshold',inf, ...
    'ValidationData',{p_test,t_test}, ...
    'ExecutionEnvironment', 'auto');        % ִ�л�����Ϊ GPU

%%  ѵ��ģ�� (������ѵ�������е�����)
% net = trainNetwork(p_train, t_train, lgraph, options);
[net,info] = trainNetwork(p_train, t_train, lgraph, options);
save(fullfile(outputFolder, 'info.mat'),"info");
auto_loss = info.TrainingLoss';
auto_accuracy = info.TrainingAccuracy';
test_loss = info.ValidationLoss';
test_accuracy = info.ValidationAccuracy';

iterations = (1:numel(auto_loss))';
data = table(iterations, auto_accuracy, auto_loss, 'VariableNames', {'��������', '׼ȷ��', '��ʧ'});
filename = fullfile(outputFolder, 'train_process_data.xlsx');
writetable(data, filename);


figure(1)
plot(info.TrainingLoss);%%����ѵ����loss
hold on
saveas(gcf, fullfile(outputFolder, 'loss.png'));
figure(2)
plot(info.TrainingAccuracy);%%����ѵ����׼ȷ��
saveas(gcf, fullfile(outputFolder, 'Accuracy.png'));
%%  Ԥ��ģ��
t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test ); 

save(fullfile(outputFolder, 't_sim1.mat'),"t_sim1");
save(fullfile(outputFolder, 't_sim2.mat'),"t_sim2");
save(fullfile(outputFolder, 'T_test.mat'),"T_test");
save(fullfile(outputFolder, 'T_train.mat'),"T_train");
%%  ����һ��
T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

%%  ��������
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  �����������ͼ
analyzeNetwork(lgraph);
%%  ��ͼ
figure
scatter(1:M, T_train, 'b', 'O');  % ����ѵ������ʵֵɢ��ͼ
hold on
scatter(1:M, T_sim1, 'r', '*');        % ����ѵ����Ԥ��ֵɢ��ͼ
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
scatter(1:N, T_test, 'b', 'O');  % ���Ʋ��Լ���ʵֵɢ��ͼ
hold on
scatter(1:N, T_sim2, 'r', '*');        % ���Ʋ��Լ�Ԥ��ֵɢ��ͼ
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


%%  ��������
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

%% ����AUC�ͻ���AOC����
% class_names = {'Cytopl', 'ER', 'Gol', 'Mito', 'Nucl', 'Vesi'};
class_names = {'Cytopl', 'ER', 'Gol'};
% ���� ROC ���ߺͼ��� AUC ֵ
roc_data = cell(1, 3);
auc_values = zeros(1, 3);
for i = 1:3
    % ��ȡ��ǰ����Ԥ����ʺ���ʵ��ǩ
    scores = t_sim2(:, i);
    labels = (T_test == i); % ����ǰ����ǩ��Ϊ���࣬�������Ϊ����
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


% ��������
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
% ��ʾƽ�� AUC ֵ
% fprintf('Mean AUC Value: %.4f\n', mean_auc);

saveas(gcf, fullfile(outputFolder, 'ROC_Curve_(Multiclass).fig'));
print(gcf, '-dpng', '-r600', fullfile(outputFolder, 'ROC_Curve_(Multiclass).png'))

%% ���ݻ����������׼ȷ�ʡ��ٻ��ʺ� F1 ֵ
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
    true_positive = C(i, i); % ��ǰ������������
    false_positive = sum(C(:, i)) - true_positive; % ��ǰ���ļ�������
    true_negative = sum(C(:)) - sum(C(i, :)) - sum(C(:, i)) + true_positive; % ���������渺����
    false_negative = sum(C(i, :)) - true_positive; % �������ļٸ�����
    binary_accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative);
    macro_accuracy = macro_accuracy + binary_accuracy;
end
macro_accuracy = macro_accuracy / num_classes;

% ����������������ָ��
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

%%  ��������ģ��
save(fullfile(outputFolder, 'net.mat'),"net");
save(fullfile(outputFolder, 'C.mat'),"C");
save(fullfile(outputFolder, 'options.mat'),"options");
save(fullfile(outputFolder, 'data.mat'),"res");










