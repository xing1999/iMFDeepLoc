% function W = getBasis(datadir)
% % % W = GETBASIS( IDX)
% % inputs:   IDX is the antibody id for a particular protein (default 1873)
% %IDX是抗体蛋白id
% % outputs:  W is the color basis matrix
% % W是颜色基础矩阵
% % Saves W to ./lib/2_separationCode.
% 
% if ~exist('datadir','var')
% 	datadir = './Multilabel2';
% end
% 
% tissuedir = [datadir];
% 
% if ~exist( tissuedir,'dir')
%     error('no antibody id');
% end
% 
% d2 = dir(tissuedir);
% d2(1:2) = [];
% 
% count = 1;
% Wbasis = {};
% for j=1:length(d2)
%     disp(j);  % zz
%     imgdir = [tissuedir '/' d2(j).name];
% 
%     d3 = dir(imgdir);
%     d3(1:2) = [];
% 
%     for k=1:length(d3)
%         infile = [imgdir '/' d3(k).name];
%         I = imread( infile);
%         counter = 1;
%         eval('W = colorbasis(I);','counter = 0;');
%         if counter
%             Wbasis{count} = W;
%             count = count + 1;
%         end
%     end
% end
% 
% 
% W = zeros(size(Wbasis{1}));
% for i=1:length(Wbasis)
%     W = W + Wbasis{i};
% end
% W=W';
% 
% return
% 
% 
% 
% function W = colorbasis( I, STOPCONN, A_THR, S_THR)
% rank = 2;  ITER = 5000;
% tic;
% if ~exist('STOPCONN','var')
%     STOPCONN = 40;
% end
% if ~exist('A_THR','var')
%     A_THR = 1000;
% end
% if ~exist('S_THR','var')
%     S_THR = 1000;
% end
% 
% I = 255 - I;
% IMGSIZE = size(I);
% 
% % ....tissue size check!
% if (IMGSIZE(1)<S_THR) || (IMGSIZE(2)<S_THR)
% 	error('Not enough useable tissue staining');
% end
% 
% 
% % ********** SEED COLORS ************
% S = size(I);
% V = reshape( I, S(1)*S(2),S(3));
% [V, ind, VIDX] = unique(V,'rows');
% VIDX = single(VIDX);
% 
% HSV = rgb2hsv( V);
% hue = HSV(:,1);
% [c, b] = hist( hue(hue<0.3), [0:0.01:1]);
% [A, i] = max(c);
% P = b(i);
% hae = mean(V(P-.01<hue & hue<P+0.01,:),1)';
% 
% [c, b] = hist( hue(hue>=0.3), [0:0.01:1]);
% [A, i] = max(c);
% P = b(i);
% dab = mean(V(P-.01<hue & hue<P+0.01,:),1)';
% 
% W = single( [hae dab] / 255);
% 

function W = getBasis(datadir)
% % W = GETBASIS( IDX)
% inputs:   IDX is the antibody id for a particular protein (default 1873)
% IDX是抗体蛋白id
% outputs:  W is the color basis matrix
% W是颜色基础矩阵
% Saves W to ./lib/2_separationCode.

if ~exist('datadir','var')
    datadir = './Multilabel2';
end

tissuedir = [datadir];

if ~exist(tissuedir, 'dir')
    error('no antibody id');
end

d2 = dir(tissuedir);
d2(1:2) = [];

Wbasis = {};
for j = 1:length(d2)
    disp(j);  % 输出当前文件夹索引以进行调试
    imgdir = fullfile(tissuedir, d2(j).name);

    d3 = dir(imgdir);
    d3(1:2) = [];

    for k = 1:length(d3)
        infile = fullfile(imgdir, d3(k).name);
        I = imread(infile);
        W = colorbasis(I);
        if ~isempty(W)  % 检查颜色基础矩阵是否为空
            Wbasis{end+1} = W;
        end
    end
end

if isempty(Wbasis)
    error('No valid data found.');
end

W = zeros(size(Wbasis{1}));
for i = 1:length(Wbasis)
    W = W + Wbasis{i};
end
W = W';

return

function W = colorbasis(I, STOPCONN, A_THR, S_THR)
rank = 2;
ITER = 5000;
tic;

if nargin < 4
    S_THR = 1000;
end

I = 255 - I;
I = double(I) / 255;
IMGSIZE = size(I);

% 检查组织大小
if (IMGSIZE(1) < S_THR) || (IMGSIZE(2) < S_THR)
    error('Not enough useable tissue staining');
end

% 提取颜色基础矩阵
S = size(I);
V = reshape(I, S(1)*S(2), S(3));
[V, ~, VIDX] = unique(V, 'rows');
VIDX = single(VIDX);

HSV = rgb2hsv(V);
hue = HSV(:, 1);

[c, b] = hist(hue(hue < 0.3), [0:0.01:1]);
[~, i] = max(c);
P = b(i);
hae = mean(V(P-.01<hue & hue<P+0.01, :), 1)';

[c, b] = hist(hue(hue >= 0.3), [0:0.01:1]);
[~, i] = max(c);
P = b(i);
dab = mean(V(P-.01<hue & hue<P+0.01, :), 1)';

W = single([hae dab] / 255);
