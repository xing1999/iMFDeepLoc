function [W,H] = findWH( V, inputs)
% OUTPUT = FINDWH(V,OPTIONS)
% inputs:   V is the sample matrix (nsamples x 3 original channels, R G B)
%           OPTIONS is a structure containins the following attributes:
%                  .INIT: the color-bases matrix for linear unmixing
%                  ('none','truncated','hue')
%输入：V是样本矩阵（nsamples x 3个原始通道，R G B.）
%OPTIONS是一个包含以下属性的结构：
%.INIT：用于线性分离的色基矩阵（‘none’，‘’，‘hue’）
%
% outputs:  OUTPUT is a structure containing the processed data
%           OUTPUT.W is color bases matrix
%           OUTPUT.Hl is the linearly unmixed weightings
%           OUTPUT.Hn is the NMF unmixed weightings
%输出：OUTPUT是包含已处理数据的结构
%输出.W是颜色基础矩阵
%输出.Hl是线性未混合权重

S = size(V);

options.ITER = 5000;       % Number of nmf iterations
options.INIT = 'hue';      % Pseudo-random W initialization
options.RANK = 2;          % Number of stains to be unmixed. FIX FOR LIN UNMIX
options.STOPCONN = 40;     % Another termination criteria for nmf
options.VERBOSE = 1;       % Verbose mode
options.RSEED = 13;

rand('seed',options.RSEED);

checker.options = fieldnames(options);
if exist('inputs','var')
    checker.inputs = fieldnames(inputs);
    for i=1:length(checker.inputs)
        checker.type = checker.inputs{i};
        checker.value = getfield( inputs,checker.inputs{i});
        options = setfield( options, checker.type, checker.value);
    end
end

if options.VERBOSE
    tic;
end

if strcmp( options.INIT,'hue')
    HSV = rgb2hsv( V); % rgb to hsv HSV  19511*3 double
    hue = smooth(HSV(:,1),30); % 將高频成分剔除
    [c, b] = hist( hue(hue<0.3), [0:0.01:1]);
    %返回c和b，包含有数目频率和间隔位置b的间隔为0.01
    [A, i] = max(c);
    P = b(i);%DNA
    hae = mean(V(P-.01<hue & hue<P+0.01,:),1)';%返回包含每一列中元素均值的行向量。
    [c, b] = hist( hue(hue>=0.3), [0:0.01:1]);
    [A, i] = max(c);
    P = b(i);%protein
    dab = mean(V(P-.01<hue & hue<P+0.01,:),1)';
    W = single( [hae dab] / 255)';
else
    W = single( rand(S(2),options.RANK))';
end
if strcmp( options.INIT,'truncated')
    W = W / 2 + .25;
end

% Continuing image transform
V = single(V)/255;
V(V==0) = 1e-9;
[c,r] = size(V);

% Blind umixing
H = single( rand( options.RANK,c))';

conn = zeros(r,r);
connold = conn;
inc=0;

for k=1:options.ITER
    W = W / norm(W);

    WH = H*W;

    % Minimizing L2 distance
    Hn = H.*((V*W') ./ (WH*W'));
    W  = W.*((H'*V) ./ (H'*WH));

    % % Minimizing divergence
    % VWH = V./WH;
    % Hn = H.*(W'*VWH) ./ repmat( sum(W,1)',[1 c]);
    % W  = W.*(VWH*H') ./ repmat( sum(H,2)', [r 1]);

    H = Hn;

    % Below is the Broad connectivity criterion
    if mod( k,10)==0
        WW=W';
        [y,i] = max(WW,[],2);
        mat1 = repmat( i,1,r);
        mat2 = repmat( i',r,1);
        conn=(mat1==mat2);

        if sum(conn(:)~=connold(:))
            inc=0;
        else
            inc=inc+1;
        end
        connold=conn;

        if inc>options.STOPCONN
            break;
        end

        Err = sum((V(:)' - WH(:)').^2)/(r*c);
        if options.VERBOSE
            disp( num2str([toc k inc Err max(W(:)) max(H(:))]));
        end
    end
end

H=H-min(H(:));
H=H/max(H(:));
H=uint8(round(H*255));
W=uint8(round(W*255));
%提高计算的精度和范围
W=im2double(W);
%%%%%%%%%%%%%%%
hsv = rgb2hsv( W);

if options.RANK==2
    if hsv(1,1)>hsv(2,1)
        W = W([2 1],: );
        H = H(:,[2 1]);
    end
end


if options.VERBOSE
    t = toc;
    t = num2str(t);
    disp([ 'Finished NMF: ' t]);
    tic;
end

