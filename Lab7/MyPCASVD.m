%% ================PCA2降维================
function [newtrainData,newtestData,K] = MyPCASVD(Datas,K,train_percent,class_num)
fprintf(' =======Do PCA Dimensionality Reduction(Based on SVD)...=======\n');
Datas = Datas';
t0 = tic;
[NN,Train_NUM]=size(Datas);
% Step 1: 中心化
    mean_val = mean(Datas,2);
    newData1 = Datas - mean_val*ones(1,Train_NUM);
% Step 2: 求协方差矩阵  ！这里改了！
    C = newData1'*newData1 / Train_NUM;
% Step 3: 计算特征向量和特征值
    %[V,D] = eig(A) 返回特征值的对角矩阵 D 和矩阵 V
    %其列是对应的右特征向量，使得 A*V = V*D。
    %[V,D] = eig(C);
    
    % 使用SVD分解
    %[V,D,U] = svd(C,'econ');
    [V,S]=Find_K_Max_Eigen(C,K);
    %Vector: 400*K value:1*K
    % 左奇异向量 以及 奇异值 M*K
    disc_value=S;% 
    disc_set=zeros(NN,K);
    
    qiyizhi = ones(K,K);
    for i = 1:K
        qiyizhi(i,i) = disc_value(i);
    end
    
    newData1=newData1/sqrt(Train_NUM-1);
    
    %disc_set = 10000 * 8  data 10000 * 400
    for k=1:K
        % 映射回feature N*M   *  M*K  ->  N * K
        disc_set(:,k)=(1/sqrt(disc_value(k)))*newData1*V(:,k);
    end
    
% Step 5: 取前K行，组成矩阵U 然后计算
    % 原特征相当于有112*92=10304个特征
    fprintf(' =============This Turn K = %d.==============\n',K);
    newData = newData1'*disc_set;
    newtrainData = newData(1:train_percent*class_num,:);
    newtestData = newData(train_percent*class_num+1:400,:);
    fprintf(' ==========Dimension From 10304 Lower To %d==========\n',K);
    fprintf(' =============== Done PCA ===============\n');
    toc(t0);
end