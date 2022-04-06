%% ================PCA降维================
function [newtrainData,newtestData,K] = MyPCA(Datas,rate,train_percent,class_num)
fprintf(' =======Do PCA Dimensionality Reduction...=======\n');
% Step 1: 中心化
    mean_val = mean(Datas);
    newData1 = Datas - mean_val;
% Step 2: 求协方差矩阵
    C = newData1'*newData1 / size(newData1,1);
% Step 3: 计算特征向量和特征值
    %[V,D] = eig(A) 返回特征值的对角矩阵 D 和矩阵 V
    %其列是对应的右特征向量，使得 A*V = V*D。
    [V,D] = eig(C);
% Step 4: 排序
    [val,order] = sort(diag(-D));% 降序
    eigenvalue = diag(D); % 得到特征值
    eigenvector = V(:,order); % 特征向量
    eigenvalue = eigenvalue(order); % 特征值也重新排序
% Step 5: 取前K行，组成矩阵U 然后计算
    % 原特征相当于有112*92=10304个特征
    % 可以选K=[1000 3000 5000 6000 7000 8000 9000 10000]
    % 此处 使用rate比例 来动态定义K
    K = 0;
    sumeigenval = sum(eigenvalue);
    for i = 1:length(eigenvalue)
        rating = sum(eigenvalue(1:i,1),1)/sumeigenval;
        if rating >= rate
            K = i;
            break;
        end
    end
    fprintf(' ================Rate = %f================\n',rate);
    fprintf(' =============This Turn K = %d.==============\n',K);
    U = eigenvector(:,1:K);
    % M*K = M*N * N*K
    newData = newData1 * U;
    newtrainData = newData(1:train_percent*class_num,:);
    newtestData = newData(train_percent*class_num+1:400,:);
    fprintf(' ==========Dimension From 10304 Lower To %d==========\n',K);
    fprintf(' =============== Done PCA ===============\n');
end