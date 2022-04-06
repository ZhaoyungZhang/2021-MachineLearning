%% Non-Linear SVM
clear,clc
Trainning_Datas = load('training_3.txt');
Data_x1 = Trainning_Datas(:,1);
Data_x2 = Trainning_Datas(:,2);
Data_label = Trainning_Datas(:,3);
X = [Data_x1,Data_x2];
Y = Data_label;
group1 = find(Data_label==1);
group2 = find(Data_label==-1);
scatter(Data_x1(group1),Data_x2(group1),'.','r');
hold on
scatter(Data_x1(group2),Data_x2(group2),'*','b');
hold on
%% 
gammas = [1 10 100 1000];
for gnum = 1:4
gamma = gammas(gnum);
%求Kernel 矩阵
Knel = get_kernel_matrix(X,gamma);
H=[];
for i=1:length(X)
    for j = 1:length(X)
        H(i,j) = Knel(i,j)*Y(i)*Y(j);
    end
end
F=-1*ones(length(X),1);%目标函数的F
aeq = Y';
beq=zeros(1,1);
ub=[];
ib=[];
%自变量约束
lb=zeros(length(X),1);
ub=[];
[alpha,fval]=quadprog(H,F,ib,ub,aeq,beq,lb,ub);
%%
a = alpha;
epsilon = 1e-5;
%查找支持向量
sv_index = find(abs(a)> epsilon);
Xsv = X(sv_index,:);
Ysv = Y(sv_index);
svnum = length(sv_index);
sum_b = 0;
for k = 1:svnum
    sum = 0;
    for i = 1:length(X)
        sum = sum + a(i,1)*Y(i,1)*Knel(i,k);
    end 
    sum_b = sum_b + Ysv(k) - sum;
end
B = sum_b/svnum;
% W不用直接求解

%% Make Classfication Predictions over a grid of values
xplot = linspace (min(X( : , 1 ) ) , max(X( : , 1 ) ),100 )';
yplot = linspace (min(X( : , 2 ) ) , max(X( : , 2 ) ) ,100)';
[XX, YY] = meshgrid(xplot,yplot);
vals = zeros(size(XX));
%% 计算vals using SVM
for i=1:length(vals)
    for j = 1:length(vals)
        test_X = [XX(i,j),YY(i,j)];
        temp = 0;
        for k=1:length(X)
            temp = temp + a(k,1) * Y(k,1)*exp(-gamma*norm(X(k,:)-test_X)^2);
        end
        vals(i,j) = temp + B;
    end
end
%%
figure
scatter(Data_x1(group1),Data_x2(group1),'.','r');
hold on
scatter(Data_x1(group2),Data_x2(group2),'*','b');
hold on
colormap bone;
contour(XX,YY,vals,[0 0],'LineWidth',2)
title(['\gamma = ',num2str(gamma)]);
end
%%
function K = get_kernel_matrix(data,gamma)
    K = [];
    for i=1:length(data)
        for j=1:length(data)
            K(i,j)=exp(-gamma*norm(data(i,:)-data(j,:))^2);
        end
    end
end

