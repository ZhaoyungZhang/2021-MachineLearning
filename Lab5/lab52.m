%% 实验五 SVM 支持向量机
%% 软间隔
%% 读数据 画散点图
clear,clc;
Trainning_Datas = load('training_2.txt');

Data_x1 = Trainning_Datas(:,1);
Data_x2 = Trainning_Datas(:,2);
Data_label = Trainning_Datas(:,3);
X = [Data_x1,Data_x2];
Y = Data_label;
%scatter(Data_x1,Data_x2,'.','r');

%% 利用quadprog 求解 $\alpha$
% 求Min 1/2 yax - a
%目标函数必须写处一个二次型H和一个矩阵相加的形式
C = 1; % 超参
H=[];%目标函数的H
for i=1:length(X)%对于所有样本都要遍历
    for j=1:length(X)
        H(i,j)=X(i,:)*(X(j,:))'*Y(i)*Y(j);
    end
end
F = -1 * ones(length(X),1);%目标函数的F
%等式约束
aeq=Y';
beq=zeros(1,1);
%不等式约束 没有
ub=[];
ib=[];
%自变量约束
lb = zeros(length(X),1);%下界
ub = zeros(length(X),1);%上界
ub(:,:) = C;
% 求解alpha
[alpha,fval]=quadprog(H,F,[],[],aeq,beq,lb,ub);%二次规划问题

%% 会求出很多alpha十分接近于0 那种不是support vector
% 但是 大概多少个点合适呢？？ 大概就是三个 因为那个分界线还是很明显的
% BUT 软间隔要全算 so 没必要分了
% 注意！！ data1 明显不需要软间隔 所有没有误分类！(除非C很小~)
% 就是正常大概是0.01 刚出头
plot(alpha,'b') %明显
title('\alpha of QP')
xlabel('x')
ylabel('\alpha')
a = alpha;
% 还是选择处理一下 因为正常来说是有=0的
for i=1:length(a)
    if a(i)<1e-8
        a(i)=0;
    end
end
%% 求w
W = 0; % 系数矩阵
u = 0;
j = find(a > 0 & a < C);
for i = 1:length(X)
        W = W + a(i)*Y(i)*X(i,:)';
end

%% 求r
R = 0;
R = C - a;

%% 求B 是用的support vector
% 有点问题... 就是 只用S V 还是全用
% 公式是S V 但是 算出来的SV 和 理想SV不一样
j = find(a > 0 & a < C); %找 S V
nums = length(j); % 多少个
temp = Y - X*W;
B = sum(temp(j)) / nums;

%% 画出Boundary
k = -W(1)./W(2);
bb = -B./W(2);
figure
group1 = find(Data_label==1);
group2 = find(Data_label==-1);
scatter(Data_x1(group1),Data_x2(group1),'.','r');
hold on
scatter(Data_x1(group2),Data_x2(group2),'*','b');
hold on


yy = k.*Data_x1 + bb;
plot(Data_x1,yy,'-','LineWidth',0.5)
hold on
yy = k.*Data_x1 + bb + 1./W(2);
plot(Data_x1,yy,'--','LineWidth',1)
hold on
yy = k.*Data_x1 + bb - 1./W(2);
plot(Data_x1,yy,'--','LineWidth',0.5)
title('Support Vector Machine C = 1')
xlabel('Dimension1')
ylabel('Dimension2')
legend('Class 1','Class 2','Separating Hyperplane')

%% Test
Predict = [];
Test_Datas = load('test_2.txt');
test_x1 = Test_Datas(:,1);
test_x2 = Test_Datas(:,2);
test_X = [test_x1,test_x2];
test_Y = Test_Datas(:,3);
Predict = sign(test_X*W + B);
Judge=(Predict==test_Y);
score = sum(Judge)./length(test_X)

%%  Draw Bar on testdata
figure
test_Class1_nums = length(find(test_Y==1));
test_Class2_nums = length(find(test_Y==-1));
Predict_Class1_nums = length(find(Predict==1));
Predict_Class2_nums = length(find(Predict==-1));
Bar_y = [test_Class1_nums,Predict_Class1_nums;test_Class2_nums,Predict_Class2_nums];
Bar_x = categorical({'Class One','Class Two'});
bar(Bar_x,Bar_y)
legend('Actual','Predict');
xlabel('Classes');
ylabel('Count');
title(['TestDataSet2 Performance']);