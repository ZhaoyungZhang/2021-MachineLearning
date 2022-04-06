%% Hand
clc,clear;
% 存一个map 这1000个sampledata里面 对应于原来的第几个
[Train_Data_X,Train_Labels] = strimage();
[Train_Data_X,Train_Labels,Map_id] = sample_random(1000,Train_Data_X,Train_Labels);

%Size = 12665 * 59
%% 
Cs = 0.1:0.1:2;
Cs = [0.01 0.02 0.04 0.06 0.08 Cs];
for Cnum=1:25
C = Cs(Cnum); % 超参
H=[];%目标函数的H
for i=1:length(Train_Data_X)%对于所有样本都要遍历
    for j=1:length(Train_Data_X)
        H(i,j)=Train_Data_X(i,:)*(Train_Data_X(j,:))'*Train_Labels(i)*Train_Labels(j);
    end
end
F = -1 * ones(length(Train_Data_X),1);%目标函数的F
%% 
%等式约束
aeq=Train_Labels';
beq=zeros(1,1);
%不等式约束 没有
ub=[];
ib=[];
%自变量约束
lb = zeros(length(Train_Data_X),1);%下界
ub = [];%没有上界要求
ub = zeros(length(Train_Data_X),1);%上界
ub(:,:) = C;
% 求解alpha
[alpha,fval]=quadprog(H,F,[],[],aeq,beq,lb,ub);%二次规划问题

%%
% plot(alpha,'b') %明显
% title('\alpha of QP')
% xlabel('x')
% ylabel('\alpha')
a = alpha;
% 还是选择处理一下 因为正常来说是有=0的
for i=1:length(a)
    if a(i)<1e-8
        a(i)=0;
    end
end

%% 求W
W = 0; % 系数矩阵
u = 0;
j = find(a > 0);
for i = 1:length(Train_Data_X)
        W = W + a(i)*Train_Labels(i)*Train_Data_X(i,:)';
end
%% 
j = find(a > 0); %找 S V
nums = length(j); % 多少个
temp = Train_Labels - Train_Data_X*W;
B = sum(temp(j)) / nums;
%% 
Predict = [];
test_X = Train_Data_X;
test_Y = Train_Labels;
Predict = sign(test_X*W + B);
Judge=(Predict==test_Y);
train_score(Cnum) = sum(Judge)./length(test_X)
%% 找误分类的
% misclass =[];
% misclass = find(Judge==0)
% for i=1:length(misclass)
%     id = misclass(i);
%     ori_id = Map_id(id);
%     showimage(ori_id)
% end
%% Test
[Test_Data_X,Test_Labels] = strimagetest();
Predict = [];
test_X = Test_Data_X;
test_Y = Test_Labels;
Predict = sign(test_X*W + B);
Judge=(Predict==test_Y);
score(Cnum) = sum(Judge)./length(test_X)
%% 找误分类的
% misclass =[];
% misclass = find(Judge==0)
% for i=1:length(misclass)
%     id = misclass(i);
%     ori_id = Map_id(id);
%     showimage(ori_id)
% end
end
figure
plot(Cs,train_score,'LineWidth',2)
title('Trainning Performance in Different C')
xlabel('C')
ylabel('Accuracy')
ylim([0.5 1])
axis()

figure
plot(Cs,score,'LineWidth',2)
title('Testing Performance in Different C')
xlabel('C')
ylabel('Accuracy')
ylim([0.5 1])
axis()
%%
function [sample_Datas,sample_labels,chose_set] = sample_random(num,datas,labels)
    % datas 为原始数据 num 为目标数目
    chose_set = [];%1000*1 即这1000个数在原来set中的id
    R = length(datas);
    a = rand(R,1);
    [b,c] = sort(a);
    chose_set = c(1:num,1);
    chose_set = sort(chose_set);
    sample_Datas = datas(chose_set,:);
    sample_labels = labels(chose_set,:);
end
