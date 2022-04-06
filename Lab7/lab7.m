%% ================读取数据================
clear,clc;
% 随机选择取5--7张作为训练集
train_percent = randi([5,7]);% 随机一个5-7
train_percent= 7;
% ===初始参数===
dimx = 112;dimy = 92;
class_num = 40;
% ===数据读取=== Dim : (class_num*train_percent,dimx*dimy)
[train_datas,test_datas] = LoadData(train_percent,dimx,dimy,class_num);
rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9];
%% 大概算了一下 K = 1 2 2 4 6 10 19 27 38 55 82
rates = [0.1,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9];
K = [1 2 4 6 10 19 27 38 55 82];

%% ================PCA降维================
Datas = [train_datas;test_datas];
misclasses11 = cell(10,1);
misclasses1all = cell(10,1);
SVMFLAGs=[1,0];
%for SVMFLAGi = 1:2
%% 
for i = 1:length(rates)
    [new_train_datas,new_test_datas,K(i)] = MyPCA(Datas,rates(i),train_percent,class_num);
    %% ================SVM多分类================
    %SVMFLAG = 0;% 0:1对多SVM 多分类 
    %SVMFLAG = 1;% 0:1对1SVM 多分类
    SVMFLAG = 0;
    if SVMFLAG == 1
        [misclasses{i},accuracy11(i)] = MySVM1(new_train_datas,new_test_datas,class_num,K(i),train_percent);
    else
        [misclasses1all{i},accuracy1all(i)] = MySVM0(new_train_datas,new_test_datas,class_num,K(i),train_percent);
    end
end
%end

%% 分析结果
figure
plot(rates,K,'.-','LineWidth',2,'color','r','MarkerSize',30)
xlabel('Rate Of Sum EigenValue')
ylabel('K')
title('不同特征值比例下的K值')
%% 准确率
figure
plot(K,accuracy11,'.-','LineWidth',2,'color','g','MarkerSize',30)
hold on
plot(K,accuracy1all,'.-','LineWidth',2,'color','b','MarkerSize',30)
xlabel('K')
ylabel('Accuracy')
title('不同K值下的准确率')
legend('One To One','One To All');
%% 找几张错误的图片
%%
ori = misclasses1all{11}{1}(:,1);
pre = misclasses1all{11}{1}(:,2);
difference = find(ori~=pre);
for i=1:length(difference)
    fprintf('误认为S%d 是 S%d\n',ori(difference(i)),pre(difference(i)));
    path = ['orl_faces\s'];
    personid= randi([1,10]);
    filename1 = [path,num2str(ori(difference(i))),'\',num2str(personid),'.pgm'];
    ori_img = double(imread(filename1));
    filename2 = [path,num2str(pre(difference(i))),'\',num2str(personid),'.pgm'];
    pre_img = double(imread(filename2));
    figure
    subplot(2,1,1);
    imshow(uint8(ori_img)),title('原人脸图像');
    subplot(2,1,2);
    imshow(uint8(pre_img)),title('被误认为的人脸')
end



