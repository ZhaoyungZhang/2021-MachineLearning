%% ================读取数据================
clear,clc;
% 随机选择取5--7张作为训练集
train_percent = randi([5,7]);% 随机一个5-7
train_percent= 7;
% ===========初始参数===============
dimx = 112;dimy = 92;
class_num = 40;
% ============数据读取============== Dim : (class_num*train_percent,dimx*dimy)
[train_datas,test_datas] = LoadData(train_percent,dimx,dimy,class_num);
%rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9];
%% 大概算了一下 K = 1 2 2 4 6 10 19 27 38 55 82
% 不用rate 直接用K 因为121 在K=10就100%了 所以这次选择从 1 : 15 15次
%K = [100,200,500,600,800,1000,1200,1500,1800,2000,2500,3000,4000,5000,7000];
 K = 1:1:15;
%% ================PCA降维================
Datas = [train_datas;test_datas];
misclasses11 = cell(15,1);
misclasses1all = cell(15,1);
SVMFLAGs=[1,0];
sumt = 0;
t1end = zeros(15,1);
t2end = zeros(15,1);
for SVMFLAGi = 1:2 
    for i = 1:15
        t0 = tic;
        %[new_train_datas,new_test_datas] = MyPCA2(Datas,K(i),train_percent,class_num);
        [new_train_datas,new_test_datas] = MyPCASVD(Datas,K(i),train_percent,class_num);
        sumt = sumt + toc(t0);
        %% ================SVM多分类================
        %SVMFLAG = 0;% 0:1对多SVM 多分类 
        %SVMFLAG = 1;% 0:1对1SVM 多分类
        SVMFLAG = SVMFLAGs(SVMFLAGi);
        if SVMFLAG == 1
            t1 = tic;
            [misclasses11{i},accuracy11(i)] = MySVM1(new_train_datas,new_test_datas,class_num,K(i),train_percent);
            t1end(i) = t1end(i) + toc(t1);
        else
            t2 = tic;
            [misclasses1all{i},accuracy1all(i)] = MySVM0(new_train_datas,new_test_datas,class_num,K(i),train_percent);
            t2end(i) = t2end(i) + toc(t2);
        end
    end
    
end
avg_timecost= sumt/30;
fprintf('使用特征值分解的PCA的平均时间花费为:%f s\n',avg_timecost);

%% 准确率
figure
plot(K,accuracy11,'.-','LineWidth',2,'color','g','MarkerSize',30)
hold on
plot(K,accuracy1all,'.-','LineWidth',2,'color','b','MarkerSize',30)
xlabel('K')
ylabel('Accuracy')
title('不同K值下的准确率(使用PCA-SVD)')
legend('One To One','One To All');
%% 对比SVM 的效率
figure
plot(K,t1end,'.-','LineWidth',2,'color','g','MarkerSize',30)
hold on
plot(K,t2end,'.-','LineWidth',2,'color','b','MarkerSize',30)
xlabel('K')
ylabel('TimeCost')
title('不同K值下的SVM代价时间(使用PCA-SVD)')
legend('One To One','One To All');

%% 找几张错误的图片
ori = misclasses1all{4}{1}(:,1);
pre = misclasses1all{4}{1}(:,2);
difference = find(ori~=pre);
ll = min(length(difference),10); % 最多10张
for i=1:ll
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


