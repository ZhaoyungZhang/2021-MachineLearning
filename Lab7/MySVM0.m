%% ================1to多 SVM多分类================
% 1对多 即任意选出一个类进行训练 这类是正其余都是负
% 训练 40次
function [misclass,hit_rate] = MySVM0(train_X,test_X,class_num,k,train_percent)
% Step 1: 
    class_nums = 40;
    data1 = cell(class_nums,1);
    data2 = cell(class_nums,1);
    for i = 1:class_nums
       data1{i,1} = LabelData(i,train_X,7);
       data2{i,1} = LabelData(i,test_X,3);
    end
% Step 2: 训练
    A = zeros(class_nums,size(train_X,1));
    W = zeros(class_nums,size(train_X,2));
    B = zeros(class_nums,1);
    for i = 1:class_nums
       svm = MysvmTrain(data1{i,1}(:,1:k),data1{i,1}(:,k+1),1);
       A(i,:) =  svm.a';
       for j = 1:size(train_X,2)
          W(i,j) = sum(A(i,:)'.*data1{i,1}(:,k+1).*data1{i,1}(:,j));
       end
       B(i,1) = sum(svm.Ysv-svm.Xsv*W(i,:)')/svm.svnum;
    end
% Step 3 测试
    labels_num = zeros(size(test_X,1),class_nums);
    for i = 1:class_nums
       labels_num(:,i) = (test_X*W(i,:)')'+B(i,1); 
    end
    predict_labels = zeros(size(test_X,1),1);
    % 转化为标签的具体整数值
    for i = 1:size(test_X,1)
        [~,index] = max(labels_num(i,:));
        predict_labels(i,1) = index;
    end
    ori_labels = zeros(size(test_X,1),1);
    %有规律的换算过去，得到测试集合的标签
    for i = 1:class_nums
       ori_labels((i-1)*3+1:(i-1)*3+3,1) = i; 
    end
    hit = find(ori_labels==predict_labels); % 160*1全是label
    hit_rate = length(hit)/size(test_X,1);
    % 找出误分类的
    miss = find(ori_labels~=predict_labels);
    % 记录误分类
    misclass = cell(1,1);
    misclass{1} = [ori_labels,predict_labels];
    fprintf(' =======Test Finish!=======\n');
    fprintf(' =======Accuracy = %f=======\n',hit_rate);
    
end