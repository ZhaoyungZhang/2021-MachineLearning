%% ================1to1 SVM多分类================
% 1对1 即任意选出两个类作为二分类的SVM训练集
% 分别贴上+-标签，训练 C40 2 = 40*39/2 = 780次
function [misclass,hit_rate] = MySVM1(train_X,test_X,class_num,k,train_percent)
    % Step 1: 标签分配 
    tp = train_percent;
    other = 10 - train_percent;
    train_times = class_num*(class_num-1)/2;
    data1 = cell(train_times,1);
    data2 = cell(train_times,1);
    index = 1;
    OneOneLabels = zeros(train_times,2);
    for i = 1:39
            for j = i+1:40
                % 成对放进元组里面，并且对应进行记录
                data1{index,1} = [[train_X(i*tp-(tp-1):i*tp,:),ones(tp,1)];[train_X(j*tp-(tp-1):j*tp,:),-ones(tp,1)]];
                data2{index,1} = [[test_X(i*other-(other-1):i*other,:),ones(other,1)];[test_X(j*other-(other-1):j*other,:),-ones(other,1)]];
                OneOneLabels(index,1) = i;
                OneOneLabels(index,2) = j;
                index = index + 1;
            end
    end
    fprintf(' =======Strat Training...=======\n');
    % Step 2: 训练
    A = zeros(train_times,2*tp);
    W = zeros(train_times,size(train_X,2));
    B = zeros(train_times,1);
    for i = 1:train_times
       svm = MysvmTrain(data1{i,1}(:,1:k),data1{i,1}(:,k+1),1);
       A(i,:) =  svm.a';
       for j = 1:size(train_X,2)
          W(i,j) = sum(A(i,:)'.*data1{i,1}(:,k+1).*data1{i,1}(:,j));
       end
       B(i,1) = sum(svm.Ysv-svm.Xsv*W(i,:)')/svm.svnum;
    end
    fprintf(' =======Training Finish=======\n');
    % Step 3: 测试
    fprintf(' =======Strat Testing...=======\n');
    labels_num = zeros(size(test_X,1),train_times);
    for i = 1:train_times
       labels_num(:,i) = (test_X*W(i,:)')'+B(i,1); 
    end
    % 分类结果
    labels_svm = zeros(40,39);
    for i = 1:40
         labels_svm(i,:) = find(OneOneLabels(:,1)==i | OneOneLabels(:,2)==i);
    end
    %统计每一类在各个分类器分类正确率、个数，投票方案
    Predict_Labels = zeros(size(test_X,1),1);
    for i = 1:size(test_X,1)
        VoteCounts = zeros(40,1);
        for j = 1:40
           for m = 1:39
               if labels_num(i,labels_svm(j,m)) > 0 && OneOneLabels(labels_svm(j,m),1) == j
                   VoteCounts(j,1) = VoteCounts(j,1) + 1;
               elseif labels_num(i,labels_svm(j,m)) < 0 && OneOneLabels(labels_svm(j,m),2) == j
                   VoteCounts(j,1) = VoteCounts(j,1) + 1;
               end
           end  
        end
        [~,index] = max(VoteCounts);
        Predict_Labels(i,1) = index;
    end
    Ori_Labels = zeros(size(test_X,1),1);
    for i = 1:40
       Ori_Labels((i-1)*other+1:(i-1)*other+other,1) = i; 
    end
    hit = find(Ori_Labels==Predict_Labels); % 160*1全是label
    hit_rate = length(hit)/size(test_X,1);
    % 找出误分类的
    miss = find(Ori_Labels~=Predict_Labels);
    % 记录误分类
    misclass = cell(1,1);
    misclass{1} = [Ori_Labels,Predict_Labels];
    fprintf(' =======Test Finish!=======\n');
    fprintf(' =======Accuracy = %f=======\n',hit_rate);
end
% 测试集的标签看比例就能推出来