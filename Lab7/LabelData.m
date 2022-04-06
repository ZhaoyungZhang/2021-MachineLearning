%% ================标注Data的label================
function data = LabelData(index,X,n)
    len = size(X,1);
    if index == 1
        % 把1-7 表正，其余的标-1
        data = [[X(1:n,:),ones(n,1)];[X(n+1:len,:),-ones(len-n,1)]];
    else
        % 找到对应样本的位置，选出来标1 其余的标-1，分了三段
        data = [[X(1:n*(index-1),:),-ones(n*(index-1),1)];[X(n*index-n+1:n*index,:),ones(n,1)];[X(n*index+1:len,:),-ones(len-n*index,1)]];
    end
end
