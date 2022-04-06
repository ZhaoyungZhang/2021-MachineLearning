clear,clc
%% 多进行几次，目的是初始点不同，找一个效果最好的。
%% K-means
% small_img : 128 * 128 * 3
% large_img : 538 * 538 * 3
small_img = imread('bird_small.tiff');
large_img = imread('bird_large.tiff');
% 转化为double 矩阵 A(50,33,3) 表示 y = 50 x =33 的Blue 的val
small_img_matrix = double(small_img);
large_img_matrix = double(large_img);
Best_new_small_pic=[];
Best_new_large_pic=[];
%% show image 
% imshow(large_img)
%%
inital_times = 10;
Loss_large_inital_times = zeros(inital_times,1);
Loss_small_inital_times = zeros(inital_times,1);
for inital_time = 1 : inital_times
    K = 16; % 16个class
    [small_dimx,small_dimy,small_dimz] = size(small_img_matrix);
    [large_dimx,large_dimy,large_dimz] = size(large_img_matrix);
    max_iter_times = 200;
    convergence_condition = 10^-5;
    [centroids,Clusters] = kmeans(K,small_img_matrix,max_iter_times,convergence_condition);
    %% 重新展示小图片
    loss_small = 1e9;
    loss_large = 1e9;
    new_small_pic=[];
    new_large_pic=[];
    % 小图片处理
    for i=1:small_dimx
        for j=1:small_dimy
            new_small_pic(i,j,:) =  centroids(Clusters(i,j),:);
        end
    end
    new_loss_small = calculate_Kmeans_Loss(small_img_matrix,new_small_pic,small_dimx,small_dimy);
    Loss_small_inital_times(inital_time) = new_loss_small;
    if new_loss_small < loss_small
        loss_small = new_loss_small;
        Best_new_small_pic = new_small_pic;
    end

    % 大图片处理
    for i=1:large_dimy
        for j=1:large_dimx
            pixel_rgb = [large_img_matrix(i,j,1),large_img_matrix(i,j,2),large_img_matrix(i,j,3)];
            diff = ones(K,1)*pixel_rgb - centroids;
            distance = sum(diff.^2,2);
            [~,class] = min(distance); % 得到类别
            new_large_pic(i,j,:) =  centroids(class,:);
        end
    end
    
    new_loss_large = calculate_Kmeans_Loss(large_img_matrix,new_large_pic,large_dimy,large_dimx);
    Loss_large_inital_times(inital_time) = new_loss_large;
    if new_loss_large < loss_large
        loss_large = new_loss_large;
        Best_new_large_pic = new_large_pic;
    end
end
%% 展示小图片 和 大图片
figure
subplot(2,1,1);
imshow(uint8(small_img_matrix)),title('Origin Small Image');
subplot(2,1,2);
imshow(uint8(Best_new_small_pic)),title('Best New Small Image')
imwrite(uint8(Best_new_small_pic),'Best New Small Image.png')

figure
subplot(2,1,1);
imshow(uint8(large_img_matrix)),title('Origin Large Image');
subplot(2,1,2);
imshow(uint8(Best_new_large_pic)),title('Best New Large Image')
imwrite(uint8(Best_new_large_pic),'Best New Large Image.png')

%% 画初始点次数和最终loss的关系
iter_times = 1:10;
figure
plot(iter_times,Loss_small_inital_times,'LineWidth',3)
title('Loss On Different Initial Points Of Small');
xlabel('Inital Times')
ylabel('Loss')

figure
plot(iter_times,Loss_large_inital_times,'LineWidth',3,'color','r')
title('Loss On Different Initial Points Of Large');
xlabel('Inital Times')
ylabel('Loss')

%% K-Means Fuction Return ?
function [centroids,Clusters] = kmeans(K,datas,max_iter_times,convergence_condition)
    % 返回中心点 K * dimz
    centroids = [];
    [dimy,dimx,dimz] = size(datas);
    % 随机初始化中心点 K*3
    centroids = sample_random(K,datas,dimx,dimy);
    for it = 1 : max_iter_times
        % 计算its nearest mean
        Clusters = calculate_Ci(centroids,dimx,dimy,datas,K);
        % 更新中心点
        new_centroids = updatemeans(Clusters,dimx,dimy,dimz,datas,K);
        % 看是否收敛
        convergence = judge(new_centroids,centroids,convergence_condition);
        centroids = new_centroids;
        if convergence
            break
        end
    end
end

%% 随机选取初始点 return K * 3 
function sample_Datas = sample_random(num,datas,dimx,dimy)
    % datas 为原始数据 num 为目标数目 
    sample_Datas = zeros(num,3);
    a = rand(dimx,1);
    b = rand(dimy,1);
    [vala,posa] = sort(a);
    [valb,posb] = sort(b);
    chose_x = posa(1:num,1);
    chose_y = posb(1:num,1);
    for i=1:num
        sample_Datas(i,:) = datas(chose_x(i),chose_y(i),:);
    end
end

%% 计算每个pixel 的 类别 return Clusters: N * N val 为 类别 
function Clusters = calculate_Ci(centroids,dimx,dimy,datas,K)
    Clusters = [];
    % 遍历每个pixel 计算一个z(i,j)
    for i= 1:dimy
        for j = 1:dimx
            % 得到xi
            pixel_rgb = [datas(i,j,1),datas(i,j,2),datas(i,j,3)];
            diff = ones(K,1)*pixel_rgb - centroids;
            distance = sum(diff.^2,2);
            [~,class] = min(distance); % 得到最小的对应的类别的index
            Clusters(i,j) = class;
        end
    end
end

%% 更新中心点 return 16 * 3
function newcenters = updatemeans(Clusters,dimx,dimy,dimz,datas,K)
    newcenters = zeros(K,dimz);
    nums =  zeros(K);
    sumred = zeros(K,1);
    sumgreen = zeros(K,1);
    sumblue = zeros(K,1);
    for i=1:dimx
        for j=1:dimy
            class = Clusters(i,j);
            nums(class) = nums(class) + 1;
            sumred(class) = sumred(class) + datas(i,j,1);
            sumgreen(class) = sumgreen(class) + datas(i,j,2);
            sumblue(class) = sumblue(class) + datas(i,j,3);
        end
    end
    for i=1:K
        if nums(i) ~= 0
            sumred(i) = sumred(i) / nums(i);
            sumgreen(i) = sumgreen(i) / nums(i);
            sumblue(i) = sumblue(i) / nums(i);
        end
    end
    newcenters = [sumred,sumgreen,sumblue];
    newcenters = round(newcenters);
end

%% 判断是否收敛
function convergence = judge(newcenter,oldcenter,condition)
    convergence = 0;
     d = sum(sqrt(sum((newcenter - oldcenter).^2, 2)));
    if d < condition
        convergence = 1;
    end
end

%% 计算K-Means 之后的图片的loss：计算所有pixel的rgb sum 然后 avg
function Loss1 = calculate_Kmeans_Loss(ori_img,new_img,dimx,dimy)
    Loss = zeros(3,1);
    div_num = dimx*dimy;
    for i=1:dimx
        for j = 1 :dimy
            Loss(1) = Loss(1) + (ori_img(i,j,1)-new_img(i,j,1)).^2 / div_num;
            Loss(2) = Loss(2) + (ori_img(i,j,2)-new_img(i,j,2)).^2 / div_num;
            Loss(3) = Loss(3) + (ori_img(i,j,3)-new_img(i,j,3)).^2 / div_num;
        end
    end
    Loss1 = sum(Loss) / 3;
end

