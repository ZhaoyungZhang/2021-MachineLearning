clear,clc
% train_nums = [100,500,1000,3000,5000,8000,10000];% 训练集数目
% train_times = 7;
train_nums = 10000;% 训练集数目
train_times = 1;
test_num = 0;
for sample_c = 1:train_times
    train_num = train_nums(sample_c);
    test_num = 12960 - train_num;
    convData(train_num);% 
    
    train_datas = load('training_data.txt');
    test_datas = load('test_data.txt');
    laplaces = get_laplaces(train_datas);
    % random sample
    
    [m1,n] = size(train_datas);
    %m1 is num of train  m2 is test
    [m2,n] = size(test_datas);% 行数 列数
    k = 5;
    py = [0,0,0,0,0]; % 对应五个类别的概率0-4
    count_0 = zeros(8,5); %y=0 下8个feature 每5个可能的取值 的count
    count_1 = zeros(8,5);
    count_2 = zeros(8,5);
    count_3 = zeros(8,5);
    count_4 = zeros(8,5);
    xx_train = train_datas(:,1:8);
    yy_train = train_datas(:,9);
    xx_test = test_datas(:,1:8);
    yy_test = test_datas(:,9);
    count_y = zeros(5,1);
    %看第i条数据 第j个属性取值K
    for i = 1:m1
        if(yy_train(i)==0)
            count_y(1) = count_y(1) + 1;
            for j = 1:8
%                 for k = 1:5
%                     if(xx_train(i,j)==k-1)
%                         count_0(j,k) = count_0(j,k) + 1;
%                         break; % 只有一个取值
%                     end
%                 end
                %！ 这么写节省一层循环 ！错了！
                count_0(j,xx_train(i,j)+1 ) = count_0(j,xx_train(i,j)+1 ) + 1;
            end
        
        elseif(yy_train(i)==1)
            count_y(2) = count_y(2) + 1;
            for j = 1:8
                count_1(j,xx_train(i,j)+1 ) = count_1(j,xx_train(i,j)+1 ) + 1;
            end
        
        elseif(yy_train(i)==2)
            count_y(3) = count_y(3) + 1;
            for j = 1:8
                count_2(j,xx_train(i,j)+1 ) = count_2(j,xx_train(i,j)+1 ) + 1;
            end
        
        elseif(yy_train(i)==3)
            count_y(4) = count_y(4) + 1;
            for j = 1:8
                count_3(j,xx_train(i,j)+1 ) = count_3(j,xx_train(i,j)+1 ) + 1;
            end
        
        elseif(yy_train(i)==4)
            count_y(5) = count_y(5) + 1;
            for j = 1:8
                count_4(j,xx_train(i,j)+1 ) = count_4(j,xx_train(i,j)+1 ) + 1;
            end
        end
    end
    
    %%
    p_all_y = get_all_py_LS(count_y,m1,k);
    %% 计算pj(x|y)  使用了 LS  最后的pj(x|y) 应该是 5*8*5的tensor
    py1_j = zeros(8,5);py2_j = zeros(8,5);py3_j = zeros(8,5); py4_j = zeros(8,5);py5_j = zeros(8,5);
    for i = 1:8
        for j = 1:5
            py1_j(i,j) = (count_0(i,j)+1)/(count_y(1)+laplaces(i));
            py2_j(i,j) = (count_1(i,j)+1)/(count_y(2)+laplaces(i));
            py3_j(i,j) = (count_2(i,j)+1)/(count_y(3)+laplaces(i));
            py4_j(i,j) = (count_3(i,j)+1)/(count_y(4)+laplaces(i));
            py5_j(i,j) = (count_4(i,j)+1)/(count_y(5)+laplaces(i));
        end
    end
    %% test 进行测试
    hit_count = test(m2,p_all_y,py1_j,py2_j,py3_j,py4_j,py5_j,xx_test,yy_test);
    accuracy = hit_count / test_num;
end


%% 统计每个属性的可能取值个数
function vjs = get_laplaces(xx_train)
    vjs = zeros(8,1);
    % 假设从1开始连续
    for i=1:8
        vjs(i) = max(xx_train(:,i)) + 1;
    end
end

%% 计算pys  使用了 LS 
function p_ys = get_all_py_LS(count_y,m1,k) 
    p_ys = zeros(5,1);
    for i = 1:5
        p_ys(i) = (count_y(i) + 1) / (m1+k);
    end
end

function hit_count = test(m2,p_all_y,py1_j,py2_j,py3_j,py4_j,py5_j,xx_test,yy_test)
    hit_count = 0;
    xx_test = xx_test + 1;
    for i = 1:m2
        %计算每种类别的概率 py* 累乘 pjxy
        poss_1 = p_all_y(1);poss_2 = p_all_y(2);poss_3 = p_all_y(3);poss_4 = p_all_y(4);poss_5 = p_all_y(5);
        for j = 1:8
            poss_1 = poss_1 * py1_j(j,xx_test(i,j));
            poss_2 = poss_2 * py2_j(j,xx_test(i,j));
            poss_3 = poss_3 * py3_j(j,xx_test(i,j));
            poss_4 = poss_4 * py4_j(j,xx_test(i,j));
            poss_5 = poss_5 * py5_j(j,xx_test(i,j));
        end
        %softmax
        sum_p = poss_1+poss_2+poss_3+poss_4+poss_5;
        po1 = poss_1 / sum_p;
        po2 = poss_2 / sum_p;
        po3 = poss_3 / sum_p;
        po4 = poss_4 / sum_p;
        po5 = poss_5 / sum_p;
        poss = [po1;po2;po3;po4;po5];
        [val,position] = max(poss);
        if position == yy_test(i)+1
            hit_count = hit_count + 1;
        end
    end
end

