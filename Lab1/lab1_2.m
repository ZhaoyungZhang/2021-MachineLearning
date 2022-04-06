%多元线性回归 y = theta0 * 1 + theta1 * x1 + theta2*x2;
x = load('ex1_2x.dat');%47 * 2 看作X1 X2？
y = load('ex1_2y.dat');
m = length(y);
x = [ones(m,1),x];%填一列 针对theta0  “Linear”
%% 内置包
%[b,bint,r,rint,stats]=regress(y,x);
%% 数据预处理 归一化
%  原理：正则化
sigma = std(x);
mu = mean(x);
x(:,2) = ( x(:,2) - mu(2) ) ./ sigma(2);
x(:,3) = ( x(:,3) - mu(3) ) ./ sigma(3);

%% 得到最好的alpha = 0.96
%  原理 看theta 变化小于10^(-4)
count = 1;
total_nums = 50 %一次多少个alpha
Interations = zeros(1,total_nums);

for alpha = 0.6:0.01:1.1
    theta = zeros(size(x(1,:)))';%initial
    J = zeros(50,1);
    Interations_times = 0;
    has_track = 0;
    
    for i = 1:50
        J(i,:) = get_Jv(theta,m,x,y);%计算损失函数   
        t = zeros(3,1);
        t(:,1) = get_new_theta(theta,alpha,3,m,x,y);%梯度下降法更新
        theta = [theta,t(:,1)];  
        
        Interations_times = Interations_times + 1;
        flag = judge(theta);
        if flag == 1 && has_track == 0
            has_track = 1;
            Interations(:,count) = Interations_times;
        end
    end
    
    if has_track == 0
        Interations(:,count) = 0;
    end
    
    plot(0:49,J(1:50),'-') 
    leg_str{count} = ['\alpha = ' num2str(alpha)];
    count = count +1;
    hold on
end
    xlabel('Number of Interations');
    ylabel('Cost J');
    legend(leg_str) %循环图例

%% 绘制迭代次数和a的关系 从 0.8 - 1.2 步长 0.01
figure
xx = 0.6:0.01:1.1;
plot(xx,Interations,'color','b')
ylabel('Number of Interations');
xlabel('\alpha');

%% 
alpha = 0.98;%best
theta = zeros(size(x(1,:)))';%initial
J = zeros(50,1);
    
%迭代求theta
for i = 1:1500
    t = zeros(3,1);
    t(:,1) = get_new_theta(theta,alpha,3,m,x,y);%梯度下降法更新
    theta = [theta,t(:,1)];
end
L = length(theta);
best_theta = theta(:,L)
    
%%  Predict
p_x = [1,1650,3];
p_x(:,2) = ( p_x(:,2) - mu(2) ) ./ sigma(2);
p_x(:,3) = ( p_x(:,3) - mu(3) ) ./ sigma(3);
p_y = p_x * best_theta
    
    
    
%% 计算Loss Function
function f = get_Jv(thetas,m,x,y)    
    %在次theta下的拟合值 
    L = length(thetas(1,:));%要取最后一个 最新的seta
    t = [];
    t = [thetas(1,L);thetas(2,L);thetas(3,L)];
    
    est_y = x * t;
    deta_y = [];
    deta_y = est_y - y;%偏差向量
    deta_y = deta_y.^2;
    f = []; 
    f = 1/(2*m) * sum(deta_y);
end

%% 得到新theta函数
function f = get_new_theta(thetas,a,n,m,x,y)
%input:
%output:
    L = length(thetas(1,:));%要取最后一个 最新的seta
    old_theta = [];
    old_theta = [thetas(1,L);thetas(2,L);thetas(3,L)];
    %在次theta下的拟合值 
    est_y = x * old_theta;
    deta_y = est_y - y;%偏差向量
    new_theta=[];
    for j = 1 : n %分别求新的theta
        new_theta(j,:) = old_theta(j,:)-a/m*sum(deta_y.*x(:,j));
    end 
    f = new_theta;
end

%% 判断是否收敛 1 收敛 0不收敛
function f = judge(thetas)
    L = length(thetas(1,:));%要取最后一个 最新的seta
    new_theta = [];
    new_theta = [thetas(1,L);thetas(2,L);thetas(3,L)];
    old_theta = [];
    old_theta = [thetas(1,L-1);thetas(2,L-1);thetas(3,L-1)];
    deta_theta = abs(old_theta - new_theta);
    flag = 1;
    for i = 1:3
        if deta_theta(i,:) > 10^(-2)
            flag = 0;
            break
        end
    end
    f = flag;
end

