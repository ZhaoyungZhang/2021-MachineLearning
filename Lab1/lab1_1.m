% EXE1 CODE：
%一维数据 h(x) = theta1*X1 + theta2
x1 = load('ex1_1x.dat');
y1 = load('ex1_1y.dat');
%%
m1 = length(y1);
x1 = [ones(m1,1),x1];
%% 线性回归的包
[b ,bint , r ,rint , s]=regress(y1 , x1 , 0.07)
%%
% h seta (x) = seta * x %X0 正好是0 所以后面一个截距
% J(seta) = 误差平方和
% 更新方法： seta j = seta j - alpha*1/m*sum(误差*X)

%%
thetas = [0;0]; %初始化theta_1 theta_0
a = 0.07;
n = 2;
for i = 1:1500 %迭代1500次认为收敛？
    new_theta = get_new_theta(thetas,a,n,m1,x1,y1);
    thetas = [thetas,new_theta];
end
len = length(thetas(1,:));
best_theta = thetas(:,len)
plot(x1(:,2),y1,'o');
xlabel('Age in years');
ylabel('Height in meters');
%best_theta = b;
hold on
plot(x1(:,2),x1*best_theta,'-')
legend('Training data','Linear Regeression')

%% 计算J 与 theta的关系
J_vals = zeros(100,100); %初始化 J
theta0_vals = linspace(-3,3,100);
theta1_vals = linspace(-1,1,100);
%theta0_vals = [-theta0_vals,theta0_vals];
%theta1_vals = [-theta1_vals,theta1_vals];

 for i = 1:length(theta0_vals)
     for j = 1:length(theta1_vals)
         t = [theta0_vals(i);theta1_vals(j)];
         J_vals(i,j) = get_J_vals(t,m1,x1,y1);
     end
 end
% Plot the surface plot
% Because of the way meshgrids work in the surf command, we
% need to transpose J_vals before calling surf , or else the
% axes will be flipped
J_vals = J_vals';
figure;
surf(theta0_vals,theta1_vals,J_vals);
xlabel('\theta_0');ylabel('\theta_1');
zlabel('J(\theta)')
title('The relationship between J and \theta')

figure;
contour(theta0_vals,theta1_vals,J_vals,'ShowText','on');
xlabel('\theta_0');ylabel('\theta_1');
title('The relationship between J and \theta')

%预测
pre_y = [1 3.5;
    1 7] * best_theta; 

%% 得到新theta
function f = get_new_theta(thetas,a,n,m,x,y)
%input:
%output:
    l0 = length(thetas(1,:));%要取最后一个 最新的seta
    l1 = length(thetas(2,:));
    
    %得到上次的theta 一个 2*1的向量 分别对应seta0 seta1
    old_theta = [];
    old_theta = thetas(1,l0);
    old_theta = [old_theta;thetas(2,l1)];
    %在次theta下的拟合值 
    est_y = x * old_theta;
    deta_y = est_y - y;%偏差向量
    new_theta=[];
    for j = 1 : n %分别求新的theta
        new_theta(j,:) = old_theta(j,:)-a/m*sum(deta_y.*x(:,j));
    end 
    f = new_theta;
end

%%
function f = get_J_vals(t,m,x,y)    
    %在次theta下的拟合值 
    est_y = x * t;
    deta_y = [];
    deta_y = est_y - y;%偏差向量
    deta_y = deta_y.^2;
    f = 1/(2*m) * sum(deta_y);
end
    
    
    