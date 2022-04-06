%% Regularized Linear Regression
x = load('ex3Linx.dat');
y = load('ex3Liny.dat');
scatter(x,y,'*','r')
hold on
m = size(x,1);
lamudas = linspace(0,10,11);
leg_str{1} = 'Traning Data';
T = [];
for i = 1:11
    order = 5;
    lamuda = lamudas(i);
    alpha = 0.03;
    xx = xdata_handler(x,order,m);
    thetas = get_theta(order,m,xx,y,lamuda);
    leg_str{i+1} = ['\lambda = ' num2str(lamuda)];
    lin_x = linspace(-1,1,100);lin_x = lin_x';
    lin_xx = xdata_handler(lin_x,order,100);
    y_hat = lin_xx*thetas;
    T =[T,thetas];
    plot(lin_x,y_hat)
end
legend(leg_str)
title('Ployfit under different \lambda')
hold off

% 画 6 个 feature 随着 lambda的变化
figure

for i = 1:1:6
    subplot(3,2,i)
    plot(1:10,T(i,2:end),'Linewidth',2,'Color','r')
    xlabel('\lambda')
    ystr = ['\theta' num2str(i)];
    ylabel(ystr);
end
sgtitle('Features under different \lambda')

%% x变换
function f = xdata_handler(x,order,m)
    xx = [ones(m,1),x]; 
    temp = x;
    for i = 2:order
        temp = temp .* x;
        xx = [xx,temp]; 
    end
    f = xx;
end

%% theta0 不算
function f = get_loss(thetas,x,y,lamuda,order,m)
    len = size(thetas,1);
    t = thetas(:,len);
    f = ( sum( (hypo(t, x,order,m)-y).^2)... 
         + lamuda*sum( t.^2) - t(1).^2 ) / (2*m);  
     %注意不包括 theta0
end

function f = hypo(t, x,order,m)
    f = 0;
    xx = [ones(m,1),x]; 
    temp = x;
    for i = 2:order
        temp = temp .* x;
        xx = [xx,temp]; 
    end
    % t is 6 * 1  xx is 7 * 6
    f = xx * t;
end

%% 得到新theta函数
function f = get_theta(n_feature,m,xx,y,lamuda)
%input:
%output:
    mt = zeros(n_feature+1,n_feature+1);
    for i = 2:n_feature+1
        mt(i,i) = 1;
    end
    
    f1 = xx'*xx + lamuda * mt;
    f = inv(f1) * xx' * y;
end