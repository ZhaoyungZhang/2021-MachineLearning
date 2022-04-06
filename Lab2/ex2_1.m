%% Load data && Plot to watch data 
x = load('ex2x.dat');
y = load('ex2y.dat');
len = length(x);
x = [ones(len,1),x];
% 进行Normalization


m = size(x,1);
n = size(x,2);
pos = find(y == 1);
neg = find(y == 0);
% subplot(1,2,1)
% plot(x(pos,2),x(pos,3),'+');
% hold on
% plot(x(neg,2),x(neg,3),'o');
% hold off

sigma = std(x);
mu = mean(x);
%x(:,2) = ( x(:,2) - mu(2) ) ./ sigma(2);
%x(:,3) = ( x(:,3) - mu(3) ) ./ sigma(3);
% subplot(1,2,2)
plot(x(pos,2),x(pos,3),'+');
hold on
plot(x(neg,2),x(neg,3),'o');


%% Logistic Regression
%function g=sigmoid(z)  h_theta(x) = g(theta*x)

%ii = [];
%for a = 1:15
    a = 0.001;
    thetas = [0;1;1];
    %thetas = zeros(3,1);
    not_cv = 1; % 未收敛
    L = get_L(thetas,x,y);
    while not_cv == 1
        %thetas = [thetas,get_new_theta_NTM(thetas,x,y)];
        thetas = [thetas,get_new_theta(thetas,a,x,y)];
        t = thetas(:,size(thetas,2));
        L = [L,get_L(t,x,y)];
        if judge_convergence(L) == 1
            not_cv = 0;
        end
    end
    iterations_times = size(thetas,2);%迭代次数
    %ii = [ii;iterations_times];
   

    xx = 1 : 1 : iterations_times;
    %plot(a,iterations_times,'-')
    %leg_str{a} = ['\alpha = ' num2str(a)];
    %hold on
    % predict score of (20,80)
    %     score = [1;20;80];
    %     score(2) = ( score(2) - mu(2) ) ./ sigma(2);
    %     score(3) = ( score(3) - mu(3) ) ./ sigma(3);
    % p_y = 1 - sigmoid(sum(score.*conv_theta)) %0.6680
%end
% plot(xx,iterations_times,'-')
% xlabel('\alpha')
% ylabel('iteration times')
%legend(leg_str);
    

%% Final theta
conv_theta = thetas(:,iterations_times);
% Plot the decision boundary
hold on;
x2 = x(:,2);
x3 = (-conv_theta(1).*x(:,1) - conv_theta(2).*x(:,2))/conv_theta(3);
plot(x2, x3,'-','color','r');
legend('Positive','Negative','\theta^{T}X=0')

xx = 1 : 1 : iterations_times;
figure
plot(xx,L,'b--.','color','r','MarkerSize',30)
xlabel('Iteration')
ylabel('L(\theta)')
%legend('\alpha = 6');
title('Newton Method')%标题
% predict score of (20,80)
score = [1;20;80];
score(2) = ( score(2) - mu(2) ) ./ sigma(2);
score(3) = ( score(3) - mu(3) ) ./ sigma(3);
p_y = 1 - sigmoid(sum(score.*conv_theta)) %0.6680

%% Get new thetas
function f = get_new_theta(thetas,a,x,y)
    l = length(thetas(1,:));
    t = thetas(:,l);
    new_t = t - a * get_gradientL(t,x,y);
    f = new_t;
end

%% Get gradient of L(theta)
function f = get_gradientL(t,x,y)
    %g = @(z)(1.0 ./ (1.0 + exp(-z)));
    m = length(x);
    h = sigmoid(x*t);
    f = 1/m * sum( (h-y) .* x );
    f = f';
end

%% Calculate L(theta) target: min L
function f = get_L(t,x,y)
    %g = @(z)(1.0 ./ (1.0 + exp(-z)));
    m = length(x);
    h = sigmoid(x*t);
    f = 1/m * sum( (-y.*log(h)) - ((1-y).*log(1-h)) );
end

%% Judge if convergence
function f = judge_convergence(L)
    l = length(L);
    if abs(L(1,l) - L(1,l-1)) <= 10^(-8)
        f = 1;%收敛了
    else
        f  = 0;
    end
end

%% Newton's Method to update theta
function f = get_new_theta_NTM(thetas,x,y)
    leng = length(thetas(1,:));
    t = thetas(:,leng);

    %g = @(z)(1.0 ./ (1.0 + exp(-z)));
    m = length(x);
    n = size(x, 2);
    
    h = sigmoid(x*t);
    G = zeros(n , 1);
    Hes = zeros(n, n);
    for i = 1:n
        dif = y - h;
        G(i) = sum(dif.* x(:, i), 1)/m;
        const_sum = h.*(ones(m,1)-h);
        for j = 1:n
            Hes(i,j) = -sum(const_sum.* x(:, i).*x(:,j), 1)/m;
        end
    end
    new_t = t - inv(Hes)*G;
    f = new_t;
end

%% Sigmoid 改进函数
function f = sigmoid(x)
%g = @(z)(1.0 ./ (1.0 + exp(-z)));
    n = size(x,1);
    y = [];
    for i = 1:n
        xx = x(i,1);
        if(x(i,1) >= 0)
            y = [y;1.0 / (1.0 + exp(-xx))];
        else
            y = [y; exp(xx) / ( 1.0 + exp(xx) ) ];
        end
    end
    f = y;
end

