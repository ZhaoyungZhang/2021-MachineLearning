%% Regularized Logistic Regression
x = load('ex3Logx.dat');
y = load('ex3Logy.dat');
figure
pos = find(y);neg = find(y==0);
plot(x(pos,1),x(pos,2),'+')
hold on
plot(x(neg,1),x(neg,2),'o')
u = x(:,1);v = x(:,2);
xx = map_feature(u,v); %117 * 28
m = size(xx,1); % 117
n = size(xx,2); % 28
leg_str{1}=['Positive'];
leg_str{2}=['Negative'];
lamudas = linspace(0,10,11);
linecolor = [0 0 1;...
             1 0 0;...
             0 1 0;...
             0 0 1;...
             1 1 0;...
             1 0 1;...
             0 1 1;...
             1 0.5 0;...
             0.5 0 0;...
             0.5 0.5 0.5;
             0.7 0.3 0.7];
 Loss = cell(11,1);  
for it =1:11
    lamuda = lamudas(it);
    leg_str{it+2} = ['\lambda = ' num2str(lamuda)];
    not_cv = 1; % 未收敛
    thetas = zeros(n,1);
    L = get_L(thetas,xx,y,lamuda);
    iter_times = 1;
    while not_cv == 1
        thetas = [thetas,get_new_theta_NTM(thetas,xx,y,lamuda,m,n)];
        t = thetas(:,size(thetas,2));
        L = [L,get_L(t,xx,y,lamuda)];
        iter_times = iter_times+1;
        if judge_convergence(L) == 1
            not_cv = 0;
        end
    end
    conv_t = thetas(:,iter_times);
    u = linspace(-1,1.5,200);
    v = linspace(-1,1.5,200);
    z = zeros(length(u),length(v));
    for i = 1:length(u)
        for j = 1:length(v)
            z ( i , j ) = map_feature (u( i ) ,  v( j ))*conv_t;
        end
    end
    z = z';
    contour(u,v,z,[0,0],'Linewidth',2,'Color',linecolor(it,:));
    Loss{it} = L;
    norm_theta(it) = norm(conv_t);
end
legend(leg_str)
xlabel('x')
ylabel('y')
hold off

figure
for i = 1:11
    it_times = size(Loss{i},2);
    j = 1:it_times;
    plot(j ,Loss{i},'Linewidth',1,'Color',linecolor(i,:))
    leg_str2{i} = ['\lambda = ' num2str(i-1)];
    hold on
end
legend(leg_str2)
xlabel('Iteration')
ylabel('Loss(\theta)')
%% 
figure
plot(1:10,norm_theta(1,2:end),'Linewidth',2,'Color','b')
xlabel('\lambda')
ylabel('L2 Norm(\theta)')
%% Newton's Method to update theta
function f = get_new_theta_NTM(thetas,x,y,lamuda,m,n)
    leng = size(thetas,2);
    t = thetas(:,leng);
    h = sigmoid(x*t);
    
    L = eye(n,n);
    L(1,1) = 0;
    Hes = (1/m).*x'*diag(h)*diag(1-h)*x + (lamuda/m)*L;
    
    G = (lamuda/m).*t; 
    G(1,1) = 0;
    J_delta = (1/m).*x'*(h-y) + G;
    new_t = t - inv(Hes)*J_delta;
    f = new_t;
end

%% Sigmoid 函数
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

%% Judge if convergence
function f = judge_convergence(L)
    l = size(L,2);
    if abs(L(1,l) - L(1,l-1)) <= 10^(-6)
        f = 1;%收敛了
    else
        f  = 0;
    end
end

%% Calculate Loss(theta) target: min L
function f = get_L(t,x,y,lamuda)
    m = length(x);
    h = sigmoid(x*t);
   % f = (1/m) * sum( (-y.*log(h)) - ((1-y).*log(1-h)) ) + (lamuda / (2*m)) * sum(t(2:end).^2);
    f = (1/m) * sum( (-y.*log(h)) - ((1-y).*log(1-h)) ) + (lamuda / (2*m)) * norm(t(2:end));
end