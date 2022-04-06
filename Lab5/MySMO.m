function [alpha,bias] = MySMO(training_X,Labels,C,maxItertimes,tolerance)

% init
[sampleNum,featuerNum]=size(training_X);
alpha=zeros(sampleNum,1);
bias=0;
iteratorTimes=0;

K=training_X*training_X';%计算K
while iteratorTimes<maxItertimes
    alphaPairsChanged=0;% 记录变化
    % find alpha1
    for i=1:sampleNum
        g1=(alpha.*Labels)'*(training_X*training_X(i,:)')+bias;
        Error1=g1-Labels(i,1);% 计算error
       % choose i: avoid KKT conditions
       % 选取i的标准 error
       if(((Error1*Labels(i,1) < -tolerance)&&alpha(i,1)<C)||...
           ((Error1*Labels(i,1)>tolerance)&&alpha(i,1)>0))
           % choose j: different from i 
           j=i;
           while j==i
                j=randi(sampleNum);% 随机另外一个alpha2
           end

            alpha1=i;
            alpha2=j;

            % 更新alpha1 & alpha2
            alpha_upd=alpha(alpha1,1);
            alpha_upd=alpha(alpha2,1);
            y1=Labels(alpha1,1);
            y2=Labels(alpha2,1);

            g2=(alpha.*Labels)'*(training_X*training_X(j,:)')+bias;
            E2=g2-Labels(j,1);%计算error2
            % 计算Lower & Higher
            if y1~=y2
                L=max(0,alpha_upd-alpha_upd);
                H=min(C,C+alpha_upd-alpha_upd);
            else
                L=max(0,alpha_upd+alpha_upd-C);
                H=min(C,alpha_upd+alpha_upd);
            end

            if L==H
                fprintf('H==L\n');
                continue;
            end

            parameter=K(alpha1,alpha1)+K(alpha2,alpha2)-2*K(alpha1,alpha2);

            if parameter<=0
                fprintf('parameter<=0\n');
                continue;
            end
               % 得到新的alpha2
            alpha2New=alpha_upd+y2*(Error1-E2)/parameter;

            if alpha2New>H
                alpha2New=H;
            end

            if alpha2New<L
                alpha2New=L;
            end

            if abs(alpha2New-alpha_upd)<=0.0001
                fprintf('change small\n');
                continue;
            end
            %得到新的alpha1
            alpha1New=alpha_upd+y1*y2*(alpha_upd-alpha2New);

            % updata bias
            bias1=-Error1-y1*K(alpha1,alpha1)*(alpha1New-alpha_upd)-y2*K(alpha2,alpha1)*(alpha2New-alpha_upd)+bias;
            bias2=-E2-y1*K(alpha1,alpha2)*(alpha1New-alpha_upd)-y2*K(alpha2,alpha2)*(alpha2New-alpha_upd)+bias;

            if alpha1New>0&&alpha1New<C
                bias=bias1;
            elseif alpha2New>0&&alpha2New<C
                bias=bias2;
            else
                bias=(bias2+bias1)/2;
            end

            alpha(alpha1,1)=alpha1New;
            alpha(alpha2,1)=alpha2New;
            alphaPairsChanged=alphaPairsChanged+1;
       end  
    end

    if alphaPairsChanged==0
        iteratorTimes=iteratorTimes+1;
    else
        iteratorTimes=0;
    end
    fprintf('iteratorTimes=%d\n',iteratorTimes);

end