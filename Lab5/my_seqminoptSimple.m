function [alpha,bias] = my_seqminoptSimple(training,groupIndex,C,maxIter,tol)

% init
[sampleNum,featuerNum]=size(training);
alpha=zeros(sampleNum,1);
bias=0;
iteratorTimes=0;

K=training*training';
while iteratorTimes<maxIter
    %iteratorTimes=iteratorTimes+1;
    alphaPairsChanged=0;
    % calculate predict value
    %K=training*training';
    %g=(alpha.*groupIndex)'*K+repmat(bias,1,sampleNum);
    %g=g';

    % calculate error
    %E=g-groupIndex;

    % find alpha1
    for i=1:sampleNum
        g1=(alpha.*groupIndex)'*(training*training(i,:)')+bias;
        E1=g1-groupIndex(i,1);
       % choose i: avoid KKT conditions
       if(((E1*groupIndex(i,1)<-tol)&&alpha(i,1)<C)||((E1*groupIndex(i,1)>tol)&&alpha(i,1)>0))
           % choose j: different from i 
           j=i;
           while j==i
                j=randi(sampleNum);
           end

            alpha1=i;
            alpha2=j;

            % updata alpha1 and alpha2
            alpha1Old=alpha(alpha1,1);
            alpha2Old=alpha(alpha2,1);
            y1=groupIndex(alpha1,1);
            y2=groupIndex(alpha2,1);

            g2=(alpha.*groupIndex)'*(training*training(j,:)')+bias;
            E2=g2-groupIndex(j,1);

            if y1~=y2
                L=max(0,alpha2Old-alpha1Old);
                H=min(C,C+alpha2Old-alpha1Old);
            else
                L=max(0,alpha2Old+alpha1Old-C);
                H=min(C,alpha2Old+alpha1Old);
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

            alpha2New=alpha2Old+y2*(E1-E2)/parameter;

            if alpha2New>H
                alpha2New=H;
            end

            if alpha2New<L
                alpha2New=L;
            end

            if abs(alpha2New-alpha2Old)<=0.0001
                fprintf('change small\n');
                continue;
            end

            alpha1New=alpha1Old+y1*y2*(alpha2Old-alpha2New);

            % updata bias
            bias1=-E1-y1*K(alpha1,alpha1)*(alpha1New-alpha1Old)-y2*K(alpha2,alpha1)*(alpha2New-alpha2Old)+bias;
            bias2=-E2-y1*K(alpha1,alpha2)*(alpha1New-alpha1Old)-y2*K(alpha2,alpha2)*(alpha2New-alpha2Old)+bias;

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