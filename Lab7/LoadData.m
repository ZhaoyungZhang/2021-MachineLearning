%% ================读取数据的函数================
function [train_datas,test_datas] = LoadData(train_percent,dimx,dimy,class_num)
    % train_datas: 40*7,112*92 
    fprintf(' Train_Percent = %d%%\n',train_percent*10);
    fprintf(' ================Loading Datas...================\n');
    path = ['orl_faces\'];
    test_percent = 10-train_percent;
    for i = 1 : class_num
        personid = ['s',num2str(i)];
        index = randperm(10);
        % train Data
        %fprintf(' ================Loading Training Datas...================\n');
        for j = 1:train_percent
            filename = [path,personid,'\',num2str(index(j)),'.pgm'];
            img = double(imread(filename));
            % 这里存取没用张量，而是每张图片flatten存
            train_datas(train_percent*(i-1)+j,:) = reshape(img,1,dimx*dimy);
        end
        % test Data
        
        for j = 1:test_percent
            filename = [path,personid,'\',num2str(index(j)),'.pgm'];
            img = double(imread(filename));
            % 这里存取没用张量，而是每张图片flatten存
            test_datas(test_percent*(i-1)+j,:) = reshape(img,1,dimx*dimy);
        end
    end
    fprintf(' ================Loading Finish!================\n');
end