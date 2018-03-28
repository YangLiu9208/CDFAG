function [autoencoder_s,autoencoder_t]=Bi_Shift_AutoEncoder(Xs,Xt,iter)
%Initialization for fc,gs,gt
% Xs=Normalize(Xs);
% Xt=Normalize(Xt);
hiddenSize=350;
%autoencoder_s 初始化
    autoencoder_s = feedforwardnet(hiddenSize);
    autoencoder_s.layers{1}.initFcn = 'initwb'; 
    autoencoder_s.layers{1}.transferFcn='logsig';
    autoencoder_s.performFcn='mse';%性能函数
    autoencoder_s.trainFcn = 'trainscg';
    autoencoder_s.trainParam.lr = 0.1;%学习速率
    autoencoder_s.trainParam.mc = 0.9;%momentum
    autoencoder_s= train(autoencoder_s,Xs,Xs);
    %autoencoder_t 初始化
    autoencoder_t = feedforwardnet(hiddenSize);
    autoencoder_t.layers{1}.initFcn = 'initwb'; 
    autoencoder_t.layers{1}.transferFcn='logsig';
    autoencoder_t.performFcn='mse';%性能函数
    autoencoder_t.trainFcn = 'trainscg';
    autoencoder_t.trainParam.lr = 0.1;%学习速率
    autoencoder_t.trainParam.mc = 0.9;%momentum
    autoencoder_t= train(autoencoder_t,Xt,Xt);
% autoencoder_s=trainAutoencoder(Xs,hiddenSize,'MaxEpochs',1000);%'ScaleData',false
% autoencoder_t=trainAutoencoder(Xt,hiddenSize,'MaxEpochs',1000);
for i=1:iter
    %step 1
    fprintf('Iter %d\n',i);
    Gs=sim(autoencoder_s,Xt);
    Gt=sim(autoencoder_t,Xs);
    %least angle regression
    Bs=[];
    Bt=[];
%     Gs=zero_mean_y(Gs);
%     Gt=zero_mean_y(Gt);
    for j=1:size(Gs,2)
    fprintf('LARS iter %d\n',j);
    bs = lars(Xt, Gs(:,j),'lasso' );
    bt= lars(Xs, Gt(:,j),'lasso' );
    Bs=[Bs; bs(end,:)];
    Bt=[Bt; bt(end,:)];
    end
    %step 2
    Xss=Xs*Bs';
    Xtt=Xt*Bt';
    %Xs_train=[Xs Xss];
    %Xt_train=[Xt Xtt];
    %Xs_train=Xss;
    %Xt_train=Xtt;
    %autoencoder_s 初始化
    autoencoder_s = feedforwardnet(hiddenSize);
    autoencoder_s.layers{1}.initFcn = 'initwb'; 
    autoencoder_s.layers{1}.transferFcn='logsig';
    autoencoder_s.performFcn='mse';%性能函数
    autoencoder_s.trainFcn = 'trainscg';
    autoencoder_s.trainParam.lr = 0.1;%学习速率
    autoencoder_s.trainParam.mc = 0.9;%momentum
    autoencoder_s= train(autoencoder_s,[Xs Xt],[Xs Xss]);
    %autoencoder_t 初始化
    autoencoder_t = feedforwardnet(hiddenSize);
    autoencoder_t.layers{1}.initFcn = 'initwb'; 
    autoencoder_t.layers{1}.transferFcn='logsig';
    autoencoder_t.performFcn='mse';%性能函数
    autoencoder_t.trainFcn = 'trainscg';
    autoencoder_t.trainParam.lr = 0.1;%学习速率
    autoencoder_t.trainParam.mc = 0.9;%momentum
    autoencoder_t= train(autoencoder_t,[Xt Xs],[Xt Xtt]);
end

end