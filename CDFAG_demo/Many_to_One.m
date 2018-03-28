function [y,net]=Many_to_One(source,Target_N,test)

%---------------------------------------------------
% 指定训练参数
%---------------------------------------------------
% net.trainFcn = 'traingd'; % 梯度下降算法
% net.trainFcn = 'traingdm'; % 动量梯度下降算法
%
% net.trainFcn = 'traingda'; % 变学习率梯度下降算法
% net.trainFcn = 'traingdx'; % 变学习率动量梯度下降算法
%
% (大型网络的首选算法)
% net.trainFcn = 'trainrp'; % RPROP(弹性BP)算法,内存需求最小
%
% (共轭梯度算法)
% net.trainFcn = 'traincgf'; % Fletcher-Reeves修正算法
% net.trainFcn = 'traincgp'; % Polak-Ribiere修正算法,内存需求比Fletcher-Reeves修正算法略大
% net.trainFcn = 'traincgb'; % Powell-Beal复位算法,内存需求比Polak-Ribiere修正算法略大
%
% (大型网络的首选算法)
%net.trainFcn = 'trainscg'; % Scaled Conjugate Gradient算法,内存需求与Fletcher-Reeves修正算法相同,计算量比上面三种算法都小很多
% net.trainFcn = 'trainbfg'; % Quasi-Newton Algorithms - BFGS Algorithm,计算量和内存需求均比共轭梯度算法大,但收敛比较快
% net.trainFcn = 'trainoss'; % One Step Secant Algorithm,计算量和内存需求均比BFGS算法小,比共轭梯度算法略大
%
% (中型网络的首选算法)
%net.trainFcn = 'trainlm'; % Levenberg-Marquardt算法,内存需求最大,收敛速度最快
% net.trainFcn = 'trainbr'; % 贝叶斯正则化算法
%设置随机种子
%setdemorandstream(pi)
%net= newff( minmax(source) , 100, { 'logsig'  }  ) ; 
%% 新建BP神经网络，并设置参数 
hiddenNet=100;
net = feedforwardnet(hiddenNet);
net.layers{1}.initFcn = 'initwb'; 
% net.layers{2}.initFcn = 'initwb';
net.layers{1}.transferFcn='logsig';
% net.layers{2}.transferFcn='logsig';
%使用含有一层隐含层的模型，隐含层有10个神经元。调用Matlab的patternnet函数
%patternnet函数的参数有（hiddenSizes，trainFcn，performFcn）三个。hiddenSizes默认值是10
%可以用数组表示多个隐含层。trainFcn默认值是‘trainscg’，performFcn默认是‘crossentropy’。
%如果想要有两个隐含层，每层的神经元都是10个，则可以写成net = patternnet（[10,10]） ;
%net = patternnet(500);
net.trainParam.epochs=1000; %最大训练次数（缺省值为10）
net.trainParam.show=25;%显示训练迭代过程（NaN表示不显示，缺省为25），每25次显示一次
net.trainParam.showCommandLine=0;%显示命令行（默认值是0） 0表示不显示
net.trainParam.showWindow=1; %显示GUI（默认值是1） 1表示显示
net.trainParam.goal=0;%训练要求精度（缺省为0）
net.trainParam.time=inf;%最大训练时间（缺省为inf）
net.trainParam.min_grad=1e-6;%最小梯度要求（缺省为1e-10）
net.trainParam.max_fail=5;%最大失败次数（缺省为5）
net.performFcn='mse';%性能函数
net.trainFcn = 'trainscg';
net.trainParam.lr = 0.1;%学习速率
net.trainParam.mc = 0.9;%momentum
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
%net = init(net); 
% 训练神经网络模型
net= train(net,source,Target_N);
%netPath = ['net\net_' dataset '_' num2str(hiddenNet) '.mat'];
%save(netPath,'net');
%view(net);
disp('BP神经网络训练完成！');
%% 使用训练好的BP神经网络进行预测
%test=importdata('test.txt');
%test=test';
% y=net(test);
y=sim(net,test);
%mse(y, T1)
%y2= sim(net,test);
%plotconfusion(target,y);
disp('预测完成！');