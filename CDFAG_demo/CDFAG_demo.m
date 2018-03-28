%---------------Proposed Method-----------------%
clc;
clear;
clear;clc;
close all; 
warning off;
addpath('D:\libsvm');
addpath('KEMA_routine');
% parameter setting
cate_names={'fight','walk','wave1','wave2','handclapping','handshake','hug','jog','jump','punch','push','skip'};
all_acc=[];
all_mean_acc=[];
all_std_acc=[];
for aa=1:1
nRounds=5;                                    
N_source_set=[40];
N_target_set=[25];
set_tr_num = 30;  % target training number per category   
set_tr_num2 =N_source_set(aa)+5;  % source training number per category  
para.img_dir = 'G:\InfAR IDT\';          % directory for dataset 
para.data_dir = 'data\';                  % directory to save the features of the chosen dataset
para.target_dataSet = 'tra_hof_mbh_target';      % target domain folder_name
para.source_dataSet = 'XD_800_tra_hof_mbh_source'; 
para.skip_dic_training = 0;
para.numClusters=4000;
para.nsmp=120000;
para.numClusters2=4000;
para.nsmp2=120000;
para.knn = 5;  
if ~exist('target_fea.mat')
[target_database,source_database ]= compute_feature(para);
[B1,B2] = retrieve_dictionary(target_database,source_database ,para);
[target_fea,target_label] = get_global(target_database,B1,para);
[source_fea,source_label] = get_global(source_database,B2,para);
save('target_fea','target_fea');
save('source_fea','source_fea');
save('target_label','target_label');
save('source_label','source_label');
else
    load target_fea;load target_label;
    load source_fea;load source_label;
end

for ii = 1:nRounds
fprintf('\nRound: %d\n', ii);
%-------------------------------------------------%
%pca降维
[pc, score, latent,tsquare]=pca(target_fea');
K1=cumsum(latent)./sum(latent);
Pca_k1 = find( K1 >= 0.99);
x0=bsxfun(@minus, target_fea', mean(target_fea',1));
[pc2, score2, latent2,tsquare2]=pca(source_fea');
K2=cumsum(latent2)./sum(latent2);
Pca_k2 = find( K2 >= 0.99);
x1=bsxfun(@minus, source_fea', mean(source_fea',1));
Pca_dimension=max(Pca_k1(1),Pca_k2(1));
%Pca_dimension=2;
Ureduce=pc(:,1:Pca_dimension);
Ureduce2=pc2(:,1:Pca_dimension);
PCA_target=x0*Ureduce;
PCA_source = x1*Ureduce2;
PCA_target=PCA_target';
PCA_source=PCA_source';

[tr_idx, ts_idx, tr_idx2,ts_idx2]=train_test_split(set_tr_num,set_tr_num2,source_label,target_label);
training.PCA_source=PCA_source(:, tr_idx2);  
training.source_label =source_label(tr_idx2);

testing.PCA_source = PCA_source (:, ts_idx2);
testing.source_label = source_label(ts_idx2);

training.PCA_target =PCA_target(:, tr_idx);
training.target_label =target_label(tr_idx);

testing.PCA_target = PCA_target(:, ts_idx);
testing.target_label = target_label(ts_idx);

training.PCA_target=training.PCA_target';
training.PCA_source=training.PCA_source';
testing.PCA_target=testing.PCA_target';
testing.PCA_source=testing.PCA_source';
%-----------------------DA---------------------------%
% Xs=pc2(:,1:Pca_dimension);
% Xt=pc(:,1:Pca_dimension);
% Xa=Xs * Xs'*Xt;
% Target_Aligned_SourceData =x1*Xa;
% Target_Projected_Data =x0* Xt;
% Target_Aligned_SourceData=Target_Aligned_SourceData';
% Target_Projected_Data=Target_Projected_Data';
% 
% DA_target_testing_set = Target_Projected_Data(:, ts_idx);
% DA_source_training_set = Target_Aligned_SourceData (:, tr_idx2);
% DA_source_testing_set = Target_Aligned_SourceData (:, ts_idx2);
% DA_target_training_set = Target_Projected_Data(:, tr_idx);

%---------------------KEMA---------------------------%
        Xtemp1=training.PCA_source;
        Xtemp2=training.PCA_target;
%         Xtemp1=DA_source_training_set';
%         Xtemp2=DA_target_training_set' ;

        Ytemp1=training.source_label;
        Ytemp2=training.target_label;
        XT1=testing.PCA_source';
        XT2=testing.PCA_target';

%         XT1=DA_source_testing_set;
%         XT2=DA_target_testing_set;
        YT1=testing.source_label;
        YT2=testing.target_label;
        NF =500;    %特征维数
        N_source=N_source_set(aa);        %每类训练样本数目
        N_target=N_target_set(aa); 
        mu = 0.5;    %(1-mu)*L  + mu*(Ls)
%         [X1_F,X2_F,X1_TF,X2_TF,Y1,Y2,YT1,YT2]=KEMA(X1,X2,Y1,Y2,NF,N,mu);
        [X1_F,X2_F,X1_TF,X2_TF,Y1,Y2,YT1,YT2]=KEMA(XT1,XT2,Xtemp1,Xtemp2,Ytemp1,Ytemp2,YT1,YT2,NF,N_source,N_target,mu);
        hidden_size=size(X2_TF,1);
%---------------------------Many-to-One Encoder--------------------%
% [training.norm_PCA_source,ts] = mapminmax(X1_F(:,1:12*N_source)',0,1);
%  [training.norm_PCA_target,ps] = mapminmax( X2_F(:,1:12* N_target)',0,1);
%  [norm_PCA_target_testing_set,tts] = mapminmax( X2_TF',0,1);
training.norm_PCA_source=X1_F(:,1:12*N_source)';
training.norm_PCA_target=X2_F(:,1:12* N_target)';
norm_PCA_target_testing_set=X2_TF';
norm_PCA_source_testing_set=X1_TF';
training.source_label=Y1;
training.target_label=Y2;
%-------------------------------------------------%
[T,Target_N1,Target_N2]=Target_Output_Generation(training);%Target_N1为target domain,Target_N2为source domain;
[y1,y2,net1]=KEMA_Encoder(training.norm_PCA_target',Target_N1, norm_PCA_target_testing_set',training.norm_PCA_target',hidden_size);
%   y1=mapminmax('reverse',y1',tts);
%   y2 = mapminmax('reverse',y2',ts);
y1=y1';
y2=y2';
[y3,y4,net2]=KEMA_Encoder(training.norm_PCA_source',Target_N2,training.norm_PCA_source',norm_PCA_source_testing_set',hidden_size);
y3=y3';
y4=y4';   
%--------------------------Plot-----------------------------------%
%             Phi1toF=y3';
%             Phi2toF=y2';
%             Phi1TtoF=y4';
%             Phi2TtoF=y1';
%             t1 = size(Phi1TtoF,2)/2;  
%             t2 = size( Phi2TtoF,2)/2; 
%             m1 = mean(Phi1toF');
%             m2 = mean(Phi2toF');
%             s1 = std(Phi1toF');
%             s2 = std(Phi2toF');
%             
%             Phi1toF = zscore(Phi1toF')';
%             Phi2toF = zscore(Phi2toF')';          
%             Phi1TtoF = ((Phi1TtoF' - repmat(m1,2*t1,1))./ repmat(s1,2*t1,1))';
%             Phi2TtoF = ((Phi2TtoF' - repmat(m2,2*t2,1))./ repmat(s2,2*t2,1))';
%           
%  figure;
%  subplot(1,2,1);
%  scatter(Phi1TtoF(1,:),Phi1TtoF(2,:),20,YT1,'f'), hold on, scatter(Phi2TtoF(1,:),Phi2TtoF(2,:),20,YT2),colormap(jet)
%  title('Encoder_Projected data (RBF)'),grid on;
%   axis([-2.5 2.5 -2.5 2.5]);
%  subplot(1,2,2);
%  plot(Phi1TtoF(1,:),Phi1TtoF(2,:),'r.'), hold on, plot(Phi2TtoF(1,:),Phi2TtoF(2,:),'.'),colormap(jet)
%  title('Encoder_Projected data (RBF, domains)'),grid on;
%   axis([-2.5 2.5 -2.5 2.5]); 
%----------------------------SVM training---------------------------%
%%% RBF-Kernel Optimal Parameters
y=[y3;y2];
y_label=[Y1;Y2];
[bestacc,bestc,bestg] = SVMcgForClass(y_label,y);
%[bestacc,bestc,bestg] = SVMcgForClass(target_training_label, training.PCA_target);
rbf_C  =bestc;	%1		 
rbf_g  =bestg;	%3.0314

%%% Set the option string for svm-train
% -t 2   :   Use the RBF Kernel
% -q     :   Quiet mode
% -c     :   Set scaling/penalty term
% -g     :   Set \gamma term
svmtrain_opts = sprintf('-s 0 -t 2 -c %0.5f -g %0.5f -q',rbf_C,rbf_g);
%svmtrain_opts = sprintf('-s 0 -t 0 -c %0.5f',rbf_C);
%%% Run Training
fprintf('Training SVM model...');
tic
model_rbf = svmtrain(y_label,y,svmtrain_opts);
%model_rbf = svmtrain(target_training_label, training.PCA_target,svmtrain_opts);
train_time = toc;
fprintf('done.\n');

%%% Run Testing
fprintf('Testing SVM model...');
tic
[predicted_labels, accuracy,toss] = svmpredict(YT2,y1,model_rbf);
%[predicted_labels, accuracy,toss] = svmpredict(target_testing_label, PCA_target_testing_set,model_rbf);
test_time = toc;
fprintf('done.\n');
confm = confusionmat(YT2,predicted_labels);
plotConfusion(cate_names, confm);
%%% Reporting final performance
fprintf('--------------------------\n');
fprintf('Training Time : %0.2e sec.\n',train_time);
fprintf('Testing Time  : %0.2e sec.\n',test_time);
fprintf('Accuracy      : %0.2f %%.\n' ,accuracy(1));
fprintf('--------------------------\n');
acc(ii) = accuracy(1);
end
all_acc=[all_acc; acc];
all_mean_acc=[all_mean_acc;mean(acc)];
all_std_acc=[all_std_acc;std(acc)];
fprintf('\nMean accuracy: %f\n', mean(acc));
fprintf('Standard deviation: %f\n', std(acc));
end