function Subspace_Alignment_Demo()

addpath('/users/visics/bfernand/Documents/Projects/libsvm/libsvm-3.16-copy/matlab');
load('data/amazon_SURF_L10.mat');
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
fts = zscore(fts);  
Amazon_Data =fts;
Amazon_lbl = labels;

load('data/Caltech10_SURF_L10.mat');
fts = fts ./ repmat(sum(fts,2),1,size(fts,2));  
fts = zscore(fts);  
Caltech10_Data =fts;
Caltech10_lbl = labels;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subspace_dim_d = 80;
% number of training samples in source per class
number_of_trn_samples = 20;

Source = Amazon_Data;
Source_lbl = Amazon_lbl;
Target = Caltech10_Data;
Target_lbl = Caltech10_lbl;

[Xss,~,~] = princomp(Source);
[Xtt,~,~] = princomp(Target);
Xs = Xss(:,1:subspace_dim_d);
Xt = Xtt(:,1:subspace_dim_d);



for iter = 1 : 20
    [trainset, trainlabels] = generateTrainSplits(Source_lbl,Source,number_of_trn_samples);
    [accuracy_na_nn(iter),accuracy_sa_nn(iter),accuracy_na_svm(iter),accuracy_sa_svm(iter)] = Subspace_Alignment(trainset,Target,trainlabels,Target_lbl,Xs,Xt);
end
clc;
fprintf('NN No adaptation Accuacry \t %1.2f \n',mean(accuracy_na_nn));
fprintf('NN SA Accuacry \t %1.2f \n',mean(accuracy_sa_nn));
fprintf('SVM No adaptation Accuacry \t %1.2f \n',mean(accuracy_na_svm));
fprintf('SVM SA Accuacry \t %1.2f \n',mean(accuracy_sa_svm));


end


function [trainset trainlabels] = generateTrainSplits(original_train_lbl,original_trainset,num_trn_lbls)
    classes = unique(original_train_lbl);
    im_index = [];
    trainlabels = [];        
    for class_index =  1 : length(classes)         
        all =  find(original_train_lbl == classes(class_index));
        rn = randperm(length(all));		
		if length(rn) > num_trn_lbls
			all = all(rn(1:num_trn_lbls));
		end
        im_index = [im_index ; all];
		if length(rn) > num_trn_lbls
			trainlabels = [trainlabels ; (zeros(num_trn_lbls,1) + classes(class_index))];   
		else
			trainlabels = [trainlabels ; (zeros(length(rn),1) + classes(class_index))]; 
		end		
    end
    trainset = original_trainset(im_index,:);          

end