function [T,Target_N1,Target_N2]=Target_Output_Generation(training)
source=training.norm_PCA_source';
% source=training.source;
target=training.norm_PCA_target';
% target=training.target;
source_label=training.source_label;
target_label= training.target_label;
clabel = unique(target_label);
nclass = length(clabel);
T=[];
Target_N1=[];
Target_N2=[];
for jj = 1:nclass
        idx_label = find( target_label == clabel(jj));
        idx_label2 = find( source_label == clabel(jj));
        num = length(idx_label); 
        num2 = length(idx_label2); 
        temp=(sum(source(:,idx_label2),2)+sum(target(:,idx_label),2))/(num+num2);
        T=[T,temp];
        Target_N1=[Target_N1 repmat(temp,1,num) ];
        Target_N2=[Target_N2 repmat(temp,1,num2)];
end