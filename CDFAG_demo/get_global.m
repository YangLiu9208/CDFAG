function [ sc_fea, sc_label ] = get_global( database,B,para )
dimFea = size(B,2);
numFea = length(database.path);
sc_fea = zeros(dimFea, numFea);
sc_label = zeros(numFea, 1);
disp('==================================================');
fprintf('Calculating the LLC feature...\n');
fprintf('Vocabulary size: %f\n',  para.numClusters);
disp('==================================================');

for iter1 = 1:numFea,  
    if ~mod(iter1, 50),
        fprintf('.\n');
    else
        fprintf('.');
    end;
    fpath = database.path{iter1};
    load(fpath);
    sc_fea(:, iter1) = LLC_pooling(feaSet, B, para.knn);
    %sc_fea(:, iter1) = feaSet.feaArr;
    sc_label(iter1) = database.label(iter1);
end

