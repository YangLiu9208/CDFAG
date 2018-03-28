function [beta] = VLAD_Encoding(feaSet, B,para)
nSmp = size(feaSet.feaArr, 2);
kdtree = vl_kdtreebuild(B) ;
nn = vl_kdtreequery(kdtree,B,feaSet.feaArr) ;
assignments = zeros(para.numClusters,size(feaSet.feaArr,2));% numDataToBeEncoded
assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
VLAD_codes = vl_vlad(feaSet.feaArr,B,assignments);
%sidxBin=1:nSmp;
%sc_codes = abs(VLAD_codes);
%beta = max(sc_codes(:, sidxBin), [], 2);
beta = VLAD_codes./sqrt(sum(VLAD_codes.^2));
end