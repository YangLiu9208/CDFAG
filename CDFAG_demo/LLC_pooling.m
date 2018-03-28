% ========================================================================
% Pooling the llc codes to form the image feature
% USAGE: [beta] = LLC_pooling(feaSet, B, pyramid, knn)
% Inputs
%       feaSet      -the coordinated local descriptors
%       B           -the codebook for llc coding
%       pyramid     -the spatial pyramid structure
%       knn         -the number of neighbors for llc coding
% Outputs
%       beta        -the output image feature
%
% Written by Jianchao Yang @ IFP UIUC
% May, 2010
% ========================================================================

function [beta] = LLC_pooling(feaSet, B, knn)

dSize = size(B, 2);
nSmp = size(feaSet.feaArr, 2);

idxBin = zeros(nSmp, 1);

% llc coding
llc_codes = LLC_coding_appr(B', feaSet.feaArr', knn);
llc_codes = llc_codes';


sidxBin=1:nSmp;
sc_codes = abs(llc_codes);
beta = max(sc_codes(:, sidxBin), [], 2);
beta = beta./sqrt(sum(beta.^2));

