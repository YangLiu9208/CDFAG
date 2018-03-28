%feature normalization
function nx = Normalize(X)
% nx is the normalized X (the design matrix)
% first, set nx as the original design matrix
nx = X;

% mu is the mean of each feature of the design matrix
% use a zero matrix
mu = zeros(1, size(X,2));
mu = mean(X);

% sig is the standard deviation of each feature of the design matrix
% use a zero matrix first
sig = zeros(1, size(X,2));
sig = std(X);

indices = 1:size(X,2); % [1,2,...,m] where m is the feature size

for i = indices,
    XminusMu = X(:, i) - mu(:, i);
    nx(:,i) = XminusMu/sig(:, i);
end
end 