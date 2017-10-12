%%% Bayes Decision Theoretic %%%
% MLE Covariance matrix 

function [cov] = covmle(tr_data, u);

X = tr_data;
u = u';
n = length(X);
M = repmat(u,n,1);

cov = (X-M)' * (X-M) / n;

