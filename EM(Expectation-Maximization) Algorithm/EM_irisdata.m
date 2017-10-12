clear all; close all; 
% X is the dataset size 100x2
% Column 1: sepal length
% Column 2: sepal width
X = dlmread('simple_iris_dataset.dat'); % Size=100x2
N = length(X); % N=100
% Initialization - take 2 random samples from data set
ctr1 = X(randi([1,N]),:);
ctr2 = X(randi([1,N]),:);
cov1 = cov(X);
cov2 = cov(X);
prior1 = 0.5;
prior2 = 0.5;
% Misc. initialization
idx_c1 = zeros(50,1);
idx_c2 = zeros(50,1);
W1 = zeros (100,1);
W2 = zeros (100,1);

u1= X(randi(N),1);
u2= X(randi(N),2);

x=size(X);

% W1 and W2 are vectors that eventually should contain each data point's
% membership grade relative to Gaussian 1 and Gaussian 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You can define some error measure smaller than some epsilon to stop
% the iteration. But for now just run for 250 iterations
u1 = [u1; u1];
u2 = [u2; u2];

for itr = 1:250

    % Perform E step
    for i=1:100
        tempX=(X(i,:))';
        X_M1=tempX-u1;
        X_M2=tempX-u2;
        
        g1=(1/(det(cov1)^0.5)) * exp((-1/2)*(X_M1)'*inv(cov1)*X_M1);
        g2=(1/(det(cov2)^0.5)) * exp((-1/2)*(X_M2)'*inv(cov2)*X_M2);
        
        W1(i,1) =(g1* prior1)/(g1* prior1 + g2* prior2);
        W2(i,1) =(g2* prior1)/(g1* prior1 + g2* prior2);
        
        end
     %Perform M step
     prior1=sum(W1)/N;
     prior2=sum(W2)/N;
     
     u1=[sum([W1, W1].* X)/sum(W1)];
     u2=[sum([W2, W2].* X)/sum(W2)];
     
     u1=u1';
     u2=u2';
     
     u1_X= repmat(u1',N,1);
     u2_X= repmat(u2',N,1);
     
     cov1=transpose([W1,W1].*(X-u1_X))*(X-u1_X)/sum(W1);
     cov2=transpose([W2,W2].*(X-u2_X))*(X-u2_X)/sum(W2);
    

end

%ctr1=[sum([W1,W1].*X)/N];
%ctr2=[sum([W2,W2].*X)/N];

ctr1=u1';
ctr2=u2';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; hold on;
title('Clustering with EM algorithm');
xlabel('Sepal Length');
ylabel('Sepal Width');
% Hard clustering assignment ? W1, W2 (100x1)
idx_c1 = find(W1 > W2);
idx_c2 = find(W1 <= W2);
% idx_c1 is a vector containing the indices of the points in X
% that belong to cluster 1 (Mx1)
% idx_c2 is a vector containing the indices of the points in X
% that belong to cluster 2 (N-M x 1)
% Plot clustered data with two different colors
plot(X(idx_c1,1),X(idx_c1,2),'r.','MarkerSize',12)
plot(X(idx_c2,1),X(idx_c2,2),'b.','MarkerSize',10)
% Plot centroid of each cluster ? ctr1, ctr2 (1x2)
plot(ctr1(:,1),ctr1(:,2), 'kx', 'MarkerSize',12,'LineWidth',2);
plot(ctr2(:,1),ctr2(:,2), 'ko', 'MarkerSize',12,'LineWidth',2);