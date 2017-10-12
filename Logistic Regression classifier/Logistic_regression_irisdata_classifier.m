clear all; close all; clc; 

data = dlmread('corrupted_2class_iris_dataset.dat');   
N = 100;  % total number of samples 
NC = 50;  % size of each class 
K = 10;  % K-fold 
d = 4;   % number of features 
nu = 0.01;  % learning rate 
% Randomly shuffle data 
%seed = 150; rand('seed', seed); 

index = randperm(N); 
datap = data(index,:); 

cumulativeAccuracy = 0;

fprintf('Classification accuracy \nans=\n');

% 10-fold cross validation 
for k=0:K-1
  % Separate training data and test data 
  % 90% of the data for training, 
  % 10% of data for testing 
  % TRAINING  
  % Size X = 90x5, the first column = 1 
  % Size y = 90x1 (class 1 and class 0 labels) 
  % Size w = 5x1, initialized randomly, w1 is the bias 
  % Apply Gradient Descent and run for 1500 iterations
  if k == 0
    newX = datap([11:end],:);   
    tdata = datap([1:10],:);
  elseif k == K-1
    newX = datap([1:90],:);
  else
    newX = datap([1:k*10 (k+1)*10+1:end],:);
  end
  
  X = newX(:,[1 2 3 4]);
  X = [repmat(1,length(X),1) X];
  
  Y = newX(:,5);
  
  W = rand(1, 5);
  W = transpose(W);
    
  J = [];  
  
  for itr=1:1500
  
    x = [] ;
    sigm = []; 
  
    Z = X*W ;
  
    for traindata=1:90    % 90% of the data for training, 
        %tmp1 = Z(i,1);
        temp = 1 / (1+exp(-1* Z(traindata,1)));
        sigm = [sigm; temp];
        temp = temp - Y(traindata,:);
        x = [x; temp];      
    end
  
    xx_1 = x'; 
    newW = xx_1 * X;
    newW = newW';
    new = W - nu * (1/(N-K)) * newW; 
  
    Mean = (-1) * (( Y .*log(sigm)) + ((1-Y).*log(1-sigm))); 
    S =sum(Mean);
    J = [J; S];
    W = new;
  
  end
  
  
  i= 0;
  j= 0; 
  
  % TESTING 
  % Compute sigm for each test data and assign label 
  % Check against true response 
  for testdata = 1:10  % 10% of data for testing 
     test_inp = tdata(testdata,[1 2 3 4]); 
     test_inp = [1 test_inp];
     P= tdata(testdata, end);
     predict = 0 ;
     
     if(  1/(1+exp(-1* test_inp*W)) > 0.5 )
         predict = 1;
     end
     
     
     if(P == predict) 
         i = i + 1;
     else 
         j = j +1;
     end
         
  end
  
  fprintf('%5.4f \n', i/10);
  cumulativeAccuracy = cumulativeAccuracy + i/10;
  
end 
% Evaluate classification accuracy  
%   Accuracy per iteration = no of correct classification / 10 
%   Average accuracy for all 10-fold CV 
fprintf('\nAverage accuracy = %5.4f\n', cumulativeAccuracy/10) %generates nice format 

% This will display provided learning rate
fprintf('\nLearning rate :');disp(nu);  
fprintf('Number of iteration are required :'); 
disp(itr); % This will display required learning rate 


% Plot cost function vs training iterations 
plot (J);    % Size J = 1500x1 
xlabel('Training iterations'); 
ylabel('Cost function J'); 