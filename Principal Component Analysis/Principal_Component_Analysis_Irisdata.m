clear all; close all; clc;
data = dlmread('iris_dataset.dat');
data =  data(randperm(end),:);
N = 150;  % total number of samples
NC = 50;  % size of each class
K = 10;  % K-fold
d=1;   % Here, d= Number of features 
% Randomly shuffle data
%seed = 150; rand('seed', seed);
%index = randperm(N);
%data_shuffled = data(index,:);

index = randperm(N); 
new_data = data(index,[1 5]);%first feature extracted for Bayes algorithm

pcdata = data(index,:); %all features for PCA algorithm

X=pcdata(:,1:4);
m = mean(X);
M = repmat(m,N,1);
XM = X - M;

[eigvec_org eigval_org] = eig(cov(XM));
eigvec = fliplr(eigvec_org);  % largest evector on 1st col
eigval = flipud(diag(eigval_org));  % largest evalue on top
pc = XM * eigvec;

cumulativeAccuracy=0;
fprintf("Step 1: Using only the 1st feature of the original dataset\n");
fprintf("Classification Accuarcy \nans =\n");
% 10-fold cross validation
for k=1:K
   % Following section is used for dividing given data set into test
   % dataset and trainging data set 
   testUpperBound=k*15;
   testLowerBound=testUpperBound-14;
   testData=data(testLowerBound:testUpperBound,:);     % Creating different test data for each iteration
   trainData= data;                                    % Copying original data in training data 
   trainData(testLowerBound:testUpperBound,:)=[];   % Removing test data from original data to create valid training data 
   %Follwoing section is used to identify 3 classes of training dataset
   trainClassA = trainData(trainData(:,5) == 1,:);
   trainClassB = trainData(trainData(:,5) == 2,:);
   trainClassC = trainData(trainData(:,5) == 3,:);
  
   %Following section is used to identify 3 classes of test dataset which will be later used for comparision purpose.
   testClassA = testData(testData(:,5) == 1,:);
   testClassB = testData(testData(:,5) == 2,:);
   testClassC = testData(testData(:,5) == 3,:);
  
   % Follwoing section is used to identify number of rows in each class within training dataset 
   [rowTrainClassA,colClassA]=size(trainClassA);
   [rowTrainClassB,colClassB]=size(trainClassB);
   [rowTrainClassC,colClassC]=size(trainClassC);
  
   % Follwoing section is used to identify number of rows in each class
   % within test dataset 
   rowTestClassA=50-rowTrainClassA;
   rowTestClassB=50-rowTrainClassB;
   rowTestClassC=50-rowTrainClassC;

   %Following section is used for considering first 4 columns of all
   %classes in training dataset
   a=trainClassA(:,1:4);
   b=trainClassB(:,1:4);
   c=trainClassC(:,1:4);
   d=1; 
  

    %Finding mean of each class of training dataset
    mu1=mean(a);
    mu2=mean(b);
    mu3=mean(c);
   
    % Subtracting mu from class elements  
    xminusmu1=a-mu1;                         % Mean matrix of 1
    xminusmu1T=xminusmu1';
    xminusmu2=b-mu2;                         % Mean matrix of 2
    xminusmu2T=xminusmu2';
    xminusmu3=c-mu3;                         % Mean matrix of 3
    xminusmu3T=xminusmu3';
    
    % Finding covariance of each matrix 
    cov1=(1/(rowTrainClassA-1))*(xminusmu1T*xminusmu1);         % Covariance matrix of a
    cov2=(1/(rowTrainClassB-1))*(xminusmu2T*xminusmu2);         % Covariance matrix of b
    cov3=(1/(rowTrainClassC-1))*(xminusmu3T*xminusmu3);         % Covariance matrix of c
    newg1=[];
    newg2=[];
    newg3=[];
    testDataLabel=testData(:,5)';
    resultDataLabel=[];
   
    
    % Finding discriminant function 
    for i=1:15
         testg1= ((-0.5)*sum(sum(((testData(i,1:1)-mu1)*inv(cov(a))*(testData(i,1:1)-mu1)'))))-((d/2)*log(2*pi))-(0.5*(log(det(cov1))))-log(1/3);
         newg1(end+1)=testg1;
         testg2= ((-0.5)*sum(sum(((testData(i,1:4)-mu2)*inv(cov(b))*(testData(i,1:4)-mu2)'))))-((d/2)*log(2*pi))-(0.5*(log(det(cov2))))-log(1/3);
         newg2(end+1)=testg2;
         testg3= ((-0.5)*sum(sum(((testData(i,1:4)-mu3)*inv(cov(c))*(testData(i,1:4)-mu3)'))))-((d/2)*log(2*pi))-(0.5*(log(det(cov3))))-log(1/3);
         newg3(end+1)=testg3;  
    end
    resultclassa=0;
    resultclassb=0;
    resultclassc=0;
  
    %Comparing discriminant function
    for j=1:15
        if(newg1(j))
            resultclassa=resultclassa+1;
        end
        if(newg2(j)>newg1(j) && newg2(j)>newg3(j))
            resultclassb=resultclassb+1;
        end
        if(newg3(j)>newg1(j) && newg3(j)>newg2(j))
            resultclassc=resultclassc+1; 
        end
    end
   accuracy= ((resultclassa/rowTestClassA)+(resultclassb/rowTestClassB)+(resultclassc/rowTestClassC))/10;
   disp(accuracy);
   cumulativeAccuracy=cumulativeAccuracy+accuracy; 
end

averageAccuracy=cumulativeAccuracy/10;           
fprintf('Average Accuracy = %5.4f\n',averageAccuracy);
%disp(averageAccuracy);


fprintf("\nStep 2: Print eigenvalues\n");
fprintf("eigval =\n");
disp(eigval);


fprintf("Step 3: Using only the 1st feature of principal Component\n");
fprintf("Classification Accuarcy \nans =\n");


pcd = [pc(:,1),pcdata(:,5)];

cumulativeAccuracy_pc=0;

for k=1:K
   testUpperBound=k*15;
   testLowerBound=testUpperBound-14;
   pcd=pcdata(testLowerBound:testUpperBound,:);    
   trainData= pcdata;                                    % Copying original data in training data 
   trainData(testLowerBound:testUpperBound,:)=[];   % Removing test data from original data to create valid training data 
   %Follwoing section is used to identify 3 classes of training dataset
   trainClassA = trainData(trainData(:,5) == 1,:);
   trainClassB = trainData(trainData(:,5) == 2,:);
   trainClassC = trainData(trainData(:,5) == 3,:);
  
   %Following section is used to identify 3 classes of test dataset which will be later used for comparision purpose.
   testClassA = pcd(pcd(:,1) == 1,:);
   testClassB = pcd(pcd(:,1) == 2,:);
   testClassC = pcd(pcd(:,1) == 3,:);
  
   % Follwoing section is used to identify number of rows in each class within training dataset 
   [rowTrainClassA,colClassA]=size(trainClassA);
   [rowTrainClassB,colClassB]=size(trainClassB);
   [rowTrainClassC,colClassC]=size(trainClassC);
  
   % Follwoing section is used to identify number of rows in each class
   % within test dataset 
   rowTestClassA=50-rowTrainClassA;
   rowTestClassB=50-rowTrainClassB;
   rowTestClassC=50-rowTrainClassC;

   %Following section is used for considering first 4 columns of all
   %classes in training dataset
   a=trainClassA(:,1:4);
   b=trainClassB(:,1:4);
   c=trainClassC(:,1:4);
   d=1;                        % Here, d= Number of features
  

    %Finding mean of each class of training dataset
    mu1=mean(a);
    mu2=mean(b);
    mu3=mean(c);
   
    % Subtracting mu from class elements  
    xminusmu1=a-mu1;                         % Mean matrix of 1
    xminusmu1T=xminusmu1';
    xminusmu2=b-mu2;                         % Mean matrix of 2
    xminusmu2T=xminusmu2';
    xminusmu3=c-mu3;                         % Mean matrix of 3
    xminusmu3T=xminusmu3';
    
    % Finding covariance of each matrix 
    cov1=(1/(rowTrainClassA-1))*(xminusmu1T*xminusmu1);         % Covariance matrix of a
    cov2=(1/(rowTrainClassB-1))*(xminusmu2T*xminusmu2);         % Covariance matrix of b
    cov3=(1/(rowTrainClassC-1))*(xminusmu3T*xminusmu3);         % Covariance matrix of c
    newg1=[];
    newg2=[];
    newg3=[];
    testDataLabel=pcd(:,5)';
    resultDataLabel=[];
    
  
    % Finding discriminant function 
    for i=1:15
         g1= ((-0.5)*sum(sum(((pcd(i,1:4)-mu1)*inv(cov(a))*(pcd(i,1:4)-mu1)'))))-((d/2)*log(2*pi))-(0.5*(log(det(cov1))))-log(1/3);
         newg1(end+1)=g1;
         g2= ((-0.5)*sum(sum(((pcd(i,1:4)-mu2)*inv(cov(b))*(pcd(i,1:4)-mu2)'))))-((d/2)*log(2*pi))-(0.5*(log(det(cov2))))-log(1/3);
         newg2(end+1)=g2;
         g3= ((-0.5)*sum(sum(((pcd(i,1:4)-mu3)*inv(cov(c))*(pcd(i,1:4)-mu3)'))))-((d/2)*log(2*pi))-(0.5*(log(det(cov3))))-log(1/3);
         newg3(end+1)=g3;  
         
    end
    resultclassa=0;
    resultclassb=0;
    resultclassc=0;
  
    %Comparing discriminant function
    for j=1:15
        if(newg1(j))
            resultclassa=resultclassa+1;
        end
        if(newg2(j)>newg1(j) && newg2(j)>newg3(j))
            resultclassb=resultclassb+1;
        end
        if(newg3(j)>newg1(j) && newg3(j)>newg2(j))
            resultclassc=resultclassc+1; 
        end
    end
    
   accuracy_pc= ((resultclassa/rowTestClassA)+(resultclassb/rowTestClassB)+(resultclassc/rowTestClassC))/10;
   disp(accuracy_pc);
   cumulativeAccuracy_pc=cumulativeAccuracy_pc+accuracy; 
end

averageAccuracy_pc=cumulativeAccuracy_pc/10;           
fprintf('Average Accuracy = %5.4f\n',averageAccuracy_pc);
%disp(averageAccuracy);
   