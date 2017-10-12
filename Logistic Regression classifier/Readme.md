
Matlab Version : 9.2.0.556344 (R2017a)

# Description of Problem: 
To implement a logistic regression classifier and use it to classify the test examples in the provided (datasetscorrupted_2class_iris_dataset.dat) which is a subset of the corrupted iris dataset used in the previous assignment. It contains only the setosa and versicolor species (called class 1 and class 0, respectively).

# Description of your Solution: 
Logistic_regression_irisdata_classifier.m is program which reads file datasetscorrupted_2class_iris_dataset.dat. This program shuffles data divides data into training data (90% of original data) and test data (10% of original data) in each iteration (total 10 iterations). Based on discriminative functions it classifies test data. Classification Accuracy during each iteration is displayed. At the end of program average accuracy= 0.9500 is also depicted. This program also has an output graph. Initial plotting, Training iteration vs Cost function J Screen shot of command window is also in this folder. Initial learning rate is 0.01 which takes 1500 iterations to converge. By changing learning rate we get different iterations required for convergence.Screenshot of Output is displayed in Logistic_regression_classifier_Output_ScreenShot.png. 




