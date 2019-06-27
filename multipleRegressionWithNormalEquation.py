#It is implemented using Normal eq instead of gradient descent algo

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#dummy encoding (making seprate columns for seprate categories )
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


#Splitting the dataset into training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Multivariant linear regression using Normal equation

theta = [0,0,0,0,0,0] #values of coffiecient of independent variable

#Making X[i][0] = 1 so that we get genral form like h = theta0+ theta1*x + theta2*x ... after multiplying theta and X_train matrices
i = 0
j = 0
while(i<40):
    X_train[i][j] = 1
    if i<10:
        X_test[i][j] = 1
    i += 1

#Applying normal eq on training matrix
from numpy.linalg import inv #to inverse matrix

theta = np.matmul(np.matmul(inv(np.matmul(X_train.transpose(), X_train)),X_train.transpose()),Y_train)

theta = np.reshape(theta,(6,1)) #reshaping theta to make it a matrix

#Predicting values of Y based on theta
Y_pred = np.matmul(theta.transpose(),X_test.transpose())
Y_pred = Y_pred.transpose()

error = 0
for (i,j) in zip(Y_pred,Y_test):
    error = error + abs(i-j)*100/j
error = error/10 #average error value

print("Error is:",error,"%")
    
