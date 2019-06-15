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

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #here we transformed X_test by fitting sc_X to X_train so that both have same scaling

#Multivariant linear regression using gradient descent

theta = [0,0,0,0,0,0] #values of coffiecient of independent variable

#Making X[i][0] = 1 so that we get genral form like h = theta0+ theta1*x + theta2*x ... after multiplying theta and X_train matrices
i = 0
j = 0
while(i<40):
    X_train[i][j] = 1
    if i<10:
        X_test[i][j] = 1
    i += 1

#function to return  derivative of cost function that will be subtracted form each theta respectevely.
def averageCost(temp):
    total = 0
    for (i,j) in zip(X_train,Y_train):
        mul = np.matmul(theta,i)
        total = total + (mul - j)*i[temp]
    return total/40

#calculating cost for each theta
cost = [0,0,0,0,0,0]

def updateCostMatrix():
    s = 0 #it denotes index of respective theta
    while s<len(theta):
        cost[s] = averageCost(s)
        s += 1

def updateTheta():
    i = 0
    while(i<len(theta)):
        theta[i] = theta[i] - 0.1*cost[i]
        i += 1
        
#Here loop stops when cost0 and cost1 becomes approximately to 0
#This loop is gradient descent algorithm
while(True):
    count = 0 #it is used to check that all cost values are zeroes
    updateCostMatrix()
    updateTheta()
    
    for i in cost:
        if(i < 0.5 and i>-0.5):
            count += 1
    
    if(count == 6):
        break

#Calculating error of predicted values.
error = 0 #initialising it.
    
for i,j in zip(X_test,Y_test):
    mul = np.matmul(theta,i)
    error = error + abs((mul - j)*100/j)
error = error/10
print("Error is:",error,"%")

temp = [] #It contanis predicted values of Y_test

for i in X_test:
    temp.append(np.matmul(theta,i))
