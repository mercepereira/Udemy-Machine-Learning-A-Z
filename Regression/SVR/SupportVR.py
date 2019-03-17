
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
Xd=dataset.iloc[:, 1:2].values
yd=dataset.iloc[:, 2].values


"""from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)"""

# feature scaling
# SVR library does not feature scaling automatically
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X=sc_X.fit_transform(Xd)
# fit_transform method requires a matrix; array(n_samples, n_features)
# reshape(-1,1) matrix of (N,1)
# reshape (1,-1) matrix of (1,N)
y=sc_Y.fit_transform(yd.reshape(-1,1)).reshape(1,-1)
# get the vector of N y_observations
y=y[0]


#fitting  SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#predict a new result with SVR
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
# inverse transform requires as well a matrix
y_pred =sc_Y.inverse_transform(y_pred.reshape(-1,1))


#visualising thre linear rehgression result
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth of Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#for higer curve
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth of Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

