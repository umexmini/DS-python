#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get dataset
dataset = pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y= dataset.iloc[:,4].values

# Ecoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
LabelEncoder_X=LabelEncoder()
X[:,3]=LabelEncoder_X.fit_transform(X[:,3])
OneHotEncoder=OneHotEncoder(categorical_features=[3])
X=OneHotEncoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X=X[:,1:]
#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#fitting MLR to Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the Test set results
Y_predict=regressor.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X ,axis=1)

X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS= sm.OLS(endog=Y,exog=X_opt).fit()
c=regressor_OLS.summary()
print(c)

X_opt=X[:,[0,1,3,4,5]]
regressor_OLS= sm.OLS(endog=Y,exog=X_opt).fit()
c=regressor_OLS.summary()
print(c)

X_opt=X[:,[0,3,4,5]]
regressor_OLS= sm.OLS(endog=Y,exog=X_opt).fit()
c=regressor_OLS.summary()
print(c)

X_opt=X[:,[0,3,5]]
regressor_OLS= sm.OLS(endog=Y,exog=X_opt).fit()
c=regressor_OLS.summary()
print(c)

X_opt=X[:,[0,3]]
regressor_OLS= sm.OLS(endog=Y,exog=X_opt).fit()
c=regressor_OLS.summary()
print(c)
