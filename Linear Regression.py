
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("D:\\Data Science\\Assignment files\\Multiple Linear regression\\Cars.csv")
data.head(10)
data.columns

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(data)
scaler_feat=scaler.transform(data)
df_feat=pd.DataFrame(scaler_feat,columns=data.columns[:])

df_feat.corr()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
X=data.iloc[:,[1]]
Y=data.iloc[:,[0,1,2,3,4]]
regressor=LinearRegression()
score=cross_val_score(regressor,X,Y,scoring="mean_squared_error",cv=5)
score
mse=np.mean(score) 
print(mse)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1,5,10,20,30,40,50,70,80,90,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring="neg_mean_squared_error",cv=5)
ridge_regressor.fit(X,Y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)



from sklearn.linear_model import Lasso

lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1,5,10,20,30,40,50,70,80,90,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring="neg_mean_squared_error",cv=5)
lasso_regressor.fit(X,Y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


