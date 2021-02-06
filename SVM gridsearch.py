
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

data=pd.read_csv("D:\\Data Science\\Assignment files\\SVM Assignments\\forestfires.csv")
data1=data.drop(["month","day"],axis=1)

def size_num(x):
    if x=='small':
        return 1
    if x=='large':
        return 2
    
data1["size_num"]=data1["size_category"].apply(size_num)
data1=data1.drop(["size_category"],axis=1)

#Standardization of data(Feature Scaling)#
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(data1.drop("size_num",axis=1))
scaler_feat=scaler.transform(data1.drop("size_num",axis=1))
df_feat=pd.DataFrame(scaler_feat,columns=data1.columns[:-1])

df_feat.describe()

#Splitting of data into train and test#
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df_feat,data1["size_num"],test_size=0.3)

#By using kernel=poly#
model=SVC(kernel="poly")
model.fit(X_train,y_train)
pred1=model.predict(X_test)
np.mean(y_test==pred1) #Accuracy=72.43

#By using kernel=rbf#
model=SVC(kernel="rbf")
model.fit(X_train,y_train)
pred2=model.predict(X_test)
np.mean(pred2==y_test) #Accuracy=74.35

#By using kernel=linear#
model=SVC(kernel="linear")
model.fit(X_train,y_train)
pred1=model.predict(X_test)
np.mean(y_test==pred1) #Accuracy=86.53

#By using GridSearcg CV to find best kernel for better accuracy#
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,10,100,1000],'kernel':['linear']},{'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9]}]
grid_search=GridSearchCV(estimator=model,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)

grid=grid_search.fit(X_train,y_train)
accuracy=grid.best_score_
accuracy

#To get Best kernel condition#
grid.best_params_ #({'C': 100, 'kernel': 'linear'})#

#By using kernel got by GridSearchCV#
model1=SVC(kernel="linear",C=100)
model1.fit(X_train,y_train)
pred3=model1.predict(X_test)
np.mean(pred3==y_test)#Accuracy=95.51
