import pandas as pd
import numpy as np

data=pd.read_csv("D:\\Data Science\\Assignment files\\Random Forests Assignment\\Company_Data.csv")

data.head()

#Functions for dummy variable#
def sales_num(x):
    if x>9:
        return "Good"
    if x<=9:
        return "Bad"
def us_num(x):
    if x=="Yes":
        return 1
    if x=="No":
        return 0

data["US_Num"]=data["US"].apply(us_num)

def urban_num(x):
    if x=="Yes":
        return 1
    if x=="No":
        return 0
data["Urban_Num"]=data["Urban"].apply(urban_num)


def shelv_num(x):
    if x=="Good":
        return 2
    if x=="Medium":
        return 1
    if x=="Bad":
        return 0
data["ShelvLoc_Num"]=data["ShelveLoc"].apply(shelv_num)   
data["sales_num"]=data["Sales"].apply(sales_num)
data=data.drop("Sales",axis=1)
data=data.drop(data.columns[[5,8,9]],axis=1)

data1["sales_num"]=data1["Sales"].apply(sales_num)
data1=data1.drop("Sales",axis=1)

data=data.drop("sales_num",axis=1)

#Standardization of data(Feature scaling)#
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(data)
scaler_feat=scaler.transform(data)
df_feat=pd.DataFrame(scaler_feat,columns=data.columns[:])

#Splitting data into train and test#
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df_feat,data1["sales_num"],test_size=0.2)

#By using Random forest#
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,Y_train)

pred=rf.predict(X_test)
np.mean(pred==Y_test)#Accuracy=82.5



#By using Random forest with certain conditions#
rf=RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=1000,criterion="entropy")
rf.fit(X_train,Y_train)
pred=rf.predict(X_test)
np.mean(pred==Y_test)#Accuracy=83.75

#By using Bagging Classifier#
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier()

from sklearn.model_selection import cross_val_score
score=cross_val_score(model,X_train,Y_train,cv=10)
score
score.mean() #Accuracy=83.75

#By using AdaBoostClassifier#
from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier()
model.fit(X_train,Y_train)
pred=model.predict(X_test)
np.mean(pred==Y_test)#Accuracy=88.75

#By using GradientBoostingClassifier#
from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(X_train,Y_train)
pred=model.predict(X_test)
np.mean(pred==Y_test)#Accuracy=85.00
