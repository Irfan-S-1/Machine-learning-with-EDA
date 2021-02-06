
import pandas as pd
import seaborn as sb
import numpy as np 
from sklearn.linear_model import LogisticRegression

data=pd.read_csv("D:\\Data Science\\Assignment files\\Logistic Regression\\creditcard.csv")
data1=pd.read_csv("D:\\Data Science\\Assignment files\\Logistic Regression\\creditcard.csv")
data.head()
data.columns

def owner_num(x):
    if x=="yes":
        return 1
    if x=="no":
        return 0

data["owner_num"]=data["owner"].apply(owner_num)

def selfemp_num(x):
    if x=="no":
        return 1
    if x=="yes":
        return 0

data["selfemp_num"]=data["selfemp"].apply(selfemp_num)

data=data.drop(["card","owner","selfemp"],axis=1)
data=data.drop(data.columns[0],axis=1)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(data)
scaled_feat=scaler.transform(data)
df_feat=pd.DataFrame(scaled_feat,columns=data.columns[:])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df_feat,data1["card"],test_size=0.3)

model=LogisticRegression()
model.fit(X_train,Y_train)

pred=model.predict(X_test)
pd.Series(pred).value_counts()
np.mean(Y_test==pred) #Accuracy=92.29


#Using Kfold cross validation#

X=df_feat.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]]
Y=data1.iloc[:,[1]]

from sklearn.model_selection import cross_val_score
score=cross_val_score(model,X,Y,cv=5)
score
score.mean() #accuracy=95.37
