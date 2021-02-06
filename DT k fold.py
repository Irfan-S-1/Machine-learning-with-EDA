import pandas as pd
data1=pd.read_csv("D:\\Data Science\\Assignment files\\Decision Tree Assignments\\Fraud_check (1).csv")

def undergrad_num(x):
    if x=="YES":
        return 1
    if x=="NO":
        return 0
    
data1['Undergrad_num']=data1["Undergrad"].apply(undergrad_num)

def martialstatus_num(x):
    if x=="Married":
        return 2
    if x=="Single":
        return 1
    if x=="Divorced":
        return 2

data1['martialstatus_num']=data1["Marital.Status"].apply(martialstatus_num)

def urban_num(x):
    if x=="YES":
        return 1
    if x=="NO":
        return 0

data1['urban_num']=data1["Urban"].apply(urban_num)

data1=data1.drop(data1.columns[[0,1,5]],axis=1)

def tax_num(x):
    if x >= 30000:
        return 0
    if x <= 30000:
        return 1

data1['tax_num']=data1["Taxable.Income"].apply(tax_num)

data1=data1.drop(data1.columns[[0]],axis=1)
X=data1.iloc[:,0:5]
Y=data1.iloc[:,5]
data1.head()
#Decision tree with using no random state#
import numpy as np
from sklearn.model_selection import train_test_split
train,test=train_test_split(data1,test_size=0.2)
train_x=train.iloc[:,0:5]
train_y=train.iloc[:,5]
test_x=test.iloc[:,0:5]
test_y=test.iloc[:,5]


from sklearn.tree import  DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train_x,train_y)

preds=model.predict(test_x)
pd.Series(preds).value_counts()
pd.crosstab(test_y,preds)
np.mean(preds==test_y)#Accuracy=61.66



#Decision tree with using  random state=100 #
train,test=train_test_split(data1,test_size=0.2,random_state=100)
train_x=train.iloc[:,0:5]
train_y=train.iloc[:,5]
test_x=test.iloc[:,0:5]
test_y=test.iloc[:,5]


from sklearn.tree import  DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train_x,train_y)

preds=model.predict(test_x)
pd.Series(preds).value_counts()
pd.crosstab(test_y,preds)
np.mean(preds==test_y)#Accuracy=56.66


#Decision tree with using  random state=200 #
train,test=train_test_split(data1,test_size=0.2,random_state=200)
train_x=train.iloc[:,0:5]
train_y=train.iloc[:,5]
test_x=test.iloc[:,0:5]
test_y=test.iloc[:,5]


from sklearn.tree import  DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train_x,train_y)

preds=model.predict(test_x)
pd.Series(preds).value_counts()
pd.crosstab(test_y,preds)
np.mean(preds==test_y)#Accuracy=58.33

#By using K Folds#
from sklearn.model_selection import cross_val_score
score=cross_val_score(model,X,Y,cv=10)
score
score.mean()#Accuracy=64.66





