import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

salarydata_train=pd.read_csv("D:\\Data Science\\Assignment files\\Naive Baiyes Assignment\\SalaryData_Train.csv")
salarydata_test=pd.read_csv("D:\\Data Science\\Assignment files\\Naive Baiyes Assignment\\SalaryData_Test.csv")

#EDA#
#Normalizing data#
from sklearn import preprocessing
number = preprocessing.LabelEncoder()

string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

for i in string_columns:
    salarydata_train[i] = number.fit_transform(salarydata_train[i])
    salarydata_test[i] = number.fit_transform(salarydata_test[i])

salarydata_train.describe()

#Pairplot of train data#
import seaborn as sns
sns.pairplot(salarydata_train,palette="Type")
plt.boxplot(salarydata_train["workclass"])
plt.boxplot(salarydata_train["education"])
plt.boxplot(salarydata_train["occupation"])
plt.boxplot(salarydata_train["relationship"])
#Splitting of Xtrain,Ytrain and Xtest,Ytest data#    
colnames1 = salarydata_train.columns  
colnames2=salarydata_test.columns 
trainX = salarydata_train[colnames1[0:13]]
trainY = salarydata_train[colnames1[13]]
testX  = salarydata_test[colnames2[0:13]]
testY  = salarydata_test[colnames2[13]]

#Making models of naive bayes (By using Gaussian Naive bayes)#
sgnb = GaussianNB()

pred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,pred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) #Accuracy=79.46


#Making models of naive bayes (By using Multinomial Naive bayes)#
smnb = MultinomialNB()
pred_mnb=smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,pred_mnb)
np.mean(testY==pred_mnb)#Accuracy 77.49

#Making models of naive bayes (By using Bernoulli Naive bayes)#
brnb=BernoulliNB()
pred_brnb=brnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,pred_brnb)
np.mean(testY==pred_brnb)#Accuracy=72.84
