
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("D:\\Data Science\\Assignment files\\KNN Assignment\\glass.csv")
df.head()

#Standardization of the data#
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df.drop('Type',axis=1))
scaled_feat=scaler.transform(df.drop("Type",axis=1))

df_feat= pd.DataFrame (scaled_feat,columns=df.columns[:-1])
df_feat.head()

import seaborn as sns
sns.pairplot(df,hue="Type")

#Splitting data into train and test#
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(df_feat,df['Type'],test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier as KNC
#Selecting value of K randomly to 3
model=KNC(n_neighbors=3) 

model.fit(X_train,Y_train)

pred=model.predict(X_test)
pd.Series(pred).value_counts()
np.mean(pred==Y_test) #Accuracy=62.79

#Selecting value of K randomly to 5
model=KNC(n_neighbors=5) 

model.fit(X_train,Y_train)

pred=model.predict(X_test)
pd.Series(pred).value_counts()
np.mean(pred==Y_test)#Accuracy=60.45

#Selecting proper K Value using Cross Validation#
from sklearn.model_selection import cross_val_score
accuracy_rate=[]

for i in range(1,40):
    model=KNC(n_neighbors=i)
    score=cross_val_score(model,df_feat,df["Type"],cv=10)
    accuracy_rate.append(score.mean())
    
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color="blue",linestyle="dashed",marker="o",markerfacecolor="red",markersize="10")
plt.title("K Value vs Accuracy rate")
plt.xlabel("K")
plt.ylabel("Accuracy Rate")

#according to the Plot K=2 will give maximum Accuracy#
model=KNC(n_neighbors=2) 

model.fit(X_train,Y_train)

pred=model.predict(X_test)
pd.Series(pred).value_counts()
np.mean(pred==Y_test)#accuracy=65.11
