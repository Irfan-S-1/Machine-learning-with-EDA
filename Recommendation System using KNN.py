import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

books=pd.read_csv("D:\\Data Science\\Assignment files\\Recoomendation system assignments\\book.csv",encoding="latin-1")



rating_count=pd.DataFrame(books.groupby('ID')['Rating'].count())
rating_count['rating mean']=pd.DataFrame(books.groupby('ID')['Rating'].mean())
rating_count.sort_values('Rating',ascending=False).head() #Most rated books 

#To convert ratings table to a pivot table#
rating_pivot=books.pivot(index='Title',columns='Serial No',values='Rating').fillna(0)
print(rating_pivot.shape)
rating_pivot.head()


#Implementing using KNN
from scipy.sparse import csr_matrix

rating_new=csr_matrix(rating_pivot.values)

from sklearn.neighbors import NearestNeighbors

model_knn=NearestNeighbors(metric="cosine",algorithm='brute')
model_knn.fit(rating_new)

query_index=np.random.choice(rating_pivot.shape[0])
print(query_index)

distance,indices=model_knn.kneighbors(rating_pivot.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)

for i in range(0, len(distance.flatten())):
    if i==0:
        print('Recommendations for {0}:\n'.format(rating_pivot.index[query_index]))
    else:
        print('{0}'.format(rating_pivot.index[indices.flatten()[i]], distance.flatten()[i]))
    
                                                
                                                

