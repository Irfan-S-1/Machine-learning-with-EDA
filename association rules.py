
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

data=pd.read_csv("D:\\Data Science\\Assignment files\\Association Rules Assignment\\transactions_retail1.csv",header=None)
num_records=len(data)
data.describe()

#Imputation of NULL values#
data.isnull().sum()
data.shape
data.dropna().shape

data[1].fillna("HEART",inplace=True)
data[5].fillna("SET",inplace=True)
data[4].fillna("SET",inplace=True)
data[3].fillna("OF",inplace=True)
data[2].fillna("RED",inplace=True)

records=[]

for i in range(557):
    
    records.append([str(data.values[i,j]) for j in range (0,5)])
    
all_data_list= [i for item in records for i in item]

from collections import Counter
item_frequencies=Counter(all_data_list)

item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])

records_series=pd.DataFrame(pd.Series(records))
records_series.columns=["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = records_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

frequent_itemsets = apriori(X,min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.shape

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.shape

rules.head(10)
rules.sort_values('lift',ascending = False,inplace=True)
