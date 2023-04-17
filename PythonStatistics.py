# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:45:46 2022
Doing Statistics using Python 

@author: https://www.youtube.com/@easydatascience2508
"""

### Lecture 1. Mean, Median, Mode


## list
L1 = [10,8,6,4,5,3,7,2,9,1,8,4,34,32,4,9,8,6,0,3,7,8]
def Average(lst):
    return sum(lst) / len(lst)

ave = Average(L1)      #mean
ave


L1.sort()
mid = len(L1) // 2
res = (L1[mid] + L1[-mid-1]) / 2      #median
res
 

from statistics import mode
mode(L1)      #mode





##numpy ndarray
import numpy as np

ages = np.random.randint(18, high=90, size=500)
np.mean(ages)      #mean

np.median(ages)    #median

from scipy import stats
stats.mode(ages)      #mode





## pandas series

import pandas as pd
import numpy as np

s = pd.Series([12, -4, 7, 9, 9], index=['a', 'b', 'c', 'd', 'e'])
s
  
s.mean()     #mean
s.median()   #median
s.mode()







## pandas DataFrame

df = pd.DataFrame({"A":[12, 5, 5, 44, 1],
                "B":[5, 2, 54, 3, 2],
                "C":[20, 16, 7, 16, 8],
                "D":[14, 2, 17, 2, 6]})
  
df

#mean of each column 
df.mean(axis = 0)

#mean of each row
df.mean(axis = 1)

#mean of specific column
df['A'].mean()

#mean of a specific row
df.loc[1].mean()

#median of each column 
df.median(axis = 0)

#median of each row
df.median(axis = 1)


#median of specific column
df['A'].median()

#median of a specific row
df.loc[1].median()

#mode of each column
df.mode(axis=0)

#mode of each row
df.mode(axis=1)

#mode of specific column
df['A'].mode()

#mean of a specific row
df.loc[1].mode()