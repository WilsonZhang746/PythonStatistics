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

















### Lecture 2. Binomial distribution

#The scipy.stats module contains various functions for statistical 
#calculations and tests. The stats() function of the 
#scipy.stats.binom module can be used to calculate a binomial
# distribution using the values of n and p.

#Syntax : 
#scipy.stats.binom.stats(n, p)
#It returns a tuple containing the mean and variance of the 
# distribution in that order.

#example: Calculating distribution table :

#Define n and p.
#Define a list of values of r from 0 to n.
#Get mean and variance.
#For each r, calculate the pmf and store in a list.

from scipy.stats import binom

# setting the values
# of n and p
n = 10
p = 0.2
# defining the list of r values
r_values = list(range(n + 1))
# obtaining the mean and variance 
mean, var = binom.stats(n, p)

mean
var


#scipy.stats.binom.pmf() function is used to obtain the 
#probability mass function for a certain value of r, n and p. 
#We can obtain the distribution by passing all possible values 
#of r(0 to n).

#Syntax : 
# scipy.stats.binom.pmf(r, n, p)

# list of pmf values
dist = [binom.pmf(r, n, p) for r in r_values ]
# printing the table
print("r\tp(r)")
for i in range(n + 1):
    print(str(r_values[i]) + "\t" + str(dist[i]))
# printing mean and variance
print("mean = "+str(mean))
print("variance = "+str(var))



# example: Plotting the graph using matplotlib.pyplot.bar() 
#function to plot vertical bars.

from scipy.stats import binom
import matplotlib.pyplot as plt
# setting the values
# of n and p
n = 1000
p = 0.2
# defining list of r values
r_values = list(range(n + 1))
# list of pmf values
dist = [binom.pmf(r, n, p) for r in r_values ]
# plotting the graph 
plt.bar(r_values, dist)
plt.show()














### Lecture 3. Poisson distribution

#How to Generate a Poisson Distribution
#You can use the poisson.rvs(mu, size) function to generate 
#random values from a Poisson distribution with a specific 
#mean value and sample size:

    
from scipy.stats import poisson

#generate random values from Poisson distribution with mean=8 and sample size=20
poisson.rvs(mu=8, size=20)


# How to Calculate Probabilities Using a Poisson Distribution
# You can use the poisson.pmf(k, mu) 
#to calculate probabilities related to the specific count value
#from Poisson distribution.

#Example 1: Probability Equal to Some Value

#A store sells 8 icecreams per day on average. What is the 
#probability that they will sell 10 icecreams on a given day? 

from scipy.stats import poisson

#calculate probability
poisson.pmf(k=10, mu=8)


#You can use the poisson.cdf(k, mu) functions to calculate 
#cumulative probabilities up to a certain discrete value
# the Poisson distribution.

#Example 2: Probability Less than Some Value
#A call center has on average 5 calls coming in per hour. 
# What is the probability that this call center has four or less incoming calls
# during a given hour?

from scipy.stats import poisson

#calculate probability
poisson.cdf(k=4, mu=5)




#Example 3: Probability where occurence Greater than Some Value

#A certain store sells 15 cans of tuna per day on average. 
#What is the probability that this store sells more than 20 
# cans of tuna in a given day?

from scipy.stats import poisson

#calculate probability
1-poisson.cdf(k=20, mu=15)






























