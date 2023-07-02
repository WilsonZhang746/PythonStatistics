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

















### Lecture 4. Normal distribution

#Histogram plotting Normal Distribution

import numpy as np
import matplotlib.pyplot as plt
  
# Mean of the distribution
Mean = 32
 
# satndard deviation of the distribution
Standard_deviation  = 8
  
# size
size = 3200
  
# creating a normal distribution data
values = np.random.normal(Mean, Standard_deviation, size)
  
# plotting histograph
plt.hist(values, 10)
# plotting mean line
plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=2)
plt.show()




#Example 1 using Normal Distribution
# Suppose student test scores follow Normal probability distribution.
# with mean 81 and standard deviation 18. 
# Calculate the Percentage of Students who have scores less than 60

# Solution: scipy.stats.norm() 

# import required libraries
from scipy.stats import norm
import numpy as np
 
# Given information
mean = 81
std_dev = 18
total_students = 100
score = 60
 
# Calculate z-score for 60
z_score = (score - mean) / std_dev
 
# Calculate the probability of getting a score less than 60
prob = norm.cdf(z_score)
 
# Calculate the percentage of students who got less than 60 marks
percent = prob * 100

# Print the result
print("Percentage of students who got less than 60 marks:", round(percent, 2), "%")




#Example 2: Calculate the Percentage of Students who have scored 
#More than 95

#To get the percentage of people who have scored more than 95. 
#We first find the probability of people who have scored less than 95 
#then we will subtract the probability from 1 to get the percent of 
# people who have scored more than 95. 

# import required libraries
from scipy.stats import norm
import numpy as np
 
# Given information
mean = 81
std_dev = 18
total_students = 100
score = 95
 
# Calculate z-score for 95
z_score = (score - mean) / std_dev
 
# Calculate the probability of getting a more than 95
prob = norm.cdf(z_score)
 
# Calculate the percentage of students who got more than 95 marks
percent = (1-prob) * 100
# Print the result
print("Percentage of students who got more than /95 marks: ", round(percent, 2), " %")



#Python Code for Percentage of Students who have scored More than 
#75 and less than 85

# import required libraries
from scipy.stats import norm
import numpy as np
 
# Given information
mean = 81
std_dev = 18
total_students = 100
min_score = 75
max_score = 85
 
# Calculate z-score for 75
z_min_score = (min_score - mean) / std_dev
# Calculate z-score for 85
z_max_score = (max_score - mean) / std_dev
 
 
# Calculate the probability of getting less than 70
min_prob = norm.cdf(z_min_score)
 
# Calculate the probability of getting  less than 85
max_prob = norm.cdf(z_max_score)
 
percent = (max_prob-min_prob) * 100
 
# Print the result
print("Percentage of students who got marks between 75 and 85 is", round(percent, 2), "%")




#Find the score under which there are about 80% of the students' scores

# import required libraries
from scipy.stats import norm
import numpy as np
 
# Given information
mean = 81
std_dev = 18
total_students = 100
q_score = 0.8

 

#find the z-value with the cumulative probability 50%
#using norm.ppf() ,which is the inverse of norm.cdf()

z_80 = norm.ppf(q_score)


z_80_score = z_80 * std_dev + mean


z_80_score

#Alternative way

z_80_score = norm.ppf(q_score, loc = mean, scale = std_dev )

z_80_score














### Lecture 5. Shapiro-Wilk test for normality

#NULL hypothesis: Sample is from the normal distributions.(Po>0.05)
#(Rejected): Sample is not from the normal distributions.

#Example 1
# import useful library
import numpy as np
from scipy.stats import shapiro
from numpy.random import randn
 
# Create data
test_data = randn(1000)
 
# conduct the  Shapiro-Wilk Test
shapiro(test_data)         

#The result does not reject the normality hypothesis
#as the p-value > 0.05


#Example 2
## import useful library
import numpy as np
from numpy.random import poisson
from numpy.random import seed
from scipy.stats import shapiro
from numpy.random import randn
 
seed(0)
# Create data
test_data = poisson(5, 200)
 
# conduct the  Shapiro-Wilk Test
shapiro(test_data)

#normality test is rejected , since p-value < 0.05














### Lecture 6. Exponential distribution

from scipy.stats import expon 
import numpy as np
import matplotlib.pyplot as plt

#scale or beta, is the average time between two events
# it is the reciprocal of hazard rate lambda in poisson distribution.
   
# Random Variates
R = expon.rvs(scale = 2,  size = 10)
print ("Random Variates : \n", R)
  

# PDF

quantile = np.arange (0.01, 3, 0.1)

#The threshold parameter defines the lowest possible value in 
#an exponential distribution. Some analysts refer to this parameter
# as the location

#probability density
Den_city = expon.pdf(quantile,  scale = 1)

plt.plot(quantile, Den_city)



#cumulative probability 
quantile = np.arange (0.01, 9, 0.1)
Cum_prob = expon.cdf(quantile,  scale = 1)

plt.plot(quantile, Cum_prob)



















### Lecture 7. Chi-Square test


from scipy.stats import chi2_contingency
 
# defining the table
data = [[32, 20, 17, 10, 7, 3], [56, 39,18, 69, 93, 66]]
stat, p, dof, expected = chi2_contingency(data)
 
# interpret p-value
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
### Lecture 8. Beta distribution

#Creating beta continuous random variable

# importing scipy
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
  

#Probability density
#create beta variates
quantile = np.arange (0.01, 1, 0.1)
a = 1
b = 1

R = beta.pdf(quantile, a, b)
print ("\nProbability Distribution : \n", R)

plt.plot(quantile,R)


a = 2
b = 2

R = beta.pdf(quantile, a, b)
print ("\nProbability Distribution : \n", R)

plt.plot(quantile,R)




#Cumulative probabilities

quantile = np.arange (0.01, 1, 0.1)
a = 2
b = 2

R = beta.cdf(quantile, a, b)
print ("\nProbability Distribution : \n", R)

plt.plot(quantile,R)



#getting quantiles from given cumulative probabilities

probs = np.arange (0.01, 1, 0.1)
a = 2
b = 2

R = beta.ppf(probs, a, b)
print ("\nProbability Distribution : \n", R)

plt.plot(probs,R)





## Generating Beta random variates
a = 6
b = 2

R = beta.rvs(a, b, size = 10)
print ("Random Variates : \n", R)


















### Lecture 9. Lognormal distribution
from scipy.stats import lognorm 
import numpy as np 
import matplotlib.pyplot as plt

quantile = np.arange (0.01, 10, 0.5) 
  
a = 5       #this location refers to lognormal variable, not for the 
             #log of lognormal variables
b = 2       #this is the sigma in density function
  
# PDF  probability density
R = lognorm.pdf(quantile, loc=a, s=b) 

plt.plot(quantile, R)


#cdf, cumulative probability
a = 0
b = 1

R = lognorm.cdf(quantile,loc=a, s=b) 
plt.plot(quantile, R)



#ppf to calculate quantiles from given cumulative probabilites
a = 0
b = 1
probs = np.arange (0.01, 1, 0.1)

R = lognorm.ppf(probs,loc=a, s=b) 
plt.plot(probs, R)




# Random Variates 
N = 32
a = 5
b = 2

R = lognorm.rvs(loc=a, s=b, size=N) 
print(R) 























### Lecture 10. Gamma distribution

from scipy.stats import gamma 
import numpy as np 
import matplotlib.pyplot as plt


quantile = np.arange (0.01, 10, 0.5) 
  
a = 1       # parameter a (shape)
b = 1       # parameter scale 
  
# PDF  probability density
# pdf(quantile, a , scale) 

R = gamma.pdf(quantile, a = a, scale=b) 

plt.plot(quantile, R)



#cdf, cumulative probability
# cdf(quantile, a , scale)

a = 1
b = 1

R = gamma.cdf(quantile,a=a, scale=b) 
plt.plot(quantile, R)



#ppf to calculate quantiles from given cumulative probabilites
a = 1
b = 1
probs = np.arange (0.01, 1, 0.1)

R = gamma.ppf(probs,a=a, scale=b) 
plt.plot(probs, R)




# Random Variates generation
N = 32
a = 3
b = 2

R = gamma.rvs(a=a, scale=b, size=N) 
print(R) 


















### Lecture 11. t distribution

from scipy.stats import t
import numpy as np 
import matplotlib.pyplot as plt



# PDF  probability density
# pdf(quantile, df , loc)
#loc default 0 

quantile = np.arange (-8, 8, 0.1) 
df = 10
loc = 2

R = t.pdf(quantile, df=df) 

plt.plot(quantile, R)


R = t.pdf(quantile, df=df, loc=loc) 

plt.plot(quantile, R)



#cdf, cumulative probability
# cdf(quantile, df , loc)

quantile = np.arange (-8, 8, 0.1) 
df = 10
loc = 2

R = t.cdf(quantile,df=df, loc=loc) 
plt.plot(quantile, R)


#ppf to calculate quantiles from given cumulative probabilites
df = 10
loc = 2
probs = np.arange (0.01, 1, 0.01)

R = t.ppf(probs,df=df, loc=loc) 
plt.plot(probs, R)



#random number generation using t.rvs()

test1 = t.rvs(10, size= 10000)


fig, ax = plt.subplots(figsize =(12, 9))
#ax.hist(test, bins = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
ax.hist(test1, bins = 50)
  
# Show plot
plt.show()


test2 = t.rvs(df=10, loc = 5, size= 10000)


fig, ax = plt.subplots(figsize =(12, 9))
#ax.hist(test, bins = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
ax.hist(test2, bins = 50)
  
# Show plot
plt.show()



##Example of t-test of population mean

# A chemical engineer claims that the population mean yield of 
#a certain batch process is 500 grams per milliliter of raw material.
#To check this claim he samples 25 batches each month. 
#If the computed t-value falls between −t0.05 and t0.05, he is 
#satisfied with this claim. What conclusion should he draw from 
#a sample that has a mean X-Ba = 518 grams per milliliter and 
#a sample standard deviation s = 40 grams? Assume the distribution 
# of yields to be approximately normal.

#Solution : 
# first we get the critical value of t0.05 
# using qt(probs, df)
import math

t_cri = t.ppf( 1-0.05, 24)
t_cri   #1.71

# then we calculate the sample t test statistics
t_sample = (518 - 500) / (40/math.sqrt(25))
t_sample     #2.25

#we can also calcualte the p_value assocated with t_sample
p_sample = 1 - t.cdf(t_sample, 24)
p_sample   #0.0169


#As t_sample > t_cri, which also means p_sample < 0.05, 
# engineer is likely to conclude that the process produces 
# a better product than he thought, i.e. larger than 500 

















### Lecture 12. Normal quantile-quantile plot

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
  
# np.random generates different random numbers
# whenever the code is executed

  
# Random data points from normal distribution generated
data_points = np.random.normal(6, 2, 100)    
  
sm.qqplot(data_points, line='s')
plt.show()



# Random data points from logistic distribution generated
data_points = np.random.logistic(size=100) 

sm.qqplot(data_points, line='s')
plt.show()






































