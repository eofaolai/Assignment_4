#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Part 1

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
boston = load_boston()

bos = pd.DataFrame(boston.data,columns=boston.feature_names)
#bos.head()
#bos.info()

##References:
    ## https://medium.com/@amitg0161/sklearn-linear-regression-tutorial-with-boston-house-dataset-cde74afd460a
    ## https://acadgild.com/blog/linear-regression-on-boston-housing-data


# In[3]:


bos.info()


# In[4]:


from sklearn.linear_model import LinearRegression

bos['MEDV'] = boston.target

X = bos[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']]
y = bos['MEDV']


# In[6]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)


# In[9]:


print(lm.score(X,y))


# In[14]:


slope = list(abs(lm.coef_))
slope


# In[15]:


max_value = slope.index(max(slope))
header = list(X.head(0))
print(header[max_value],'is the factor which has the largest effect on the price of housing in Boston')


# In[16]:


### This seems wrong... so tried not using the abs value of slope

slope = list(lm.coef_)

max_value = slope.index(max(slope))
header = list(X.head(0))
print(header[max_value],'is the factor which has the largest effect on the price of housing in Boston')

