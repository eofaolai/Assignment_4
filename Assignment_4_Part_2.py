#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Implement sklearn models to create predictors.
# Use a KMeans regression model with the Iris data set. 
# Graph the fit when using differing numbers of clusters. 
# Graph the result and either corroborate or refute the assumption 
# that the data set represents 3 different varieties of iris.


# In[2]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[4]:


df = pd.read_csv('iris.csv')
df.head(10)


# In[5]:


x = df.iloc[:, [0,1,2,3]].values


# In[6]:


kmeans5 = KMeans(n_clusters=5)
y_kmeans5 = kmeans5.fit_predict(x)
print(y_kmeans5)

kmeans5.cluster_centers_


# In[9]:


plt.scatter(x[:,0],x[:,1],c=y_kmeans5)
plt.title('k=5')
plt.show()


# In[7]:


kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)

kmeans3.cluster_centers_


# In[11]:


plt.scatter(x[:,0],x[:,1],c=y_kmeans3)
plt.title('k=3')
plt.show()


# In[8]:


Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()


# In[ ]:


## Elbow graph shows k=3 is an appropriate

## References:
    ## https://heartbeat.fritz.ai/k-means-clustering-using-sklearn-and-python-4a054d67b187

