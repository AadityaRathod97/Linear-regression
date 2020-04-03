
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:


de = pd.read_csv('delivery_time.csv')


# In[3]:


de.describe()


# In[4]:


de.info()


# In[5]:


sns.pairplot(de)


# In[6]:


sns.distplot(de['Delivery Time'])


# In[7]:


sns.distplot(de['Sorting Time'])


# In[9]:


plt.scatter(de['Delivery Time'],de['Sorting Time'])


# In[10]:


de.corr()


# In[39]:


X = de['Sorting Time'].values.reshape(-1,1)
print(X)


# In[40]:


Y = de['Delivery Time'].values.reshape(-1,1)
print(Y)


# In[134]:


from sklearn.model_selection import train_test_split


# In[174]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5)


# In[175]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

model = lm.fit(X_train,Y_train)


# In[176]:


print(lm.intercept_)


# In[177]:


print(lm.coef_)


# In[178]:


model.score(X_train,Y_train)  #R^2 value 


# In[179]:


predictions = model.predict(X_test)


# In[180]:


plt.scatter(Y_test,predictions)


# In[181]:




plt.scatter(
    de['Delivery Time'],
    de['Sorting Time'],
    c='black'
)
plt.plot(
    X_test,
    predictions,
    c='blue',
    linewidth=2
)


# In[182]:


from sklearn import metrics


# In[183]:


print('MAE',metrics.mean_absolute_error(Y_test,predictions))


# In[53]:


print('MSE',metrics.mean_squared_error(Y_test,predictions))


# In[54]:


print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))

