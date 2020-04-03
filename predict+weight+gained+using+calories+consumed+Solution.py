
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[8]:


cal= pd.read_csv('calories_consumed.csv')


# In[9]:


cal.describe()


# In[10]:


cal['Calories Consumed'].skew()


# In[11]:


plt.hist(cal['Calories Consumed'])


# In[12]:


cal['Calories Consumed'].kurt()


# In[13]:


cal.info()


# In[14]:


sns.pairplot(cal)


# In[15]:


sns.distplot(cal['Calories Consumed'])


# In[16]:


sns.distplot(cal['Weight gained (grams)'])


# In[17]:


cal.corr()


# In[28]:


X = cal['Calories Consumed'].values.reshape(-1,1)
print(X)

Y = cal['Weight gained (grams)'].values.reshape(-1,1)
print(Y)


# In[29]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)


# In[38]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

model = lm.fit(X_train,Y_train)


# In[56]:


print(lm.intercept_)


# In[40]:


print(lm.coef_)


# In[52]:


model.score(X_train,Y_train)   #R^2 value = 0.8908


# In[48]:


predictions = model.predict(X_test)


# In[50]:


plt.scatter(Y_test,predictions)


# In[58]:


plt.scatter(
    cal['Calories Consumed'],
    cal['Weight gained (grams)'],
    c='black'
)
plt.plot(
    X_test,
    predictions,
    c='blue',
    linewidth=2
)


# In[59]:


from sklearn import metrics


# In[61]:


print('MAE',metrics.mean_absolute_error(Y_test,predictions))


# In[62]:


print('MSE',metrics.mean_squared_error(Y_test,predictions))


# In[64]:


print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))

