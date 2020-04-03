
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:


sal = pd.read_csv('Salary_data.csv')


# In[3]:


sal.describe()


# In[4]:


sal.mean()


# In[5]:


sal.info()


# In[6]:


sns.distplot(sal['YearsExperience'])


# In[7]:


sns.distplot(sal['Salary'])


# In[8]:


sns.pairplot(sal)


# In[9]:


plt.scatter(sal['Salary'],sal['YearsExperience'])


# In[10]:


sal.corr()


# In[11]:


X= sal['YearsExperience'].values.reshape(-1,1)
print(X)


# In[12]:


Y= sal['Salary'].values.reshape(-1,1)
print(Y)


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5)


# In[15]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

model = lm.fit(X_train,Y_train)


# In[16]:


print(lm.intercept_)


# In[17]:


print(lm.coef_)


# In[18]:


model.score(X_train,Y_train)  #R^2 value 


# In[19]:


predictions = model.predict(X_test)


# In[20]:


plt.scatter(Y_test,predictions)


# In[21]:




plt.scatter(
    sal['YearsExperience'],
    sal['Salary'],
    c='black'
)
plt.plot(
    X_test,
    predictions,
    c='blue',
    linewidth=2
)


# In[22]:


from sklearn import metrics


# In[23]:


print('MAE',metrics.mean_absolute_error(Y_test,predictions))


# In[24]:


print('MSE',metrics.mean_squared_error(Y_test,predictions))


# In[25]:


print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))

