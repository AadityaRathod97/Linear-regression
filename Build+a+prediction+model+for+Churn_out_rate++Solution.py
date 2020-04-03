
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
get_ipython().magic('matplotlib inline')



# In[3]:


emp = pd.read_csv('emp_data.csv')


# In[4]:


emp.describe()


# In[5]:


emp.info()


# In[6]:


sns.distplot(emp['Salary_hike'])


# In[7]:


sns.distplot(emp['Churn_out_rate'])


# In[8]:


sns.pairplot(emp) #1.Little or no Multicollinearity between the features:


# In[9]:


plt.scatter(emp['Churn_out_rate'],emp['Salary_hike']) 
#2. Linear Relationship between the features and target:


# In[10]:


emp.corr()


# In[11]:


X= emp['Salary_hike'].values.reshape(-1,1)
print(X)


# In[12]:


Y= emp['Churn_out_rate'].values.reshape(-1,1)

print(Y)


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4)


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
    emp['Salary_hike'],
    emp['Churn_out_rate'],
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


# In[26]:


import pylab


# In[27]:


Y1 = np.sin(X)
pylab.plot(X,Y1)
print(Y1)


# In[28]:


Y3 = np.log(X)
pylab.plot(X,Y3)
print(Y3)


# In[29]:


import yellowbrick as yb
from yellowbrick.regressor import ResidualsPlot

from sklearn.linear_model import LinearRegression
lm=LinearRegression
visualizer = ResidualsPlot(lm)
visualizer.fit(X_train,Y_train) #Fit the training data to the model
visualizer.score(X_test,Y_test) # Evaluate the model on the test data
visualizer.poof() # Draw/show/proof the data


# In[30]:


import statsmodels.api as sm
#4.Normal distribution of error terms:
model = sm.OLS(Y_train,X_train).fit()
res = model.resid #residuals
fig = sm.qqplot(res,fit=True,line='45')
plt.show()  #Q-Qplot for the advertising data set


# In[31]:


mod= sm.OLS(Y_train,X_train) #5.Little or No autocorrelation in the residuals:
results = mod.fit()
print(results.summary())  #Summary of the fitted Linear Model

