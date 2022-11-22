#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[6]:


data=pd.read_csv('D:\\IIT Madras Python classes\\New_Start\\Assignment1\\Supervised_learning _assignment\\data (2).csv')


# In[8]:


data.head()


# In[10]:


data.describe()


# In[12]:


data.info()


# In[13]:


data.shape


# In[27]:


x=data.iloc[:,:-1]


# In[28]:


x.head()


# In[29]:


y=data['Salary']


# In[30]:


y.head()


# In[39]:


x=pd.DataFrame(data.iloc[:,:-1])
y=data['Salary']


# In[42]:


from sklearn.model_selection import train_test_split


# In[45]:


x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.66, random_state=0)


# In[46]:


x_train.shape


# In[50]:


from sklearn.linear_model import LinearRegression 


# In[51]:


model=LinearRegression()


# In[52]:


model.fit(x_train,y_train)


# In[53]:


ypred=model.predict(x_test)


# In[55]:


ypred


# In[56]:


from sklearn.metrics import r2_score


# In[57]:


r2_score(y_test,ypred)


# In[ ]:




