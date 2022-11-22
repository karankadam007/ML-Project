#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


data=pd.read_csv('D:\\IIT Madras Python classes\\New_Start\\Assignment1\\Supervised_learning _assignment\\diabetes (2).csv')


# In[4]:


data.head()


# In[6]:


data.shape


# In[8]:


data.describe()


# In[11]:


data.info()


# In[12]:


x=data.iloc[:,:-1]
y=data['Outcome']


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.70, random_state=0)


# In[15]:


from sklearn.tree import DecisionTreeClassifier


# In[16]:


model=DecisionTreeClassifier()


# In[17]:


model.fit(x_train,y_train)


# In[20]:


ypred=model.predict(x_test)


# In[21]:


ypred


# In[22]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[23]:


confusion_matrix(y_test,ypred)


# In[24]:


(127+44)/(127+30+30+44)


# In[25]:


accuracy_score(y_test,ypred)


# In[ ]:




