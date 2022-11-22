#!/usr/bin/env python
# coding: utf-8

# In[4]:


data=pd.read_csv('D:\\IIT Madras Python classes\\New_Start\\Assignment1\\Supervised_learning _assignment\\diabetes-1.csv')


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


data.size


# In[8]:


data.info()


# In[9]:


data.describe()


# In[13]:


data.iloc[:,:-1]


# In[19]:


pd.DataFrame


# In[29]:


x=data.iloc[:,:-1]
y=data['Outcome']


# In[24]:


x


# In[26]:


y


# In[30]:


import pandas as pd
import numpy as np
x=pd.DataFrame(data.iloc[:,:-1])
y=data['Outcome']


# In[31]:


x


# In[32]:


y


# In[39]:


from sklearn.model_selection import train_test_split


# In[40]:


x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.7, random_state=0)


# In[42]:


x_train.shape


# In[44]:


from sklearn.linear_model import LogisticRegression


# In[47]:


import warnings
warnings.filterwarnings('ignore')


# In[48]:


model2=LogisticRegression()


# In[49]:


model2.fit(x_train,y_train)


# In[50]:


y_pred2=model2.predict(x_test)


# In[51]:


y_pred2


# In[52]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[53]:


confusion_matrix(y_test,y_pred2)


# In[54]:


(141+39)/(141+16+35+39)


# In[56]:


accuracy_score(y_test,y_pred2)


# In[ ]:




