#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np


# In[30]:


data=pd.read_csv('D:\\IITMadrasPythonclasses\\New_Start\\Assignment1\\Supervised_learning _assignment\\diabetes-1.csv')


# In[31]:


data.head()


# In[32]:


data.shape


# In[33]:


data.describe()


# In[34]:


data.info()


# In[35]:


x=data.iloc[:,:-1]


# In[36]:


x


# In[37]:


y=data['Outcome']


# In[38]:


y


# In[39]:


from sklearn.model_selection import train_test_split


# In[40]:


x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.70, random_state=0)


# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[53]:


x_train.shape


# In[42]:


model=RandomForestClassifier()


# In[43]:


model.fit(x_train,y_train)


# In[54]:


y_pred=model.predict(x_test)


# In[55]:


y_pred


# In[56]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[57]:


confusion_matrix(y_test,y_pred)


# In[58]:


accuracy_score(y_test,y_pred)


# In[ ]:





# In[ ]:




