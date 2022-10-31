#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('advertising.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.drop('Ad Topic Line',axis=1,inplace=True)


# In[7]:


df.drop('City',axis=1,inplace=True)
df.drop('Country',axis=1,inplace=True)
df.drop('Timestamp',axis=1,inplace=True)


# In[8]:


df.head()


# In[9]:


y = df['Clicked on Ad']


# In[10]:


df.drop('Clicked on Ad',axis=1,inplace=True)


# In[11]:


x = df


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


lm = LogisticRegression()


# In[16]:


lm.fit(x_train,y_train)


# In[17]:


lm.score(x_test,y_test)


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


df.hist(figsize=(10,11))
plt.show()


# In[ ]:




