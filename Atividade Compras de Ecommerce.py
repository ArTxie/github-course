#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


ecom = pd.read_csv('Ecommerce Purchases.csv')


# In[5]:


ecom.head()


# In[19]:


ecom.columns


# In[20]:


len(ecom.columns)


# In[21]:


len(ecom.index)


# In[6]:


ecom.info()


# In[7]:


ecom['Purchase Price'].mean()


# In[8]:


ecom['Purchase Price'].max()


# In[9]:


ecom['Purchase Price'].min()


# In[22]:


ecom.head(3)


# In[10]:


ecom[ecom['Language']=='en'].count()


# In[11]:


ecom[ecom['Job'] == 'Lawyer'].info()


# In[26]:


len(ecom[ecom['Job'] == 'Lawyer'].index)


# In[12]:


ecom['AM or PM'].value_counts()


# In[13]:


ecom['Job'].value_counts().head(5)


# In[14]:


ecom[ecom['Lot']=='90 WT']['Purchase Price']


# In[15]:


ecom[ecom["Credit Card"] == 4926535242672853]['Email']


# In[16]:


ecom[(ecom['CC Provider']=='American Express') & (ecom['Purchase Price']>95)].count()


# In[29]:


len(ecom[(ecom['CC Provider']=='American Express') & (ecom['Purchase Price']>95)].index)


# In[17]:


sum(ecom['CC Exp Date'].apply(lambda x: x[3:]) == '25')


# In[31]:


ecom[ecom['CC Exp Date'].apply(lambda exp: exp[3:]== '25')].count()


# In[32]:


ecom['Email']


# In[33]:


example_email = ecom['Email'].iloc[0]


# In[34]:


example_email.split('@')[1]


# In[35]:


ecom['Email'].apply(lambda email: email.split('@')[1]) #.value_counts()


# In[18]:


ecom['Email'].apply(lambda x: x.split('@')[1]).value_counts().head(5)


# In[ ]:




