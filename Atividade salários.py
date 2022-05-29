#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


import numpy as np


# In[5]:


pwd


# In[7]:


sal = pd.read_csv('Salaries.csv')


# In[8]:


sal.head()


# In[9]:


sal.head(2)


# In[10]:


sal.info()


# In[11]:


sal['BasePay'] = pd.to_numeric(sal['BasePay'], errors='coerce')


# In[12]:


sal['BasePay'].mean()


# In[13]:


sal['OvertimePay'] = pd.to_numeric(sal['OvertimePay'], errors='coerce')


# In[14]:


sal['OvertimePay'].max()


# In[15]:


sal['JobTitle']


# In[16]:


sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']


# In[23]:


sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']


# In[17]:


sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']


# In[26]:


sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']


# In[18]:


sal['TotalPayBenefits'].max()


# In[19]:


sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()


# In[20]:


sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName']


# In[21]:


sal.loc[sal['TotalPayBenefits'].idxmax()]


# In[22]:


sal['TotalPayBenefits'].argmax()


# In[23]:


sal.iloc[sal['TotalPayBenefits'].argmax()]


# In[24]:


sal['TotalPayBenefits'].argmin()


# In[25]:


sal.iloc[sal['TotalPayBenefits'].argmin()]


# In[27]:


sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]


# In[28]:


sal[sal['TotalPayBenefits']  == sal['TotalPayBenefits'].min()]


# In[26]:


sal.groupby('Year').mean()


# In[43]:


sal.groupby('Year').mean()['BasePay']


# In[27]:


sal['JobTitle'].unique()


# In[30]:


sal['JobTitle'].nunique()


# In[28]:


sal['JobTitle'].value_counts().head(5)


# In[30]:


sal[sal['Year']==2013]['JobTitle'].value_counts()


# In[31]:


sal[sal['Year']==2013]['JobTitle'].value_counts() == 1


# In[32]:


sum(sal[sal['Year']==2013]['JobTitle'].value_counts() == 1)


# In[33]:


sal['JobTitle']


# In[34]:


def chief_string(title):    
    if 'chief' in title.lower().split():
        return True
    else:
        return False


# In[35]:


sal['JobTitle'].iloc[0]


# In[36]:


chief_string('GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY')


# In[37]:


chief_string('CHIEF MANAGER-METROPOLITAN TRANSIT AUTHORITY')


# In[39]:


sal['JobTitle'].apply(lambda x: chief_string(x))


# In[40]:


sum(sal['JobTitle'].apply(lambda x: chief_string(x)))


# In[41]:


sal['title_len'] = sal['JobTitle'].apply(len)


# In[42]:


sal[['JobTitle','title_len']]


# In[44]:


sal[['JobTitle','title_len']].corr()


# In[45]:


sal[['TotalPayBenefits','title_len']].corr()


# In[ ]:




