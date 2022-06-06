#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt;


# In[12]:


import seaborn as sns


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


import pandas as pd


# In[25]:


mit = pd.read_csv('Spotify 2010 - 2019 Top 100.csv')


# In[26]:


mit.head(10)


# In[41]:


sns.distplot(mit['bpm'], kde=False, bins = 50)


# In[43]:


sns.distplot(mit['year released'], kde=False, bins = 40)


# In[46]:


sns.distplot(mit['top year'], kde=False)


# In[62]:


sns.distplot(mit['nrgy'], kde=False, bins = 50)


# In[63]:


sns.distplot(mit['dnce'], kde=False, bins = 40)


# In[61]:


sns.distplot(mit['dur'], kde=False, bins = 50)


# In[66]:


sns.jointplot(x='dur', y='bpm', data=mit)


# In[67]:


sns.jointplot(x='year released', y='dur', data=mit)


# In[68]:


sns.jointplot(x='dur', y='bpm', data=mit, kind='hex')


# In[71]:


sns.jointplot(x='dur', y='bpm', data=mit, kind='reg')


# In[74]:


sns.jointplot(x='dur', y='bpm', data=mit, kind='kde')


# In[81]:


sns.pairplot(mit, hue='artist type', palette='coolwarm')


# In[82]:


sns.rugplot(mit['bpm'])


# In[83]:


sns.distplot(mit['bpm'])


# In[86]:


sns.barplot(x='dur', y='bpm', data=mit)


# In[87]:


sns.countplot(x='year released', data=mit)


# In[90]:


sns.countplot(x='artist type', data=mit)


# In[98]:


sns.boxplot(x='artist type', y='bpm', data=mit)


# In[100]:


sns.violinplot(x='artist type', y='dur', data=mit, split=True)


# In[103]:


sns.stripplot(x='artist type', y='bpm', data=mit, jitter=True, split=True)


# In[111]:


sns.violinplot(x='artist type', y='bpm', data=mit)
sns.swarmplot(x='artist type', y='bpm', data=mit, color='purple')


# In[116]:


sns.factorplot(x='artist type', y='dur', data=mit, kind='bar')


# In[117]:


sns.factorplot(x='artist type', y='dur', data=mit, kind='violin')


# In[123]:


tc = mit.corr()


# In[133]:


sns.heatmap(tc,annot=True,cmap='coolwarm')


# In[125]:


tc


# In[132]:


sns.heatmap(tc,cmap='magma', linecolor='white', linewidths=3)


# In[134]:


sns.heatmap(tc,cmap='coolwarm', linecolor='black', linewidths=3)


# In[135]:


mit['artist type'].unique()


# In[137]:


sns.pairplot(mit)


# In[139]:


g = sns.PairGrid(mit)
g.map(plt.scatter)


# In[143]:


g = sns.PairGrid(mit)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)


# In[148]:


g = sns.FacetGrid(data=mit, col='year released', row='artist type')
g.map(sns.distplot,'dur')


# In[149]:


g = sns.FacetGrid(data=mit, col='year released', row='artist type')
g.map(sns.distplot,'dur','bpm')


# In[160]:


sns.lmplot(x='bpm',y='dur',data=mit,hue='artist type')


# In[162]:


sns.lmplot(x='bpm',y='dur',data=mit,col='artist type',hue='year released')


# In[172]:


sns.set_style('ticks')
sns.countplot(x='artist type', data=mit)


# In[173]:


sns.set_style('ticks')
sns.countplot(x='artist type', data=mit)
sns.despine(left=True,bottom=True)


# In[174]:


plt.figure(figsize=(12,3))
sns.countplot(x='artist type', data=mit)


# In[179]:


sns.set_context('poster')
sns.countplot(x='artist type',data=mit)


# In[180]:


sns.set_context('notebook')
sns.countplot(x='artist type',data=mit)


# In[184]:


sns.lmplot(x='dur',y='bpm',data=mit,hue='artist type', palette='coolwarm')


# In[185]:


sns.lmplot(x='dur',y='bpm',data=mit,hue='artist type', palette='seismic')


# In[ ]:




