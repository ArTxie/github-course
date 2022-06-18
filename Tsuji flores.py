#!/usr/bin/env python
# coding: utf-8

# In[387]:


import matplotlib.pyplot as plt


# In[388]:


import seaborn as sns


# In[389]:


import pandas as pd


# In[447]:


mit = pd.read_csv('iris.csv')


# In[391]:


mit.head(100)


# In[392]:


from sklearn.linear_model import LinearRegression


# In[393]:


model = LinearRegression(normalize=True)
print(model)


# In[394]:


LinearRegression(copy_X=True, fit_intercept=True, normalize=True)


# In[395]:


import numpy as np


# In[396]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[397]:


mit.info()


# In[398]:


mit.describe()


# In[399]:


mit.columns


# In[400]:


sns.pairplot(mit)


# In[401]:


sns.distplot(mit['sepal.length'])


# In[402]:


mit.corr()


# In[403]:


sns.heatmap(mit.corr(),annot=True)


# In[404]:


mit.columns


# In[405]:


X = mit[['sepal.width', 'petal.length', 'petal.width']]


# In[406]:


y = mit['sepal.length']


# In[407]:


from sklearn.model_selection import train_test_split


# In[408]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[409]:


from sklearn.linear_model import LinearRegression


# In[410]:


lm = LinearRegression()


# In[411]:


lm.fit(X_train,y_train)


# In[412]:


print(lm.intercept_)


# In[413]:


lm.coef_


# In[414]:


X_train.columns


# In[415]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[416]:


cdf


# In[417]:


from sklearn.datasets import load_boston


# In[418]:


boston = load_boston()


# In[419]:


boston.keys()


# In[420]:


predictions = lm.predict(X_test)


# In[421]:


predictions


# In[422]:


y_test


# In[423]:


plt.scatter(y_test,predictions)


# In[424]:


sns.distplot((y_test-predictions))


# In[425]:


from sklearn import metrics


# In[426]:


metrics.mean_absolute_error(y_test,predictions)


# In[427]:


metrics.mean_squared_error(y_test,predictions)


# In[428]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))


# In[429]:


metrics.explained_variance_score(y_test,predictions)


# In[430]:


sns.distplot((y_test-predictions),bins=50)


# In[431]:


mit.head()


# In[343]:


mit.isnull()


# In[344]:


sns.heatmap(mit.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[345]:


sns.set_style('whitegrid')


# In[346]:


sns.countplot(x='variety',data=mit)


# In[347]:


sns.distplot(mit['sepal.length'].dropna(),kde=False,bins=30)


# In[348]:


mit['sepal.length'].plot.hist(bins=35)


# In[349]:


mit.info()


# In[350]:


import cufflinks as cf


# In[351]:


cf.go_offline()


# In[352]:


mit['sepal.length'].iplot(kind='hist',bins=50)


# In[353]:


plt.figure(figsize=(10,7))
sns.boxplot(x='variety',y='sepal.length',data=mit)


# In[354]:


def impute_sl(cols):
    sepal.length = cols[0]
    variety = cols[1]
    
    if pd.isnull(sepal.length):
    
        if variety == 1:
            return 5
        elif variety == 2:
            return 5.9
        else:
            return 6.5
    else:
        return sepal.length
    


# In[432]:


mit['sepal.length'] = mit[['sepal.length','variety']].apply(impute_sl,axis=1)


# In[357]:


sns.heatmap(mit.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[358]:


mit.drop('variety',axis=1,inplace=True)


# In[359]:


mit.head()


# In[377]:


mit.dropna(inplace=True)


# In[433]:


spl = pd.get_dummies(mit['sepal.length'],drop_first=True)


# In[362]:


spl.head()


# In[363]:


spw = pd.get_dummies(mit['sepal.width'],drop_first=True)


# In[364]:


mit = pd.concat([mit,spl,spw],axis=1)


# In[375]:


mit.head()


# In[435]:


X = mit.drop(['sepal.length','sepal.width'],axis=1,inplace=True)


# In[446]:


mit.head()


# In[368]:


from sklearn.linear_model import LogisticRegression


# In[369]:


logmodel = LogisticRegression()


# In[448]:


logmodel.fit(X_train,y_train)


# In[449]:


predictions = logmodel.predict(X_test)


# In[439]:


from sklearn.metrics import classification_report


# In[440]:


print(classification_report(y_test,predictions))


# In[441]:


from sklearn.metrics import confusion_matrix


# In[442]:


confusion_matrix(y_test,predictions)


# In[443]:


metrics.classification_report(y_test,predictions)


# In[450]:


mit.head()


# In[451]:


from sklearn.preprocessing import StandartScaler


# In[452]:


scaler = StandartScaler()


# In[453]:


scaler.fit(mit.drop('sepal.length',axis=1))


# In[456]:


scaled_features = scaler.transform(mit.drop('sepal.length',axis=1))


# In[458]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[459]:


df_feat.head()


# In[460]:


from sklearn.cross_validation import train_test_split


# In[462]:


X = df_feat
y = df['sepal.length']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# In[463]:


from sklearn.neighbors import KNeighborsClassifier


# In[464]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[465]:


knn.fit(X_train,y_train)


# In[466]:


pred = knn.predict(X_test)


# In[467]:


from sklearn.metrics import classification_report,confusion_matrix


# In[468]:


print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[469]:


error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[473]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[474]:


knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:




