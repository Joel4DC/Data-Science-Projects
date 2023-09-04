#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Car Price Prediction


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Data Acquisition


# In[45]:


df=pd.read_csv(r"D:\Professional\Data Science\Acmegrade\Acmegrade Data Science Projects\Car Price Prediction\audi.csv")
display(df)


# In[46]:


#Data Exploration


# In[5]:


import pandas_profiling as pf
display(pf.ProfileReport(df))


# In[6]:


len(df)


# In[7]:


df.shape


# In[8]:


df.dtypes


# In[9]:


df.isna().sum()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


X = df.iloc[:,[0,1,3,4,5,6,7,8]].values
display (X.shape)
display (X)


# In[13]:


Y = df.iloc[:,[2]].values
display (Y.shape)
display (Y)


# In[47]:


display(pd.DataFrame(X).head(5))


# In[ ]:


#Data Pre-processing


# In[15]:


from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
X[:,0] = le1.fit_transform(X[:,0])
le2 = LabelEncoder()
X[:,-4] = le2.fit_transform(X[:,-4])
display (X)


# In[16]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[2])],remainder='passthrough')
X = ct.fit_transform(X)
display (X.shape)
display (pd.DataFrame(X))


# In[17]:


display (pd.DataFrame(X))


# In[18]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
display (pd.DataFrame(X))


# In[19]:


from sklearn.model_selection import train_test_split
(X_train,X_test,Y_train,Y_test) = train_test_split(X,Y,test_size=0.2,random_state=0)
print (X.shape, Y.shape)
print (X_train.shape, Y_train.shape)
print (X_test.shape, Y_test.shape)


# In[20]:


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(random_state=0)
regression.fit(X_train,Y_train)
display (regression)


# In[21]:


y_pred = regression.predict(X_test)
display (y_pred)


# In[22]:


print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))


# In[23]:


from sklearn.metrics import r2_score,mean_absolute_error
print  ('R2 Score ', r2_score(Y_test, y_pred))
print  ('Mean Absolute Error', mean_absolute_error(Y_test,y_pred))


# In[24]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
print(reg)


# In[25]:


y_pred = reg.predict(X_test)
display (y_pred)


# In[26]:


print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))


# In[27]:


from sklearn.metrics import r2_score,mean_absolute_error
print  ('R2 Score ', r2_score(Y_test, y_pred))
print  ('Mean Absolute Error', mean_absolute_error(Y_test,y_pred))


# In[28]:


y_pred = reg.predict(X)
display (y_pred)


# In[29]:


result = pd.concat([df,pd.DataFrame(y_pred)],axis=1)
display( result)


# In[30]:


from sklearn.ensemble import  ExtraTreesRegressor
ET_Model=ExtraTreesRegressor(n_estimators = 120)
ET_Model.fit(X_train,Y_train)
y_predict=ET_Model.predict(X_test)
from sklearn.metrics import r2_score,mean_absolute_error
print  ('R2 Score ', r2_score(Y_test, y_predict))
print  ('Mean Absolute Error', mean_absolute_error(Y_test,y_predict))


# In[31]:


y_pred = reg.predict(X)
display (y_pred)
result = pd.concat([df,pd.DataFrame(y_pred)],axis=1)
display( result)


# In[32]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 80, stop = 1500, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(6, 45, num = 5)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

# create random grid

rand_grid={'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf=RandomForestRegressor()

rCV=RandomizedSearchCV(estimator=rf,param_distributions=rand_grid,scoring='neg_mean_squared_error',n_iter=3,cv=3,random_state=42, n_jobs = 1)


# In[33]:


display (rCV.fit(X_train,Y_train))


# In[34]:


rf_pred=rCV.predict(X_test)
display (rf_pred)


# In[35]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
print('MAE',mean_absolute_error(Y_test,rf_pred))
print('MSE',mean_squared_error(Y_test,rf_pred))


# In[44]:


display (r2_score(Y_test,rf_pred))


# In[ ]:


#Model Training


# In[37]:


pip install catboost


# In[38]:


from catboost import CatBoostRegressor
cat=CatBoostRegressor()
print (cat.fit(X_train,Y_train))


# In[39]:


cat_pred=cat.predict(X_test)
display (cat_pred)


# In[40]:


display (r2_score(Y_test,cat_pred))


# In[ ]:


#Model Deployment


# In[41]:


import pickle 
# Saving model to disk
pickle.dump(cat, open('model.pkl','wb'))


# In[42]:


model=pickle.load(open('model.pkl','rb'))
print (model.predict (X_train))


# In[43]:


#Completed

