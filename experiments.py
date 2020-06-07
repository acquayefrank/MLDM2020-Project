#!/usr/bin/env python
# coding: utf-8

# ## Feature Engineering

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('data/train.csv')


# In[3]:


df.rename(
    columns={'Province_State': 'State', 'Country_Region': 'Country'}, inplace=True
)


# In[4]:


df.head()


# In[5]:


df = pd.concat([df, pd.get_dummies(df['Country'], prefix='Country')], axis=1)


# In[6]:


df.head()


# In[7]:


df = pd.concat([df, pd.get_dummies(df['Target'], prefix='Target')], axis=1)


# In[8]:


df.head()


# In[9]:


# function for replacing all the missings in the state column
def missings(field, country):
    return country if pd.isna(field) == True else field


# In[10]:


df.head()


# In[11]:


df['State'] = df.apply(lambda x: missings(x['State'], x['Country']), axis=1)


# In[12]:


del df['County']


# In[13]:


df.head()


# In[14]:


df['Week'] = pd.to_datetime(df['Date']).dt.week
df['Day'] = pd.to_datetime(df['Date']).dt.day
df['Weekday'] = pd.to_datetime(df['Date']).dt.dayofweek
df['DayOfYear'] = pd.to_datetime(df['Date']).dt.dayofyear


# In[15]:


del df['Country']


# In[16]:


df.head()


# In[17]:


df = pd.concat([df, pd.get_dummies(df['State'], prefix='State')], axis=1)


# In[18]:


df.head()


# In[19]:


del df['State']


# In[20]:


df.head()


# In[21]:


del df['Target']


# In[22]:


mask_test = df['Date'] >= '2020-05-20'
mask_train = df['Date'] < '2020-05-20'


# In[23]:


test_df = df.loc[mask_test]
train_df = df.loc[mask_train]


# In[24]:


del train_df['Date']
del test_df['Date']


# In[25]:


train_df.head()


# In[26]:


test_df.head()


# # Model selection

# ### Ridge Regression

# In[27]:


import numpy as np
from sklearn import linear_model


# In[28]:


reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))


# In[29]:


features_train = list(train_df.columns)
features_test = list(test_df.columns)


# In[30]:


features_train.remove('TargetValue')
features_train.remove('Id')

features_test.remove('TargetValue')
features_test.remove('Id')


# In[31]:


x_train = train_df.loc[:, features_train]
x_test = test_df.loc[:, features_test]


# In[32]:


y_train = train_df.loc[:, ['TargetValue']]
y_test = test_df.loc[:, ['TargetValue']]


# In[ ]:


reg.fit(x_train, x_train)


# In[ ]:


reg_score = reg.score(y_test, y_test)

print(reg_score)

# In[ ]:


from sklearn import svm


# In[ ]:


regr = svm.SVR()


# In[ ]:
