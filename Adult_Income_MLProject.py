#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np
df = pd.read_csv(r"C:\Users\Mauricio Cantu\OneDrive\Documents\Datasets and Projects\Adult_Income_MLProject\adult.csv")


# In[59]:


df.head()


# In[60]:


df.education.value_counts()


# In[61]:


df.workclass.value_counts()


# In[62]:


df.occupation.value_counts()


# In[63]:


df.gender.value_counts()


# In[64]:


df = pd.concat([df.drop('occupation', axis=1), pd.get_dummies(df.workclass).add_prefix('occupation_')], axis = 1)
df = pd.concat([df.drop('workclass', axis=1), pd.get_dummies(df.workclass).add_prefix('workclass_')], axis = 1)
df = df.drop('education', axis=1)
df = pd.concat([df.drop('marital-status', axis=1), pd.get_dummies(df['marital-status']).add_prefix('marital-status_')], axis = 1)
df = pd.concat([df.drop('relationship', axis=1), pd.get_dummies(df.relationship).add_prefix('relationship_')], axis = 1)
df = pd.concat([df.drop('race', axis=1), pd.get_dummies(df.race).add_prefix('race_')], axis = 1)
df = pd.concat([df.drop('native-country', axis=1), pd.get_dummies(df['native-country']).add_prefix('native-country_')], axis = 1)


# In[65]:


df.head()


# In[66]:


df['gender'] = df['gender'].apply(lambda x:1 if x == 'Male' else 0)
df['income'] = df['income'].apply(lambda x:1 if x == '>50K' else 0)


# In[67]:


df


# In[68]:


df.columns.values


# In[69]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(18,12))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')


# In[71]:


df.corr()


# In[72]:


correlations = df.corr()['income'].abs()
sorted_correlations = correlations.sort_values()
num_cols_to_drop = int(0.8* len(df.columns))
cols_to_drop = sorted_correlations.iloc[:num_cols_to_drop].index 
df_dropped = df.drop(cols_to_drop, axis=1)


# In[73]:


df_dropped


# In[74]:


plt.figure(figsize=(16,10))
sns.heatmap(df_dropped.corr(), annot=True, cmap='coolwarm')


# In[92]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = df.drop('fnlwgt', axis=1)

train_df, test_df = train_test_split(df, test_size=0.2)


# In[93]:


train_df


# In[94]:


test_df


# In[95]:


train_X = train_df.drop('income', axis=1)
train_y = train_df['income']

test_X = test_df.drop('income', axis=1)
test_y = test_df['income']


# In[96]:


forest = RandomForestClassifier()


# In[97]:


forest.fit(train_X, train_y)


# In[98]:


forest.score(test_X, test_y)


# In[99]:


forest.feature_importances_


# In[100]:


forest.feature_names_in_


# In[101]:


importances = dict(zip(forest.feature_names_in_, forest.feature_importances_))
importances = {k : v for k, v in sorted(importances.items(), key = lambda x:x[1], reverse=True)}


# In[102]:


importances


# In[103]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 250],
    'max_depth': [ 5, 10 , 15, None],
    'min_samples_split': [2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(estimator = RandomForestClassifier(),
                          param_grid = param_grid, verbose=10)


# In[104]:


grid_search.fit(train_X, train_y)


# In[105]:


forest = grid_search.best_estimator_


# In[106]:


forest.score(test_X, test_y)


# In[107]:


importances = dict(zip(forest.feature_names_in_, forest.feature_importances_))
importances = {k : v for k, v in sorted(importances.items(), key = lambda x:x[1], reverse=True)}


# In[108]:


importances


# In[ ]:




