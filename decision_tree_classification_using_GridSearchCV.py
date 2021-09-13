#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classification

# ## Importing the libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Importing the dataset

# In[3]:


dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.head()


# In[4]:


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X.shape


# ## Splitting the dataset into the Training set and Test set

# In[5]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)


# In[6]:


X_train


# In[36]:


X_test.shape


# In[37]:


y_train.shape


# In[38]:


y_test.shape


# ## Feature Scaling

# In[10]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[11]:


print(X_train)


# ## Training the Decision Tree Classification model on the Training set

# In[12]:


from sklearn.tree import DecisionTreeClassifier
Decision_tree = DecisionTreeClassifier(criterion='entropy',random_state=0)
Decision_tree.fit(X_train,y_train)


# ## Predicting a new result

# In[13]:


Decision_tree.predict([[25,56000]])


# ## Predicting the Test set results

# In[14]:


y_pred = Decision_tree.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test),1),y_pred.reshape(len(y_pred),1)),1))


#  Making the Confusion Matrix

# In[15]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
acc1 = accuracy_score(y_test,y_pred)
print(cm)
print(acc1)


# In[32]:


## GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters_dict = {'criterion':['gini','entropy'],
    'splitter':['best','random'],
    'max_depth': range(1,10),
    'min_samples_split': range(1,10),
    'min_samples_leaf': range(1,5),}
grid = GridSearchCV(estimator = Decision_tree ,
    param_grid = parameters_dict,
    scoring='accuracy',
    n_jobs= -1,
    iid='deprecated',
    cv= 10,
    verbose=1)

grid.fit(X_train,y_train)


# In[33]:


grid.best_params_


# In[34]:


grid.best_estimator_


# In[35]:


grid.best_score_

