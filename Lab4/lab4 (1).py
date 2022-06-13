#!/usr/bin/env python
# coding: utf-8

# In[70]:


from sklearn import datasets

data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[71]:


data_iris = datasets.load_iris()


# In[72]:


from sklearn.model_selection import train_test_split

X = data_breast_cancer.data.iloc[:, 3:5]
y = data_breast_cancer.target
dbc_X_train, dbc_X_test, dbc_y_train, dbc_y_test= train_test_split(X,y, test_size = 0.2, random_state=42)


# In[73]:


import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

svm_clf = Pipeline([("linear_svc",LinearSVC(C=1, loss="hinge")),])
svm_clf.fit(dbc_X_train,dbc_y_train)
svm_clf_automatic = Pipeline([("scaler",StandardScaler()),("linear_svc",LinearSVC(C=1, loss="hinge")),])
svm_clf_automatic.fit(dbc_X_train,dbc_y_train)


# In[74]:


train_svm_clf_automatic_score = svm_clf_automatic.score(dbc_X_train,dbc_y_train)
test_svm_automatic_clf_score = svm_clf_automatic.score(dbc_X_test,dbc_y_test)
train_svm_clf_score = svm_clf.score(dbc_X_train,dbc_y_train)
test_svm_clf_score = svm_clf.score(dbc_X_test,dbc_y_test)


# In[75]:


data1 = [(train_svm_clf_automatic_score),(test_svm_automatic_clf_score),(train_svm_clf_score),(test_svm_clf_score)]


# In[76]:


import pickle

with open('bc_acc.pkl', 'wb') as f:
    pickle.dump(data1,f)


# In[77]:


X_i = data_iris["data"][:,(2,3)]
y_i = (data_iris["target"] == 2).astype(np.int8)

iris_X_train, iris_X_test, iris_y_train, iris_y_test= train_test_split(X_i,y_i, test_size = 0.2, random_state=42)


# In[78]:


svm_clf_i_aut = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1,
                                 loss="hinge"
                                 )),
    ])
svm_clf_i = Pipeline([
        ("linear_svc", LinearSVC(C=1,
                                 loss="hinge"
                                 )),
    ])


# In[79]:


svm_clf_i_aut.fit(iris_X_train,iris_y_train)
svm_clf_i.fit(iris_X_train,iris_y_train)


# In[80]:


train_svm_clf_i_aut = svm_clf_i_aut.score(iris_X_train,iris_y_train)
test_svm_clf_i_aut = svm_clf_i_aut.score(iris_X_test,iris_y_test)
train_svm_clf_i = svm_clf_i.score(iris_X_train,iris_y_train)
test_svm_clf_i = svm_clf_i.score(iris_X_test,iris_y_test)


# In[81]:


data2 = [(train_svm_clf_i_aut),(test_svm_clf_i_aut),(train_svm_clf_i),(test_svm_clf_i)]


# In[82]:


with open('iris_acc.pkl', 'wb') as f:
    pickle.dump(data2,f)

