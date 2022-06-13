#!/usr/bin/env python
# coding: utf-8

# In[71]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer()


# In[72]:


from sklearn.datasets import load_iris 
data_iris = load_iris()


# In[73]:


X_cancer = data_breast_cancer.data
y_cancer = data_breast_cancer.target


# In[74]:


X_iris = data_iris.data
y_iris = data_iris.target


# In[75]:


from sklearn.decomposition import PCA
pca_cancer = PCA(n_components=0.9)
X_cancer_reduced = pca_cancer.fit_transform(X_cancer)
print(pca_cancer.explained_variance_ratio_)


# In[76]:


from sklearn.decomposition import PCA
pca_iris = PCA(n_components=0.90)
X_iris_reduced = pca_iris.fit_transform(X_iris)
print(pca_iris.explained_variance_ratio_)


# In[77]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_cancer_scaled = scaler.fit_transform(X_cancer)
X_iris_scaled = scaler.fit_transform(X_iris)


# In[78]:


pca_cancer = PCA(n_components=0.9)
X_cancer_reduced = pca_cancer.fit_transform(X_cancer_scaled)
print(pca_cancer.explained_variance_ratio_)


# In[79]:


pca_iris = PCA(n_components=0.90)
X_iris_reduced = pca_iris.fit_transform(X_iris_scaled)
print(pca_iris.explained_variance_ratio_)


# In[80]:


import pickle 
with open('pca_bc.pkl', 'wb') as f:
   pickle.dump(pca_cancer.explained_variance_ratio_,f)

import pickle 
with open('pca_ir.pkl', 'wb') as f:
   pickle.dump(pca_iris.explained_variance_ratio_,f)


# In[81]:


import numpy as np
list1 = []
for row in pca_cancer.components_:
    list1.append(np.argmax(row))


# In[82]:


print(list1)


# In[83]:


import numpy as np
list2 = []
for row in pca_iris.components_:
    list2.append(np.argmax(row))


# In[84]:


print(list2)


# In[85]:


import pickle 
with open('idx_bc.pkl', 'wb') as f:
   pickle.dump(list1,f) #dla danych przeskalowanych

import pickle 
with open('idx_ir.pkl', 'wb') as f:
   pickle.dump(list2,f) #dla danych przeskalowanych

