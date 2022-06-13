#!/usr/bin/env python
# coding: utf-8

# In[499]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True) 
print(data_breast_cancer['DESCR'])


# In[500]:


import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})


# In[501]:


from sklearn.tree import DecisionTreeClassifier


# In[502]:


from sklearn.model_selection import train_test_split

X = data_breast_cancer.data[['mean texture', 'mean symmetry']]
y = data_breast_cancer.target
cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test= train_test_split(X,y, test_size = 0.2, random_state=42)


# In[503]:


tree_clf = DecisionTreeClassifier(max_depth=3,random_state=42)
tree_clf.fit(cancer_X_train, cancer_y_train)


# In[504]:


from sklearn.metrics import f1_score
cancer_y_predict_test = tree_clf.predict(cancer_X_test)
cancer_y_predict_train = tree_clf.predict(cancer_X_train)

print(f1_score(cancer_y_test,cancer_y_predict_test))
print(f1_score(cancer_y_train,cancer_y_predict_train))


# In[505]:


from sklearn.tree import export_graphviz
f = "bc"
export_graphviz(tree_clf,out_file=f,feature_names=data_breast_cancer.feature_names[[1,8]],class_names=[str(num)+", "+name for num,name in zip(set(data_breast_cancer.target),data_breast_cancer.target_names)],
        rounded=True,
        filled=True
)


# In[506]:


#import graphviz
#print(graphviz.render('dot', 'png', f))


# In[507]:


cancer_y_predict_train_score = tree_clf.score(cancer_X_train, cancer_y_train)
cancer_y_predict_test_score = tree_clf.score(cancer_X_test, cancer_y_test)


List = [(tree_clf.max_depth),(f1_score(cancer_y_train,cancer_y_predict_train)),(f1_score(cancer_y_test,cancer_y_predict_test)),(cancer_y_predict_train_score),(cancer_y_predict_test_score)]


# In[508]:


print(List)


# In[509]:


import pickle

with open('f1acc_tree.pkl', 'wb') as f:
    pickle.dump(List,f)


# In[518]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2, random_state = 42)


# In[519]:


from sklearn.tree import DecisionTreeRegressor


tree_clf = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_clf.fit(train[["x"]], train.y)
predicted_train = tree_clf.predict(train[["x"]])
predicted_test = tree_clf.predict(test[["x"]])


# In[520]:


from sklearn.metrics import mean_squared_error
mean_squared_error(predicted_train, train.y)


# In[521]:


mean_squared_error(predicted_test, test.y)


# In[522]:


tree_clf.max_depth


# In[523]:


List2 = [(tree_clf.max_depth),(mean_squared_error(predicted_train, train.y)),(mean_squared_error(predicted_test, test.y))]


# In[524]:


print(List2)


# In[525]:


import pickle

with open('mse_tree.pkl', 'wb') as f:
    pickle.dump(List2,f)


# In[ ]:




