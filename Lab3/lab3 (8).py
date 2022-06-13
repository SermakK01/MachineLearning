#!/usr/bin/env python
# coding: utf-8

# In[18]:





import numpy as np
import pandas as pd

size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4 
df = pd.DataFrame({'x': X, 'y': y}) 
df.to_csv('dane_do_regresji.csv',index=None)






dataset = pd.read_csv('dane_do_regresji.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values





from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size = 0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)





from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)





import matplotlib.pyplot as plt




import sklearn.neighbors
knn_3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_3_reg.fit(X_train, y_train)
y_predi3 = knn_3_reg.predict(X_test)








import sklearn.neighbors
knn_5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_5_reg.fit(X_train, y_train)
#print(knn_reg5.predict(X_new))
y_predi5 = knn_5_reg.predict(X_test)





from sklearn.linear_model import LinearRegression # import the Linear Regression model
lin_reg = LinearRegression() # creat model object
lin_reg.fit(X_train, y_train)




from sklearn.preprocessing import PolynomialFeatures # importing a class for Polynomial Regression
poly_2_reg = PolynomialFeatures(degree = 2) # our polynomial model is of order
X_poly = poly_2_reg.fit_transform(X_train) # transforms the features to the polynomial form
lin_reg_2 = LinearRegression() # creates a linear regression object
lin_reg_2.fit(X_poly, y_train)





X_grid = np.arange(min(X_test), max(X_test), 0.1) # choice of 0.1 instead of 0.01 to make the graph smoother
X_grid = X_grid.reshape((len(X_grid), 1)) # reshapes the array to be a matrix






poly_3_reg = PolynomialFeatures(degree = 3) # our polynomial model is of order
X_poly3 = poly_3_reg.fit_transform(X_train) # transforms the features to the polynomial form
lin_reg_3 = LinearRegression() # creates a linear regression object
lin_reg_3.fit(X_poly3, y_train)





X_grid3 = np.arange(min(X_test), max(X_test), 0.1) # choice of 0.1 instead of 0.01 to make the graph smoother
X_grid3 = X_grid3.reshape((len(X_grid3), 1)) # reshapes the array to be a matrix






poly_4_reg = PolynomialFeatures(degree = 4) # our polynomial model is of order
X_poly4 = poly_4_reg.fit_transform(X_train) # transforms the features to the polynomial form
lin_reg_4 = LinearRegression() # creates a linear regression object
lin_reg_4.fit(X_poly4, y_train)





X_grid4 = np.arange(min(X_test), max(X_test), 0.1) # choice of 0.1 instead of 0.01 to make the graph smoother
X_grid4 = X_grid4.reshape((len(X_grid4), 1)) # reshapes the array to be a matrix





poly_5_reg = PolynomialFeatures(degree = 5) # our polynomial model is of order
X_poly5 = poly_5_reg.fit_transform(X_train) # transforms the features to the polynomial form
lin_reg_5 = LinearRegression() # creates a linear regression object
lin_reg_5.fit(X_poly5, y_train)




X_grid5 = np.arange(min(X_test), max(X_test), 0.1) # choice of 0.1 instead of 0.01 to make the graph smoother
X_grid5 = X_grid5.reshape((len(X_grid5), 1)) # reshapes the array to be a matrix



import sklearn.metrics as metrics
a = metrics.mean_squared_error(df_train.y, lin_reg.predict(df_train[["x"]]))
b = metrics.mean_squared_error(df_train.y, knn_3_reg.predict(df_train[["x"]]))
c = metrics.mean_squared_error(df_train.y, knn_5_reg.predict(df_train[["x"]]))
d = metrics.mean_squared_error(df_train.y, lin_reg_2.predict(poly_2_reg.fit_transform(df_train[["x"]])))
e = metrics.mean_squared_error(df_train.y, lin_reg_3.predict(poly_3_reg.fit_transform(df_train[["x"]])))
f = metrics.mean_squared_error(df_train.y, lin_reg_4.predict(poly_4_reg.fit_transform(df_train[["x"]])))
g = metrics.mean_squared_error(df_train.y, lin_reg_5.predict(poly_5_reg.fit_transform(df_train[["x"]])))


a1 = metrics.mean_squared_error(df_test.y, lin_reg.predict(df_test[["x"]]))
b1 = metrics.mean_squared_error(df_test.y, knn_3_reg.predict(df_test[["x"]]))
c1 = metrics.mean_squared_error(df_test.y, knn_5_reg.predict(df_test[["x"]]))
d1 = metrics.mean_squared_error(df_test.y, lin_reg_2.predict(poly_2_reg.fit_transform(df_test[["x"]])))
e1 = metrics.mean_squared_error(df_test.y, lin_reg_3.predict(poly_3_reg.fit_transform(df_test[["x"]])))
f1 = metrics.mean_squared_error(df_test.y, lin_reg_4.predict(poly_4_reg.fit_transform(df_test[["x"]])))
g1 = metrics.mean_squared_error(df_test.y, lin_reg_5.predict(poly_5_reg.fit_transform(df_test[["x"]])))






data = {'train_mse':[a,b,c,d,e,f,g], 'test_mse':[a1,b1,c1,d1,e1,f1,g1]}

df_final = pd.DataFrame(data, index = ['lin_reg', 'knn_3_reg', 'knn_5_reg',
'poly_2_reg', 'poly_3_reg', 'poly_4_reg', 'poly_5_reg'])
print(df_final)





import pickle

with open('mse.pkl', 'wb') as f:
    pickle.dump(df_final,f)





poly_feature_2 = PolynomialFeatures(degree = 2, include_bias=False)
poly_feature_3 = PolynomialFeatures(degree = 3, include_bias=False)
poly_feature_4 = PolynomialFeatures(degree = 4, include_bias=False)
poly_feature_5 = PolynomialFeatures(degree = 5, include_bias=False)

poly_2_reg = lin_reg_2
poly_3_reg = lin_reg_3
poly_4_reg = lin_reg_4
poly_5_reg = lin_reg_5





data1 = [(lin_reg,None),(knn_3_reg,None),(knn_5_reg,None),(poly_2_reg,poly_feature_2),(poly_3_reg,poly_feature_3),(poly_4_reg,poly_feature_4),(poly_5_reg,poly_feature_5)]





import pickle

with open('reg.pkl', 'wb') as f:
    pickle.dump(data1,f)




df2 = pd.read_pickle('reg.pkl')

df2




# In[ ]:





# In[ ]:




