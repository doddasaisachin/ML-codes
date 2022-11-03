import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('sample data\Iris.csv')
print('Dimension of dataset =', df.shape)
df.head()

X = df.values[:, 1:3]  # get input values from first two columns
y = df.values[:, 5]  # get output values from last coulmn
m = len(y) # Number of training examples

print('Total no of training examples (m) = %s \n' %(m))

# Show only first 5 records
for i in range(5):
    print('X =', X[i, ], ', y =', y[i])

model_ols =  linear_model.LinearRegression(normalize=True)
model_ols.fit(X,y)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)

coef = model_ols.coef_
intercept = model_ols.intercept_
print('coef= ', coef)
print('intercept= ', intercept)

predictedPrice = pd.DataFrame(model_ols.predict(X), columns=['Predicted Price']) # Create new dataframe of column'Predicted Price'
actualPrice = pd.DataFrame(y, columns=['Actual Price'])
actualPrice = actualPrice.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
df_actual_vs_predicted = pd.concat([actualPrice,predictedPrice],axis =1)
df_actual_vs_predicted.T

plt.scatter(y, model_ols.predict(X))
plt.xlabel('Price From Dataset')
plt.ylabel('Price Predicted By Model')
plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches
plt.title("Price From Dataset Vs Price Predicted By Model")

price = model_ols.predict([[2104,    3]])
print('Predicted price of a 1650 sq-ft, 3 br house:', price)

model_r = linear_model.Ridge(normalize= True, alpha= 35)
model_r.fit(X,y)
print('coef= ' , model_r.coef_)
print('intercept= ' , model_r.intercept_)
price = model_r.predict([[2104,    3]])
print('Predicted price of a 1650 sq-ft, 3 br house:', price)

