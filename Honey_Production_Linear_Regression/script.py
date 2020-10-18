#import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
plt.style.use('seaborn-poster')
plt.style.use('seaborn-colorblind')

df = pd.read_csv('honeyproduction.csv')
print(df.head())

prod_per_year = df.groupby('year').totalprod.mean().reset_index()
X=prod_per_year['year']
X=X.values.reshape(-1,1)
y= prod_per_year['totalprod']

plt.scatter(X,y)


regr = linear_model.LinearRegression()
regr.fit(X,y)
print(regr.coef_[0])
print(regr.intercept_)

y_predict = regr.predict(X)
plt.plot(X,y_predict)
plt.savefig('linear_regression.png')
plt.show()

X_future = np.array(range(2013,2051))
X_future=X_future.reshape(-1,1)


future_predict = regr.predict(X_future)
plt.figure()
plt.plot(X_future,future_predict)
plt.savefig('future_predict.png')
plt.show()
