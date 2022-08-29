import numpy as np
import pandas as pd
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('car_data.csv')
# print(df.shape)


################## Data Preprocessing #####################


# checking unique values
print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
print('\n')

print(df['Selling_Price'].mean())

## check missing values
print(df.isnull().sum())

## filling missing value
## df['Selling_Price'].fillna((df['Selling_Price'].mean()), inplace=True)


fd = df[
    ['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
print(fd.head())

current_year = date.today().year
print(current_year)

fd['no_year'] = current_year - fd['Year']
print(fd.head())
print('\n')

fd.drop(['Year'], axis=1, inplace=True)
print(fd.head())

fd = pd.get_dummies(fd, drop_first=True)
print(fd.head())
print('\n')
print(fd.corr())
print('\n')

# sns.heatmap(fd.corr(), annot=True, cmap='coolwarm')
plt.show()
# get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10, 5))
# plot heat map
g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
# plt.show()

X = fd.iloc[:, 1:]
y = fd.iloc[:, 0]
# print("\n")
# print(y)
# print("\n")
X['Owner'].unique()
X.head()
y.head()

### model creation

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
# Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
# Fitting the training data to our model
regressor.fit(X_train, y_train)
# score of this model
score = regressor.score(X_test, y_test)
print(score)

file = open('lenior_regression.pkl', 'wb')

# dump information to that file
pickle.dump(regressor, file)

# predict the y values
y_pred = regressor.predict(X_test)
# a data frame with actual and predicted values of y
evaluate = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
evaluate.head(10)
