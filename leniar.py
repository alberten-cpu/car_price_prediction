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
