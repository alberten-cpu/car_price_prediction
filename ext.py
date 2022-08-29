### Feature Importance

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

model = ExtraTreesRegressor()
model.fit(X, y)
score = model.score(X, y)
# print(score * 100)
print(model.feature_importances_)
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()
file = open('extratree_org.pkl', 'wb')

# dump information to that file
pickle.dump(model, file)
# exit()
