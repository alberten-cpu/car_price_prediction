lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

model = tree.DecisionTreeClassifier()
model.fit(X, y_transformed)
score = model.score(X, y_transformed)
print(score * 100)

file = open('decision_tree.pkl', 'wb')

# dump information to that file
pickle.dump(model, file)
