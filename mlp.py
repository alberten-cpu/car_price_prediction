import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

model = MLPClassifier(hidden_layer_sizes=(5, 5, 3, 5), activation="tanh", solver='adam', learning_rate_init=0.01,
                      alpha=1e-4, max_iter=1000)
model.fit(X, y)
print(model.loss_)
file = open('mlp_class.pkl', 'wb')

# dump information to that file
pickle.dump(model, file)
