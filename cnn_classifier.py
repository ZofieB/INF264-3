import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection

x_data = pd.read_csv('handwritten_digits_images.csv', header = None).to_numpy()
y_data = pd.read_csv('handwritten_digits_labels.csv', header = None).to_numpy()

x_data = x_data.reshape(x_data.shape[0], 28, 28)

#one-hot encoding for labels
y_data_ohe = np.array([])
for label in y_data:
    ohe_list = [0,0,0,0,0,0,0,0,0,0]
    ohe_list[label] = 1
    y_data_ohe.append(ohe_list)

seed = 666
X_train, X_val_test, Y_train, Y_val_test = model_selection.train_test_split(x_data, y_data, test_size= 0.2, shuffle=True, random_state = seed)
X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X_val_test, Y_val_test, test_size = 0.5, shuffle=True, random_state=seed)

#reshape, to fit the needed input shape for the model
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)

print("finish")