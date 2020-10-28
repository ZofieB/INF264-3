import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as python_random
import tensorflow as tf
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

#setting seeds for reproducability
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)

x_data = pd.read_csv('/content/drive/My Drive/handwritten_digits_images.csv', header = None).to_numpy()
y_data = pd.read_csv('/content/drive/My Drive/handwritten_digits_labels.csv', header = None).to_numpy()

x_data = x_data.reshape(x_data.shape[0], 28, 28)
#one-hot encoding for labels
y_data_ohe = []
for label in y_data:
    ohe_list = [0 for _ in range(10)]
    ohe_list[label[0]] = 1
    y_data_ohe.append(ohe_list)
y_data_ohe = np.array(y_data_ohe)
seed = 666
X_train, X_val_test, Y_train, Y_val_test = model_selection.train_test_split(x_data, y_data_ohe, test_size= 0.2, shuffle=True, random_state = seed)
X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X_val_test, Y_val_test, test_size = 0.5, shuffle=True, random_state=seed)

#reshape, to fit the needed input shape for the model
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)

learning_rates = [0.01, 0.001, 0.0001, 0.00001]
val_accs = []
val_losses = []

for lr in learning_rates:
  tf.random.set_seed(1234)
  #use keras to create a cnn model
  cnn = Sequential()
  #add layers to the model
  cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
  cnn.add(Conv2D(32, kernel_size=3, activation='relu'))
  cnn.add(Flatten())
  cnn.add(Dense(10, activation='softmax'))

  #compile cnn model 
  opt = Adam(learning_rate=lr)
  cnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

  #train cnn model
  cnn_train = cnn.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs = 20, verbose=0)
  #plot accuracies
  print('Learning rate: ', lr)
  
  accur = cnn_train.history['accuracy']
  val_acc = cnn_train.history['val_accuracy']
  val_accs.append(val_acc)
  loss = cnn_train.history['loss']
  val_loss = cnn_train.history['val_loss']
  val_losses.append(val_loss)
  epochs = range(len(accur))
  plt.plot(epochs, accur, 'r', label='Training accuracy')
  plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.figure()
  plt.plot(epochs, loss, 'r', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()

  #test evaluation
  test_evaluation = cnn.evaluate(X_test, Y_test)

  print('Test loss:', test_evaluation[0])
  print('Test accuracy:', test_evaluation[1])
  print('\n--------------------------------------------------------\n')

  #model selection
colors = colors = ['cornflowerblue',
          'tab:orange',
          'tab:green',
          'r',
          'tab:purple',
          'tab:brown',
          'tab:pink',
          'b',
          'tab:olive',
          'tab:cyan',
          'lightcoral',
          'chocolate',
          'springgreen',
          'g']
acc_loss_ratios = [[val_accs[i][j] / val_losses[i][j] for j in range(len(val_accs[i]))] for i in range(len(val_accs))]
plt.figure()
for i in range(len(acc_loss_ratios)): 
  plt.plot(epochs, acc_loss_ratios[i], colors[i], label=(10 ** (- (i + 2))))
plt.title('Accuracy-loss ratios')
plt.legend()
plt.show()

#accuracy seems to be good for 0.0001 learning rate and 5 epochs -> selected model
tf.random.set_seed(1234)
#use keras to create a cnn model
cnn = Sequential()
#add layers to the model
cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
cnn.add(Conv2D(32, kernel_size=3, activation='relu'))
cnn.add(Flatten())
cnn.add(Dense(10, activation='softmax'))

#compile cnn model 
opt = Adam(learning_rate=0.0001)
cnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#train cnn model
cnn_train = cnn.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs = 5, verbose=0)