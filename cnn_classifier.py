import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as python_random
import tensorflow as tf
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
class cnn_test:
  def __init__(self):
    pass
  learning_rates = [0.01, 0.001, 0.0001, 0.00001]
  colors = colors = ['cornflowerblue',
              'plum',
              'tab:olive',
              'turquoise'
              ]

  def preprocess_data(self, x_data, y_data):
    #one-hot encoding for labels in each data set
    y_data_ohe = self.onehot_encoding(y_data)

    #reshape, to fit the needed input shape for the model
    x_data = x_data.reshape(x_data.shape[0], 28, 28)
    x_data = x_data.reshape(x_data.shape[0], 28, 28, 1)
    return (x_data, y_data_ohe)

  def onehot_encoding(self, data):
    #one-hot encoding for labels in each data set
    data_ohe = []
    for label in data:
        ohe_list = [0 for _ in range(10)]
        ohe_list[label[0]] = 1
        data_ohe.append(ohe_list)
    return np.array(data_ohe)

  def create_cnn(self, lr):
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
    return cnn
  
  def plot_acc_loss(self, accur, val_accur, loss, val_loss):
    epochs = range(len(accur))
    plt.plot(epochs, accur, self.colors[0], label='Training accuracy')
    plt.plot(epochs, val_accur, self.colors[1], label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, self.colors[2], label='Training loss')
    plt.plot(epochs, val_loss, self.colors[3], label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
  
  def train_cnn(self, X_train, Y_train, X_val, Y_val):
    val_accs = []
    val_losses = []

    for lr in self.learning_rates:
      cnn = self.create_cnn(lr)
      #train cnn model
      cnn_train = cnn.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs = 15, verbose=0)
      #plot accuracies
      print('Learning rate: ', lr)
      
      accur = cnn_train.history['accuracy']
      val_acc = cnn_train.history['val_accuracy']
      val_accs.append(val_acc)
      loss = cnn_train.history['loss']
      val_loss = cnn_train.history['val_loss']
      val_losses.append(val_loss)

      self.plot_acc_loss(accur, val_acc, loss, val_loss)

    return (val_accs, val_losses)

  def choose_model(self, val_accs, val_losses):
    #model selection
    acc_loss_ratios = [[val_accs[i][j] / val_losses[i][j] for j in range(len(val_accs[i]))] for i in range(len(val_accs))]
    plt.figure()
    for i in range(len(acc_loss_ratios)): 
      plt.plot(range(len(acc_loss_ratios[i])), acc_loss_ratios[i], self.colors[i], label=(10 ** (- (i + 2))))
    plt.title('Accuracy-loss ratios')
    plt.legend()
    plt.show()

    index = [0,0]
    max = 0
    for i in range(len(acc_loss_ratios)):
      for j in range(len(acc_loss_ratios[i])):
        if acc_loss_ratios[i][j] > max:
          max = acc_loss_ratios[i][j]
          index[0] = i
          index[1] = j

    print('Best CNN model:\n\tLearningrate: ', self.learning_rates[index[0]], ' Epochs: ', index[1])
    return (self.learning_rates[index[0]], index[1])

  def get_best_model(self, X_train, Y_train, X_val, Y_val):
    X_train_pp, Y_train_pp = self.preprocess_data(X_train, Y_train)
    print(np.shape(X_train_pp), np.shape(Y_train_pp))
    X_val_pp, Y_val_pp = self.preprocess_data(X_val, Y_val)
    print(np.shape(X_val_pp), np.shape(Y_val_pp))
    val_acc, val_loss = self.train_cnn(X_train_pp, Y_train_pp, X_val_pp, Y_val_pp)
    lr, epoch = self.choose_model(val_acc, val_loss)

    #create and train cnn model
    cnn = self.create_cnn(lr)
    cnn_train = cnn.fit(X_train_pp, Y_train_pp, validation_data=(X_val_pp, Y_val_pp), epochs = epoch, verbose=0)
    accs = cnn_train.history['val_accuracy']
    return (accs[-1], cnn)