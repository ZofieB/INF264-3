import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as python_random
import tensorflow as tf
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
  
  class knn_test:
    def _init(self):
        pass
    def get_best_model(self,X_train, Y_train, X_val, Y_val):
        pca = PCA(n_components = 12)
        degrees = range(1,8)

        acc_train = []
        acc_val = []
        acc_max = 0
        best_model = make_pipeline(pca, KNeighborsClassifier(n_neighbors=1))
        for k in degrees:
            knn = make_pipeline(pca, KNeighborsClassifier(n_neighbors = k))
            knn.fit(X_train, Y_train)

            acc_train.append(knn.score(X_train,Y_train))
            acc_val.append(knn.score(X_val,Y_val))

            if(knn.score(X_val, Y_val)>acc_max):
                acc_max = knn.score(X_val, Y_val)
                best_model = knn

        plt.figure(figsize=(10,10))
        plt.ylim(0,1)
        plt.plot(degrees, acc_train, label="acc_train")
        plt.plot(degrees, acc_val, label="acc_val")
        plt.title("Accuracies") 
        plt.xlabel("Number of Neighbors")
        plt.legend()

        plt.show()

        #best without scaling: k = 5: val_acc = 0.95
        #best with scaling: k = 7: val_acc = 0.924857
        return (acc_max, best_model)




class svm_test:
    def _init(self):
        pass
    def get_best_model(self, X_train, Y_train, X_val, Y_val):
        pca = PCA(n_components=12)
        degrees = range(1,8)
        acc_train = []
        acc_val = []
        best_acc = 0
        best_model = make_pipeline(pca, SVC(kernel="poly", degree=1))

        for k in degrees:
            svm = make_pipeline(pca, SVC(kernel="poly", degree=k))
            svm.fit(X_train, Y_train)

            acc_train.append(svm.score(X_train, Y_train))
            acc_val.append(svm.score(X_val,Y_val))

            if(svm.score(X_val, Y_val)>best_acc):
                best_acc = svm.score(X_val, Y_val)
                best_model = make_pipeline(pca, SVC(kernel="poly", degree = k))

        plt.figure(figsize=(10,10))
        plt.ylim(0,1)
        plt.plot(degrees, acc_train, label="acc_train")
        plt.plot(degrees, acc_val, label="acc_val")
        plt.title("Accuracies")
        plt.xlabel("degree")
        plt.legend()

        plt.show()

        #best without scaling is k = 3: val_acc = 0.94357
        #best with scaling: k = 3: val_acc = 0.927714
        return (best_acc, best_model)





class test_pipeline:
    def _init(self):
        pass
    def test(self):
        #setting seeds for reproducability
        np.random.seed(123)
        python_random.seed(123)
        tf.random.set_seed(1234)

        #Get Data
        X = pd.read_csv('/content/drive/My Drive/handwritten_digits_images.csv', header = None).to_numpy()
        Y = pd.read_csv('/content/drive/My Drive/handwritten_digits_labels.csv', header = None).to_numpy()

        #Train-test-split
        seed = 414
        X_train, X_val_test, Y_train, Y_val_test = model_selection.train_test_split(X,Y, test_size = 0.2, shuffle=True, random_state=seed)
        seed = 213
        X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X_val_test,Y_val_test, test_size = 0.5, shuffle=True, random_state=seed)


        #Preprocessing: Find best dimension for PCA
        print("Both the knn and svm use a pca to preprocess data. The next graph shows the process of finding the right dimension.")
        explained_variance = []
        dimensions = range(2,20)
        for i in dimensions:
            pca = PCA(n_components=i)
            pca.fit(X_train)

            explained = 0
            for j in range(1,i):
                explained += pca.explained_variance_[j]
                
            explained_variance.append(explained)
        #Search a good dimension
        plt.figure(figsize=(10,10))
        plt.ylim(0,2000000)
        plt.plot(dimensions, explained_variance, label="explained_variance")
        plt.title("Explained Variance")
        plt.xlabel("dimension")
        plt.legend()

        plt.show()
        print("Decided to use dimension=12. This is already implicitly used in the classes knn_test and svm_test")

        #Getting best model and accuracies from each model-family      
        knn_test_ = knn_test()
        svm_test_ = svm_test()
        cnn_test_ = cnn_test()

        knn_tupel = knn_test_.get_best_model(X_train, Y_train, X_val, Y_val)
        svm_tupel = svm_test_.get_best_model(X_train, Y_train, X_val, Y_val)
        cnn_tupel = cnn_test_.get_best_model(X_train, Y_train, X_val, Y_val)

        accuracies = [knn_tupel[0], svm_tupel[0], cnn_tupel[0]]
        models = [knn_tupel[1], svm_tupel[1], cnn_tupel[1]]

        i = np.argmax(accuracies)
        best_acc = accuracies[i]
        best_model = models[i]

        if (i==0):
            name = "knn"
        elif(i==1):
            name = "svm"
        else: 
            name = "cnn"
        
        print("The best model is a "+name+" with an validation-accuracy of "+ str(best_acc))


        #Testing

        #Special Case: Need to specially preprocess if model is cnn
        if (i==2):
            Y_test_predict = best_model.predict_classes(cnn_test_.preprocess_data(X_test,Y_test)[0])
            print(Y_test_predict)
        else:
            Y_test_predict = best_model.predict(X_test)
            print(Y_test_predict)
        print(str(Y_test_predict))
        test_acc = accuracy_score(Y_test_predict, Y_test)

        print("The testing-accuracy is "+str(test_acc))

test = test_pipeline()
test.test()