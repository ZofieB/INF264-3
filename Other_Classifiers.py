import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



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
        #EINFÜGEN: 3 RANDOM SEEDS DAMit keras funkt

        #Get Data
        X = pd.read_csv('handwritten_digits_images.csv', header = None).to_numpy()
        Y = pd.read_csv('handwritten_digits_labels.csv', header = None).to_numpy()

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
        knn_test = knn_test(X_train,Y_train,X_val,Y_val)
        svm_test = svm_test(X_train,Y_train,X_val,Y_val)
        cnn_test = cnn_test(X_train,Y_train,X_val,Y_val)

        knn_tupel = knn_test.get_best_model
        svm_tupel = svm_test.get_best_model
        cnn_tupel = cnn_test.get_best_model

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
            Y_test_predict = best_model.predict(best_model.preprocess_data(X_test,Y_test)[0])
        else:
            Y_test_predict = best_model.predict(X_test)
        
        test_acc = accuracy_score(Y_test_predict, Y_test)

        print("The testing-accuracy is "+str(test_acc))
        




