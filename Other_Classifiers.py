import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler



    
#Get Data
X = pd.read_csv('handwritten_digits_images.csv', header = None).to_numpy()
Y = pd.read_csv('handwritten_digits_labels.csv', header = None).to_numpy()


#Train-test-split
seed = 414
X_train, X_val_test, Y_train, Y_val_test = model_selection.train_test_split(X,Y, test_size = 0.2, shuffle=True, random_state=seed)
seed = 213
X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X_val_test,Y_val_test, test_size = 0.5, shuffle=True, random_state=seed)

#Preprocessing: Find best dimension for PCA
explained_variance = []
dimensions = range(2,20)
for i in dimensions:
    pca = PCA(n_components=i)
    pca.fit(X_train)

    explained = 0
    for j in range(1,i):
        explained += pca.explained_variance_[j]
        
    explained_variance.append(explained)
print("explained_variances: "+str(explained_variance))
#Search a good dimension
plt.figure(figsize=(10,10))
plt.ylim(0,2000000)
plt.plot(dimensions, explained_variance, label="explained_variance")
plt.title("Explained Variance")
plt.xlabel("dimension")
plt.legend()

plt.show()
#decided for i = 12


class knn_test:
    def _init(X_train, Y_train, X_val, Y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val

    def get_best_model(X_train, Y_train, X_val, Y_val):
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
    def _init(X_train, Y_train, X_val, Y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val

    def get_best_model(X_train, Y_train, X_val, Y_val):
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


