import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import svm




#Get Data
X = pd.read_csv('handwritten_digits_images.csv', header = None).to_numpy()
Y = pd.read_csv('handwritten_digits_labels.csv', header = None).to_numpy()

#Train-test-split
seed = 414
X_train, X_val_test, Y_train, Y_val_test = model_selection.train_test_split(X,Y, test_size = 0.2, shuffle=True, random_state=seed)
seed = 213
X_val, X_test, Y_val, Y_test = model_selection.train_test_split(X_val_test,Y_val_test, test_size = 0.5, shuffle=True, random_state=seed)

#Preprocessing
explained_variance = []
dimensions = range(2,8)
for i in dimensions:
    pca = PCA(n_components=i)
    pca.fit(X_train)

    explained = 0
    for j in range(1,i):
        explained += pca.explained_variance[j]

    explained_variance.append(explained)

#Search a good dimension
plt.figure(figsize=(10,10))
plt.ylim(0,1)
plt.plot(dimensions, explained_variance, label="explained_variance")
plt.title("Explained Variance")
plt.xlabel("dimension")
plt.legend()

pca = ...
#models
#Support Vector Classifier
acc_train = []
acc_val = []

degrees = range(1,8)
for k in degrees:
    svm = make_pipeline(pca, svm.SVC(kernel=poly, degree=k))
    svm.fit(X_train)

    acc_train.append(svm.score(X_train, Y_train))
    acc_val.append(svm.score(X_val,Y_val))

plt.figure(figsize=(10,10))
plt.ylim(0,1)
plt.plot(degrees, acc_train, label="acc_train")
plt.plot(degrees, acc_val, label="acc_val")
plt.title("Accuracies")
plt.xlabel("degree")
plt.legend()


#K-NN Classifier
acc_train = []
acc_val = []

for k in degrees:
    knn = make_pipeline(pca, KNeighborsClassifier(n_neighbors = k))
    knn.fit(X_train)

    acc_train.append(knn.score(X_train,Y_train))
    acc_val.append(score(X_val,Y_val))

plt.figure(figsize=(10,10))
plt.ylim(0,1)
plt.plot(degrees, acc_train, label="acc_train")
plt.plot(degrees, acc_val, label="acc_val")
plt.title("Accuracies")
plt.xlabel("degree")
plt.legend()


#Testing

