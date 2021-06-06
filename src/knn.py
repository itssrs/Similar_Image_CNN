import os
os.chdir('E:\\Image-Classification')
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

#loading csv files. 
train_files = list(pd.read_csv('Dataset/train_file.csv')['file_name'])
test_files = list(pd.read_csv('Dataset/test_file.csv')['file_name'])
X_encoded_reshape = np.load("Dataset/X_encoded_compressed.npy")
lisp = train_files
lisp.extend(test_files)
kmeans = joblib.load('models/kmeans_model.pkl','w+')

def pre_processing():
    clusters_features = []
    cluster_files=[]
    data=[]
    files=[]
    for i in [0,1,2,3,4,5]:
        i_cluster = []
        i_labels=[]
        labels=[]
        for iter,j in enumerate(kmeans.labels_):
            if j==i:
                i_cluster.append(X_encoded_reshape[iter])
                i_labels.append(lisp[iter])
        i_cluster = np.array(i_cluster)
        clusters_features.append(i_cluster)
        cluster_files.append(i_labels)
    for iter,i in enumerate(clusters_features):
        data.extend(i)
        labels.extend([iter for i in range(i.shape[0])])
        files.extend(cluster_files[iter])
    print(np.array(labels).shape)
    print(np.array(data).shape)
    print(np.array(files).shape)
    return labels, data, files

def knn_model(labels, data):
    knn = KNeighborsClassifier(n_neighbors=9,algorithm='ball_tree',n_jobs=-1)
    knn.fit(np.array(data),np.array(labels))
    joblib.dump(knn,'models/knn_model.pkl')    

def knn_training():
    print("lisp shape: ",len(lisp))
    print("X_encoded shape: ",X_encoded_reshape.shape)
    print("Kmeans Label",len(kmeans.labels_))
    labels, data, files = pre_processing()
    print("knn training..")
    knn_model(labels, data)
    print("saving files..")
    kmeans_files_name = pd.DataFrame(files, columns=["Image_name"])
    kmeans_files_name.to_csv("Dataset/kmeans_files_name", index=False)