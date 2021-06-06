import os
os.chdir('E:\\Image-Classification')
import numpy as np
from sklearn.cluster import KMeans
import joblib
import pickle

def kmeans_training():
    X_encoded = np.load("Dataset/X_encoded_compressed.npy")
    print("feature data shape: ", X_encoded.shape)
    print("Started kmeans training...")
    kmeans = KMeans(n_clusters = 6, random_state=0).fit(X_encoded)
    print("Saving kmeans model...")
    pickle.dump(kmeans, open("models/kmeans_model.pkl","wb"))
    
# if __name__ == "__main__":
#     kmeans_training()