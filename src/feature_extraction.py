import os
os.chdir('E:\\Image-Classification')
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf

train_data = np.load("Dataset/train_data_arr.npy")
test_data = np.load("Dataset/test_data_arr.npy")

optimizer = Adam(learning_rate=0.001)
model = tf.keras.models.load_model("models/encoder_model.h5")
model.compile(optimizer=optimizer, loss='mse') 

def feature_extraction(model, data, layer = 10):
    
    """
    Creating a function to run the initial layers of the encoder model. (to get feature extraction from any layer of the model)
    Arguments:
    model - (Auto encoder model) - Trained model
    data - (np.ndarray) - list of images to get feature extraction from trained model
    layer - (int) - from which layer to take the features(by default = 4)
    Returns:
    pooled_array - (np.ndarray) - array of extracted features of given images
    """

    encoded = K.function([model.layers[0].input],[model.layers[layer].output])
    encoded_array = encoded([data])[0]
    pooled_array = encoded_array.max(axis=-1)
    return encoded_array

def get_batches(data, batch_size=1000):
    
    """
    Taking batch of images for extraction of images.
    Arguments:
    data - (np.ndarray or list) - list of image array to get extracted features.
    batch_size - (int) - Number of images per each batch
    Returns:
    list - extracted features of each images
    """

    if len(data) < batch_size:
        return [data]
    n_batches = len(data) // batch_size
    
    # If batches fit exactly into the size of df.
    if len(data) % batch_size == 0:
        return [data[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]   

    # If there is a remainder.
    else:
        return [data[i*batch_size:min((i+1)*batch_size, len(data))] for i in range(n_batches+1)]

def encoded(d, model):
    X_encoded = []
    i=0
    # Iterate through the full training set.
    for batch in get_batches(d, batch_size=300):
        i+=1
        # This line runs our pooling function on the model for each batch.
        X_encoded.append(feature_extraction(model, batch))
    X_encoded = np.concatenate(X_encoded)
    X_encoded = X_encoded.reshape(X_encoded.shape[0], X_encoded.shape[1]*X_encoded.shape[2]*X_encoded.shape[3])
    return X_encoded

def model_feature_extraction():
    full_data = np.concatenate([train_data,test_data],axis=0)
    X_encoded_reshape = encoded(full_data, model)
    print(X_encoded_reshape.shape)
    np.save('Dataset/X_encoded_compressed.npy',X_encoded_reshape)



