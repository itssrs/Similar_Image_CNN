import os
os.chdir('E:\\Image-Classification')
import numpy as np
import joblib
import cv2
import imageio as io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import streamlit as st 


def load_models():
    optimizer = Adam(learning_rate=0.001) 
    model = load_model("models/encoder_model.h5")
    model.compile(optimizer=optimizer, loss='mse')
    knn = joblib.load('models/knn_model.pkl','w+')
    return model, knn


def predict(label,N=9,isurl=False):

    """
    Making predictions for the query images and returns N similar images from the dataset.
    We can either pass filename or the url for the image.
    Arguments:
    label - (string) - file name of the query image.
    N - (int) - Number of images to be returned
    isurl - (string) - if query image is from google is set to True else False(By default = False)
    """
    model, knn = load_models()

    if isurl:
        img = io.imread(label)
        img = cv2.resize(img,(128,128))
    else:
        img_path = 'Dataset/animal/'+label
        img = image.load_img(img_path, target_size=(128,128))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data,axis=0)
    img_data = preprocess_input(img_data)
    feature = K.function([model.layers[0].input],[model.layers[12].output])
    feature = feature(img_data)[0]
    feature = np.array(feature).flatten().reshape(1,-1)
    res = knn.kneighbors(feature.reshape(1,-1),return_distance=True,n_neighbors=N)
    # results_(img,list(res[1][0])[1:])
    return img,list(res[1][0])[1:]