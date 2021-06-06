import os
os.chdir('E:\\Image-Classification')
import numpy as np
#For model training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import imageio as io

train_data = np.load("Dataset/train_data_arr.npy")
test_data = np.load("Dataset/test_data_arr.npy")


def encoder_decoder_model():
    
    """
    Used to build Convolutional Autoencoder model architecture to get compressed image data which is easier to process.
    Returns:
    Auto encoder model
    """
    #Encoder 
    model = Sequential(name='Convolutional_AutoEncoder_Model')
    model.add(Conv2D(8, kernel_size=(2, 2),activation='relu',input_shape=(128, 128, 3),padding='same', name='Encoding_Conv2D_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='Encoding_MaxPooling2D_1'))
    model.add(Conv2D(16, kernel_size=(2, 2),strides=1,kernel_regularizer = tf.keras.regularizers.l2(0.001),activation='relu',padding='same', name='Encoding_Conv2D_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='Encoding_MaxPooling2D_2'))
    model.add(Conv2D(16, kernel_size=(2, 2),strides=1,kernel_regularizer = tf.keras.regularizers.l2(0.001),activation='relu',padding='same', name='Encoding_Conv2D_3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='Encoding_MaxPooling2D_3'))
    
    #Decoder
    model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', kernel_regularizer = tf.keras.regularizers.l2(0.001), padding='same',name='Decoding_Conv2D_4'))
    model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_4'))
    model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', kernel_regularizer = tf.keras.regularizers.l2(0.001), padding='same',name='Decoding_Conv2D_5'))
    model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_5'))
    model.add(Conv2D(8, kernel_size=(2, 2), activation='relu', kernel_regularizer = tf.keras.regularizers.l2(0.001), padding='same',name='Decoding_Conv2D_6'))
    model.add(UpSampling2D((2, 2),name='Decoding_Upsamping2D_6'))
    model.add(Conv2D(3, kernel_size=(2, 2), padding='same',activation='sigmoid',name='Decoding_Output'))
    return model

def cnn_training():
    optimizer = Adam(learning_rate=0.001) 
    model = encoder_decoder_model() 
    print(model.summary())
    model.compile(optimizer=optimizer, loss='mse') 
    early_stopping = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=6,min_delta=0.0001) 
    checkpoint = ModelCheckpoint('models/encoder_model.h5', monitor='val_loss', mode='min', save_best_only=True)   
    print("Model Training Started...")
    model.fit(train_data, train_data, epochs=35, batch_size=16,validation_data=(test_data,test_data),callbacks=[early_stopping,checkpoint]) 

        