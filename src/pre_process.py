# For commands
import os
os.chdir('E:\\Image-Classification')
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

file_path = os.listdir('Dataset/animal')

def split_dataset(file_path):

    train_data, test_data = train_test_split(file_path, test_size=0.2)
    train_data = pd.DataFrame(train_data, columns=['file_name'])
    test_data = pd.DataFrame(test_data, columns=['file_name'])
    return train_data, test_data

def img2Array(file_array):
    """
    Reading and Converting images into numpy array by taking path of images.
    Arguments:
    file_array - (list) - list of file(path) names
    Returns:
    A numpy array of images. (np.ndarray)
    """
    
    img_array = []
    for path in file_array:
        img = cv2.imread('Dataset/animal/'+path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,128))
        img_array.append(np.array(img))
    img_array = np.array(img_array)
    img_array = img_array.reshape(img_array.shape[0], 128, 128, 3)
    img_array = img_array.astype('float32')
    img_array /= 255
    
    return np.array(img_array)

def img_pre_process():
    train_file, test_file = split_dataset(file_path)
    train_data = pd.DataFrame(train_file, columns=["file_name"])
    test_data = pd.DataFrame(test_file, columns=["file_name"])
    train_data_list = list(train_data['file_name'])
    test_data_list = list(test_data['file_name'])
    print("Converting image to array..")
    train_data_arr = img2Array(train_data_list)
    test_data_arr = img2Array(test_data_list)
    # save_to_csv(train_data_arr, test_data_arr)
    print("Saving pre-processed file...")
    train_data.to_csv('Dataset/train_file.csv')
    test_data.to_csv('Dataset/test_file.csv')
    np.save("Dataset/train_data_arr.npy",train_data_arr)
    np.save("Dataset/test_data_arr.npy",test_data_arr)


