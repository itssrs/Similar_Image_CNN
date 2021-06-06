import streamlit as st 
from PIL import Image
from src import prediction
import os
os.chdir('E:\\Image-Classification')
import cv2
import pandas as pd

kmeans_files = list(pd.read_csv('Dataset/kmeans_files.csv')['Image_name'])

@st.cache
def read(img):
        image = cv2.imread('Dataset/animal/'+img)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image


if __name__ == "__main__":
    st.title("Similar Image Classification")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = uploaded_file.name
        img, img_list = prediction.predict(label)
        for i in range(len(img_list)):
            st.write("Similar Image are:")
            st.image(read(kmeans_files[i]), use_column_width='auto')
