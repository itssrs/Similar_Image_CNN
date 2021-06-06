# Similar Image Retrieval using Autoencoders

`Business Problem`

Problem statement is that we need to find top N similar images given on a query image from a given dataset. Something like this…

![image](https://user-images.githubusercontent.com/62031889/120933146-630df880-c716-11eb-95be-ba8138e64a3e.png)

`Approach`
* Data Extraction / Importing the Data : In this splitting the data and converting Image into array
* Convolutional Auto Encoders : Convolutional Autoencoders(CAEs) are a type of convolutional neural networks. The main difference between them is CAEs are unsupervised learning models in which the former is trained end-to-end to learn filters and combine features with the aim of classifying their input.
It tries to keep the spatial information of the input image data as they are and extract information gently.
> * Encoders: Converting input image into latent space representation through a series of convolutional operations. (Left to centroid)
> * Decoders: It tries to restore the original image from the latent space through a series of upsampling/transpose convolution operations. (centroid to Right) Also known as Deconvolution.

![image](https://user-images.githubusercontent.com/62031889/120933704-ed575c00-c718-11eb-9ade-1f2b07fe4529.png)

* K-Means Clustering : After getting compressed data representation of all images we hereby can apply the K-Means clustering algorithm to group the images into different clusters. This helps us to label the unlabeled data.
* Similarity model through K-Nearest Neighbors : After clustering the data we got labeled data. Now we can perform the K-NN algorithm to find similar images(Nearest Neighbors).

![image](https://user-images.githubusercontent.com/62031889/120933891-c8171d80-c719-11eb-9f46-1fc5506755fe.png)

`Step to run`

> * Download the dataset from [https://drive.google.com/file/d/1VT-8w1rTT2GCE5IE5zFJPMzv7bqca-Ri/view]
> * Download the Repo and copy the downloaded dataset into dataset/animal directory.
> * Run the training.py file which will create models in the **models** directory and save file into **Dataset** directory
> * You can test this using web-based application and simple jupyter notebook.
> * Run main.py file and it will open localhost web (using Streamlit)
> * Or run output.ipynb file and test it.
