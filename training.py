from src import pre_process
from src import CNN_encoder
from src import feature_extraction
from src import k_means
from src import knn


if __name__ == "__main__":
    print("#"*50)
    print("\t\t\tPre-process in progress...")
    pre_process.img_pre_process()
    print("\t\t\tCNN_encoder training in progress...")
    CNN_encoder.cnn_training()
    print("\t\t\tFeature extraction in progress...")
    feature_extraction.model_feature_extraction()
    print("\t\t\tKmeans training in progress...")
    k_means.kmeans_training()
    print("\t\t\tKNN in progress...")
    knn.knn_training()
    print("\t\t\tTraining Completed")
    print("#"*50)