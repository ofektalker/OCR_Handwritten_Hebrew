# imports:
import numpy as np
import cv2 as cv2
import os
from sklearn.model_selection import train_test_split
from skimage import feature
from skimage import data, exposure
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# pre-processing to classyfied hebrew letters images
# steps:
#   a.change image to grayscale
#   b.padding -----> useful function cv2.copyMakeBorder
#       1.if the image width is smaller then height then add padding rigth and left
#       2.if the image is greater then the height then add padding up and down
#   c.resize to (40,40)


def pre_process_per_image(path):
    img = cv2.imread(path, 0)
    # Padding:
    width, height = img.shape[:2]
    if width < height:
        padded = cv2.copyMakeBorder(img, 0, 0, height - width, height - width, cv2.BORDER_REPLICATE)
    elif width >= height:
        padded = cv2.copyMakeBorder(img, width - height, width - height, 0, 0, cv2.BORDER_REPLICATE)
    #resize to (40,40)
    size_const = (40, 40)
    resized_img = cv2.resize(padded, size_const)
    cv2.imwrite(path, resized_img)
    return resized_img

def pre_process_directory(path):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(path):
        labels.append(subdir)
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".jpg") or filepath.endswith(".png"):
                filepath.replace('/', '\\')
                img = pre_process_per_image(filepath)
                images.append(img)
    labels.pop()
    #using np.array to reduce the memory allocation
    labels = np.array(labels)
    images = np.array(images)
    return images, labels

#test_path = "E:/Studies/4/1/Image processing and computer vision/HW2/hhd_dataset/TRAIN/3/3_1.png"

dir_path = "E:/Studies/4/1/Image processing and computer vision/HW2/hhd_dataset/TRAIN/"
processed_dataset, labels = pre_process_directory(dir_path)
print(processed_dataset.shape)
#processed_dataset = processed_dataset.reshape(processed_dataset.shape[1:])
print(processed_dataset.shape)
print(labels.shape)

# split data to train and validation sets
#train_set, test_set = train_test_split(processed_dataset,labels,train_size=0.9,test_size=0.1, random_state=0)
# sample = train_set[0]
# # HOG
# # TODO: speed up vectorization over for loop
# HOG_IMAGES = []
# for image in train_set:
#     H = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, block_norm="L2")
#     HOG_IMAGES.append(H)

#K-NN Classifier Train:
# k is between 1-15
# distance function are : Euclidean distance , Chi-Square distance



# calculating error for K values between 1 to 15 to get the best k value
# error = []
# for i in range(1,15):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(HOG_IMAGES)
#     pred_i = knn.predict(test_set)
#     error.append(np.mean(pred_i != y_test))

#   Questions for irina:
#       1. did we need to do pre-processe to the test data set??
#       2. did we need to get label vector for classifier?
#       3. sub-dir can be label ?
#       4. did we need to use np.array to reduce the data size ?
#       5. did we to faltten the data
