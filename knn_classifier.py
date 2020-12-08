# imports:
import csv
import numpy as np
import cv2 as cv2
import os
import sys
from sklearn.model_selection import train_test_split
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier

def pre_process_per_image(path):
    img = cv2.imread(path, 0)
    # Padding:
    width, height = img.shape[:2]
    if width < height:
        padded = cv2.copyMakeBorder(img, 0, 0, height - width, height - width, cv2.BORDER_REPLICATE)
    elif width >= height:
        padded = cv2.copyMakeBorder(img, width - height, width - height, 0, 0, cv2.BORDER_REPLICATE)
    # resize and normalization:
    size_const = (40, 40)
    resized_img = cv2.resize(padded, size_const)/255
    return resized_img

def pre_process_directory(path):
    images = []
    labels = []
    index = -1
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".jpg") or filepath.endswith(".png"):
                labels.append(index)
                filepath.replace('/', '\\')
                img = pre_process_per_image(filepath)
                img = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, block_norm="L2")
                images.append(img)
        index += 1
    #using np.array to reduce the memory allocation
    labels = np.array(labels)
    images = np.array(images)
    return images, labels

<<<<<<< HEAD
train_path = "E:/Studies/4/1/Image processing and computer vision/HW2/hhd_dataset/TRAIN"
test_path = "E:/Studies/4/1/Image processing and computer vision/HW2/hhd_dataset/TEST"
train_processed_dataset, train_labels = pre_process_directory(train_path)
test_processed_dataset, test_labels = pre_process_directory(test_path)
=======
#test_path = "E:/Studies/4/1/Image processing and computer vision/HW2/hhd_dataset/TRAIN/3/3_1.png"
dir_path = "E:/Studies/4/1/Image processing and computer vision/HW2/hhd_dataset/TRAIN"
Gilad_dir_Path = "C:/Users/gilad/PycharmProjects/OCR_Handwritten_Hebrew/hhd_dataset/TRAIN"
#cmd_path = sys.argv[1]
processed_dataset, labels = pre_process_directory(dir_path)
print(processed_dataset.shape)
>>>>>>> origin/master


# split data to train and validation sets
x_train, x_val, y_train, y_val = train_test_split(train_processed_dataset,train_labels,train_size=0.9,test_size=0.1, random_state=41)

<<<<<<< HEAD
def chi_square_distance(vec1, vec2):
    return 0.5 * np.sum((vec1 - vec2) ** 2 / (vec1 + vec2 + 1e-6))

# K-NN Classifier:
# loop to check the which parameters gives the best accuracy
# code reference to test the accuracy results
# for i in range(1, 15, 2):
#     knn = KNeighborsClassifier(n_neighbors=i, metric=chi_square_distance)
#     knn.fit(x_train, y_train)
#     pred_i = knn.predict(x_val)
#     acc = knn.score(x_val, y_val)
#     print("[INFO] K-nn Model: K = {}   accuracy: {:.2f}%".format(i,acc * 100))
=======
# reshape to config to Classifier function
print(HOG_train.shape)
print(HOG_test.shape)
#K-NN Classifier:
>>>>>>> origin/master

knn = KNeighborsClassifier(n_neighbors=9, metric=chi_square_distance)
knn.fit(x_train, y_train)
acc = knn.score(x_val, y_val)
print("[INFO] K-nn Model: K = {}   accuracy: {:.2f}%".format(acc * 100))

# results:  The chosen parameters are : (metric = chi-square distance , K = 9 , random state = 41) => accuracy = 76.9

# TODO:
#  1.run knn over TEST dir
#  2. create "results.txt"
#  3.confusion_matrix
#  4. take path argument from command line
#  5. refactor -> move to Functions

<<<<<<< HEAD
=======
#   Questions for irina:
#       1. did we need to do pre-processe to the test data set??
#       2. did we need to get label vector for classifier?
#       3. sub-dir can be label ?
#       4. did we need to use np.array to reduce the data size ?
#       5. did we to faltten the data

def resultTxtCreate():
    f = open("result.txt", "w")

def confusionMatrixCsv():
    with open('result.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        with open('confusion_matrix.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)
>>>>>>> origin/master
