# imports:
import numpy as np
import cv2 as cv2
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn import metrics
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier
import pathlib
from datetime import datetime

# functions:
def pre_process_per_image(path):
    img = cv2.imread(path, 0)
    # Padding:
    width, height = img.shape[:2]
    if width < height:
        padded = cv2.copyMakeBorder(img, 0, 0, height - width, height - width, cv2.BORDER_REPLICATE)
    else:
        padded = cv2.copyMakeBorder(img, width - height, width - height, 0, 0, cv2.BORDER_REPLICATE)
    # resize, normalization and global centering :
    size_const = (40, 40)
    resized_img = cv2.resize(padded, size_const).astype('float32')/255
    mean = resized_img.mean()
    resized_img = resized_img - mean
    return resized_img

def pre_process_directory(path):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".jpg") or filepath.endswith(".png"):
                path = pathlib.PurePath(subdir)
                labels.append(path.name)
                filepath.replace('/', '\\')
                img = pre_process_per_image(filepath)
                img = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                  transform_sqrt=False, block_norm="L2")
                images.append(img)
    #using np.array to reduce the memory allocation
    labels = np.array(labels)
    images = np.array(images)
    return images, labels

# chi square function
def chi_square_distance(vec1, vec2):
    return 0.5 * np.sum((vec1 - vec2) ** 2 / (vec1 + vec2 + 1e-6))

# Experiment function over KNN classifier
def observation_knn(func = "euclidean") :
    args = ["func", 0, 0]
    # k is odd to prevent indecision
    for k in range(1, 15, 2):
        classifier = KNeighborsClassifier(n_neighbors=k, metric= func)
        classifier.fit(x_train, y_train)
        acc = classifier.score(x_val, y_val)
        if acc > args[2]:
            args = [func, k, acc]
        print("[INFO] K-nn Model: K = {}   accuracy: {:.2f}%".format(k, acc * 100))
    return args

def take_max_and_run_it(args1,args2):
    max_args = args1 if args1[2] > args2[2] else args2
    knn = KNeighborsClassifier(n_neighbors=max_args[1], metric=max_args[0])
    knn.fit(train_processed, train_labels)
    y_predict = knn.predict(test_processed)
    acc = metrics.accuracy_score(test_labels, y_predict)
    print("[INFO] K-nn Model: K = {}   accuracy: {:.2f}%".format(max_args[1], acc * 100))
    return y_predict, max_args

def make_output(args,test_labels,y_predict):
    header_line = "k = {}, distance function is: {}\nLetter  Accuracy".format(args[1], args[0])
    confusion_matrix = metrics.confusion_matrix(test_labels, y_predict)
    np.savetxt("confusion_matrix.csv", confusion_matrix, delimiter=",")
    accuracies = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    accuracies = accuracies * 100
    letter_acc = [np.append(labels[i], accuracies[i]) for i in range(27)]
    np.savetxt("results.txt", letter_acc, header=header_line, fmt=("%d", "%.2f"), comments='')

start_time = datetime.now()
start_time = start_time.strftime("%H:%M:%S")
print("Start time: ", start_time)
start_time = datetime.now()
# get path from command line
cmd_path = sys.argv[1]
train_path = cmd_path + "/TRAIN"
test_path = cmd_path + "/TEST"
train_processed, train_labels = pre_process_directory(train_path)
test_processed, test_labels = pre_process_directory(test_path)
# split data to train and validation sets
x_train, x_val, y_train, y_val = train_test_split(train_processed, train_labels, train_size=0.9,
                                               test_size=0.1, random_state=41)
# get the ideal arguments for each metric
euclidean_ideal_args = observation_knn()
chi_square_ideal_args = observation_knn(chi_square_distance)
y_predict, ideal_args = take_max_and_run_it(euclidean_ideal_args, chi_square_ideal_args)
ideal_args[0] = ideal_args[0].__name__ if type(ideal_args[0]) != str else str(ideal_args[0])
# preprocess parameters to output
labels = list(range(27))
labels = np.array(labels)
make_output(ideal_args, test_labels, y_predict)

end_time = datetime.now()
end_time = end_time.strftime("%H:%M:%S")
print("end time: ", end_time)
end_time = datetime.now()
total_time = end_time - start_time
print("total time: ", total_time)
