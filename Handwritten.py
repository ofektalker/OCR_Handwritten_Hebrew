# imports:
import numpy as np
import cv2 as cv2
import os
from sklearn.model_selection import train_test_split
from skimage import feature
from skimage import data, exposure
import matplotlib.pyplot as plt

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
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".jpg") or filepath.endswith(".png"):
                filepath.replace('/', '\\')
                img = pre_process_per_image(filepath)
                images.append(img)
    return images

dir_path = "E:/Studies/4/1/Image processing and computer vision/HW2/hhd_dataset/TRAIN"
test_path =  "E:/Studies/4/1/Image processing and computer vision/HW2/hhd_dataset/TRAIN/4/4_1.png"
processed_dataset = pre_process_directory(dir_path)
# split data to train and validation sets
train_set , test_set = train_test_split(processed_dataset,train_size=0.9,test_size=0.1, random_state= 0)
# HOG
sample_image = train_set[1]
#hog = cv2.HOGDescriptor()
#im = cv2.imread(test_path, 0)
#h = hog.compute(im)

for image in train_set:
    (H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2", visualize=True)
    cv2.imshow("HOG", hogImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#ch_hog = hog(sample_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, block_norm="L2")
#hog_image_rescaled = exposure.rescale_intensity(ch_hog, in_range=(0, 10))

