# OCR Handwritten Hebrew
Optical character recognition handwritten for Hebrew language using K-NN classifier over HHD Dataset [1].
Comparative experiments between two distance metrics: Euclidean, Chi Square and k value among range of 1-15.

## Installation requirement

Python



## Import libraries requirements

```python
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
```

## Run command via Terminal
> python knn_classifier.py HHD_Dataset_path

## Output format
1. results.txt
2. confusion matrix.csv
## Reference
[I. Rabaev, B. Kurar Barakat, A. Churkin and J. El-Sana. The HHD Dataset. The 17th International Conference on Frontiers in Handwriting Recognition, pp. 228-233, 2020.](https://www.researchgate.net/publication/343880780_The_HHD_Dataset)
