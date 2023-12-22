# POVa  - Classification of Animals in the Trail Camera  Photos

This project focuses on training a convolution neural network model to detect animals in photographs taken with a trailcam.
## Data preprocessing
For the purpose of the experiments it was necessary to perform the required modifications to the dataset, such as image clustering or background subtraction. The following scripts are used for these purposes.

### [clustering.py](https://github.com/IanNobody/pova/blob/main/clustering.py)

A script that sorts the dataset images from the input directory into a selected number of clusters and creates a structure in the destination directory according to this partition.

usage:
```
clustering.py [-h] -i INPUT_PATH -o OUTPUT_PATH [-f FEATURES_PATH] -n N_CLUSTERS

optional arguments:
  -h --help           show this help message and exit
  -i --input-path      Path to input images.
  -o --output-path     Where to save the clustered images.
  -f --features-path   Where to save the extracted features.
  -n --n-clusters      Number of expected clusters for K-means.

```

### [diff.py](https://github.com/IanNobody/pova/blob/main/diff.py)
A script that aligns the tested image to the reference one and executes 
one of the implemented methods for background subtraction, or just aligns the images and creates pairs.


usage:
```
diff.py [-h] -i INPUT_PATH -o OUTPUT_PATH [-m {diff,background-reduction,align}] 
		[-r RESIZE_FACTOR] [-v] [-p REFERENCE_FILE_PREFIX]

optional arguments:
  -h --help           show this help message and exit
  -i --input-path      Path to input images.
  -o --output-path     Where to save the output images.
  -m --diff-method     Choose diff method: 
                            diff - use rgb diff between images, 
                            background-reduction - use advanced background reduction.
                            align - align images and create pairs (no diff)
  -r --resize-factor   Resize factor for image aligment. Smaller factor will be faster but less accurate.
  -v --verbose         Show images.
  -p  --reference-file-prefix 	Reference file prefix.

```
