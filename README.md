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

## Training the model

Before training and evaluating the model, one must first download the pretrained weight and put to the `Image-Classification-PyTorch/pretrained_base/` subdirectory.

The weights file can be downloaded from [GDrive](https://drive.google.com/file/d/1h9nnDOLKFoxDkNqdNtA5MroktJzpQ7N-/view?usp=share_link).
All variants of the custom trailcam dataset can be downloaded from this [link](https://drive.google.com/drive/folders/1cUVipu0qP5ok07SDldgXmP49moeCHuHc?usp=share_link).
All present datasets are already splitted to train/test subsets. 

For training the original AlexNet model on one of the specially preprocessed datasets, you can use the following:

`python main.py --model alexnet --data_path /path/to/data --model_save True`
(For this combination of parameters, `custom_masked.zip` and `custom_raw.zip` datasets can be used.)


When training on dataset of image pairs (target/background), you need to also specify the following:

`python main.py --model custom --data_path /path/to/data --model_save True --ref_dataset True`
(Corresponding dataset from the download link is named `custom_paired.zip`)

