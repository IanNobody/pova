import os
from os.path import join
import shutil
import pickle
import numpy as np
from PIL import Image
import argparse

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from keras.preprocessing.image import load_img
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.models import Model

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def parseargs():
    parser = argparse.ArgumentParser('Cluster images by their similarity.')
    parser.add_argument('-i', '--input-path', required=True, help='Path to input images.')
    parser.add_argument('-o', '--output-path', required=True, help='Where to save the clustered images.')
    parser.add_argument('-f', '--features-path', default=r"./features", help='Where to save the extracted features.')
    parser.add_argument('-n', '--n-clusters', required=True, type=int, help='Number of expected clusters for K-means.')
    args = parser.parse_args()
    return args


class ImageDataset(Dataset):
    def __init__(self, images_dir):
        self.images = [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".JPG")]
        self.dir = images_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = load_img(join(self.dir, self.images[idx]), target_size=(224,224))
        img = np.array(img)
        reshaped_img = img.reshape(224, 224, 3)
        return {"filename": self.images[idx], "data": preprocess_input(reshaped_img)}


def collate(data):
    filenames = [d["filename"] for d in data]
    images = [d["data"] for d in data]
    return {"filename": np.array(filenames), "data": np.array(images)}


def save_features_batch(data, dest_path: str, batch_id):
    file_name = "features_" + str(batch_id) + ".bin"
    file_path = join(dest_path, file_name)
    with open(file_path, 'wb') as features_file:
        pickle.dump(data, features_file)


def load_feature_batch(file_name, dest_path):
    file_path = join(dest_path, file_name)
    with open(file_path, 'rb') as features_file:
        return pickle.load(features_file)


def save_batch(batch_data, filenames, dest_path, batch_id):
    transformed_dict = dict(zip(filenames, batch_data))
    save_features_batch(transformed_dict, dest_path, batch_id)


def sort_images_by_kmeans(groups, src_path, dst_path):
    for group in groups:
      dst_dir = join(dst_path, str(group))
      os.mkdir(dst_dir)
      for img_file in groups[group]:
        img = Image.open(join(src_path, img_file[0]))
        img.save(join(dst_dir, img_file[0]))


def get_images(path):
    image_names = []
    with os.scandir(path) as files:
        for file in files:
            if file.name.endswith('.JPG'):
                image_names.append(file.name)
    return image_names


def extract_features(dataloader, model, dest_path):
    for bidx, batch in enumerate(dataloader):
        filenames = batch["filename"]
        feat = model.predict(batch["data"], batch_size=len(batch), use_multiprocessing=True)
        save_batch(feat, filenames, dest_path, bidx)


if __name__ == '__main__':
    args = parseargs()
    data_path = args.input_path
    output_path = args.output_path
    features_path = args.features_path
    n_clusters = args.n_clusters

    shutil.rmtree(features_path, ignore_errors=True)
    os.mkdir(features_path)
    model = VGG19()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    data = ImageDataset(data_path)
    dataloader = DataLoader(data, batch_size=80, collate_fn=collate)
    extract_features(dataloader, model, features_path)

    data = {}
    for file in os.listdir(features_path):
        batch_feat = load_feature_batch(file, features_path)
        data.update(batch_feat)

    filenames = np.vstack(list(data.keys()))
    feat = np.vstack(list(data.values()))
    feat = feat.reshape(-1, 4096)

    pca = PCA(n_components=min(feat.shape[0], 100), random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=22)
    kmeans.fit(x)

    groups = {}
    for file, cluster in zip(filenames, kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    sort_images_by_kmeans(groups, data_path, output_path)


