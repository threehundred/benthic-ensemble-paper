import tensorflow as tf

from random import shuffle, sample

import os
import random
import tensorflow.keras
import pandas as pd
import sqlite3
import numpy as np
import pickle

from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import utils
#from import utils
from multilabeldirectoryiterator import MultiLabelDirectoryIterator
from fullimagepointcroppingloader import FullImagePointCroppingLoader


class KerasDataset:
    SQLITE = "SQLITE"
    CSV = "CSV"

    def __init__(self,
                 filepath,
                 label_key,
                 image_path_key,
                 category_limit=10000000,
                 query=None,
                 save_path=None,
                 img_width=256,
                 img_height=256,
                 batch_size=16,
                 patch_sizes=[]):

        self.save_path = save_path
        self.IMG_WIDTH = img_width
        self.IMG_HEIGHT = img_height
        self.BATCH_SIZE = batch_size


        if ".sqlite" in filepath:
            self.X_train, \
            self.X_val, \
            self.X_test, \
            self.y_train, \
            self.y_val, \
            self.y_test, \
            self.classes, \
            self.class_weight_dict = self.package_from_sqlite(filepath, query, label_key, image_path_key, category_limit, save_path)
            self.mean_image = self.calculate_mean_image(self.X_train)
        elif ".csv" in filepath:
            self.X_train, \
            self.X_val, \
            self.X_test, \
            self.y_train, \
            self.y_val, \
            self.y_test, \
            self.classes, \
            self.class_weight_dict = self.package_from_csv(filepath, label_key, image_path_key, category_limit, save_path)
            self.mean_image = self.calculate_mean_image(self.X_train)
        else:
            self.X_train, \
            self.X_val, \
            self.X_test, \
            self.y_train, \
            self.y_val, \
            self.y_test, \
            self.classes, \
            self.class_weight_dict = self.load_saved_data(filepath)
            self.mean_image = self.load_mean_image(filepath)

        self.training = self.make_train_generator(self.X_train, self.y_train, patch_sizes)
        self.validation = self.make_val_generator(self.X_val, self.y_val, patch_sizes)

    def train_val_test(self, df, label_key, image_path_key, limit):
        LABEL_KEY = label_key
        SAMPLE_SIZE = limit

        labels = df[LABEL_KEY].unique()
        dfs = []
        for label in labels:
            sub_df = df[df[LABEL_KEY] == label]
            if len(sub_df) <= SAMPLE_SIZE:
                dfs.append(sub_df)
            else:
                dfs.append(sub_df.sample(n=SAMPLE_SIZE))

        df = pd.concat(dfs)

        X = []
        y = []

        for index, row in df.iterrows():
            X.append(row[image_path_key])
            y.append(row[LABEL_KEY])

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        from sklearn.utils import class_weight
        class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y)
        class_weight_dict = dict(enumerate(class_weight))

        onehot_y = np.zeros((len(y), len(le.classes_)), dtype="float16")
        for i, label_index in enumerate(y):
            onehot_y[i, label_index] = 1.

        y = onehot_y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test, le.classes_, class_weight_dict

    def package_from_dataframe(self, df, label_key, image_path_key, category_limit, save_path=None):
        X_train, X_val, X_test, y_train, y_val, y_test, classes, class_weight_dict = self.train_val_test(df,
                                                                                                    label_key=label_key,
                                                                                                    image_path_key=image_path_key,
                                                                                                    limit=category_limit)

        if save_path is not None:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            self.pickle_objects(save_path,
                           [X_train, X_val, X_test, y_train, y_val, y_test, classes, class_weight_dict],
                           ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "classes", "class_weight_dict"])
            self.save_labels(save_path, classes)

        return X_train, X_val, X_test, y_train, y_val, y_test, classes, class_weight_dict

    def package_from_csv(self, csv_file, label_key, image_path_key, category_limit, save_path=None):
        all_photos = pd.read_csv(csv_file)
        return self.package_from_dataframe(all_photos, label_key=label_key, image_path_key=image_path_key,
                                      category_limit=category_limit, save_path=save_path)

    def package_from_sqlite(self, sqlite_file, query, label_key, image_path_key, category_limit, save_path=None):
        con = sqlite3.connect(sqlite_file)
        all_photos = pd.read_sql_query(query, con)
        con.close()

        return self.package_from_dataframe(all_photos, label_key=label_key, image_path_key=image_path_key,
                                      category_limit=category_limit, save_path=save_path)

    def pickle_objects(self, destination_path, objects_to_save, filenames):
        for index, item_to_save in enumerate(objects_to_save):
            pickle.dump(item_to_save, open(os.path.join(destination_path, str(filenames[index]) + ".p"), "wb"))

    def save_labels(self, destination_path, classes):
        with open(os.path.join(destination_path, "labels.txt"), 'w') as file_handler:
            for item in classes:
                file_handler.write("{}\n".format(item))

    def load_saved_data(self, path):
        X_train = pickle.load(open(os.path.join(path ,"X_train.p"), "rb"))
        X_val = pickle.load(open(os.path.join(path, "X_val.p"), "rb"))
        X_test = None#pickle.load(open(os.path.join(path, "X_test.p"), "rb"))
        y_train = pickle.load(open(os.path.join(path, "y_train.p"), "rb"))
        y_val = pickle.load(open(os.path.join(path, "y_val.p"), "rb"))
        y_test = None #pickle.load(open(os.path.join(path, "y_test.p"), "rb"))
        classes = pickle.load(open(os.path.join(path, "classes.p"), "rb"))
        class_weight_dict = pickle.load(open(os.path.join(path, "class_weight_dict.p"), "rb"))

        return X_train, X_val, X_test, y_train, y_val, y_test, classes, class_weight_dict

    def make_train_generator(self, X_train, y_train, patch_sizes):

        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=self.preprocess_img)

        train_generator = MultiLabelDirectoryIterator(
            X_train, y_train, train_datagen,
            target_size=(self.IMG_WIDTH, self.IMG_HEIGHT),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical')
        # save_to_dir="./augmentedsamples")

        return train_generator

    def make_val_generator(self, X_val, y_val, patch_sizes):

        val_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_img)

        val_generator = MultiLabelDirectoryIterator(
            X_val, y_val, val_datagen,
            target_size=(self.IMG_WIDTH, self.IMG_HEIGHT),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical')
        # save_to_dir="./augmentedsamples")

        return val_generator

    def calculate_mean_image(self, X_train):
        mean_image = utils.calculate_mean_image_from_file_list(X_train)
        im = Image.fromarray(mean_image)
        im.save(os.path.join(self.save_path, "mean_image.jpg"))
        return np.array(im, dtype=np.float)

    def load_mean_image(self, filepath):
        mean_image = Image.open(os.path.join(filepath, "mean_image.jpg"))
        return mean_image

    def preprocess_img(self, img):
        img -= self.mean_image
        return img


class KerasFullImageDataset:
    SQLITE = "SQLITE"
    CSV = "CSV"

    def __init__(self,
                 filepath,
                 label_key,
                 image_path_key,
                 point_x_key,
                 point_y_key,
                 category_limit=10000000,
                 query=None,
                 save_path=None,
                 img_width=256,
                 img_height=256,
                 batch_size=16,
                 patch_sizes=[]):

        self.save_path = save_path
        self.IMG_WIDTH = img_width
        self.IMG_HEIGHT = img_height
        self.BATCH_SIZE = batch_size


        if ".sqlite" in filepath:
            self.X_train, \
            self.X_val, \
            self.X_test, \
            self.y_train, \
            self.y_val, \
            self.y_test, \
            self.classes, \
            self.class_weight_dict = self.package_from_sqlite(filepath, query, label_key, image_path_key, point_x_key, point_y_key, category_limit, save_path)
            self.mean_image = self.calculate_mean_image(self.X_train)
        elif ".csv" in filepath:
            self.X_train, \
            self.X_val, \
            self.X_test, \
            self.y_train, \
            self.y_val, \
            self.y_test, \
            self.classes, \
            self.class_weight_dict = self.package_from_csv(filepath, label_key, image_path_key, point_x_key, point_y_key, category_limit, save_path)
            self.mean_image = self.calculate_mean_image(self.X_train)
        else:
            self.X_train, \
            self.X_val, \
            self.X_test, \
            self.y_train, \
            self.y_val, \
            self.y_test, \
            self.classes, \
            self.class_weight_dict = self.load_saved_data(filepath)
            self.mean_image = self.load_mean_image(filepath)

        self.training = self.make_train_generator(self.X_train, self.y_train, patch_sizes)
        self.validation = self.make_val_generator(self.X_val, self.y_val, patch_sizes)

    def train_val_test(self, df, label_key, image_path_key, point_x_key, point_y_key, limit):
        LABEL_KEY = label_key
        SAMPLE_SIZE = limit

        labels = df[LABEL_KEY].unique()
        dfs = []
        for label in labels:
            sub_df = df[df[LABEL_KEY] == label]
            if len(sub_df) <= SAMPLE_SIZE:
                dfs.append(sub_df)
            else:
                dfs.append(sub_df.sample(n=SAMPLE_SIZE))

        df = pd.concat(dfs)

        X = []
        y = []

        for index, row in df.iterrows():
            X.append({"image_path": row[image_path_key], "point_x": row[point_x_key], "point_y": row[point_y_key]})
            y.append(row[LABEL_KEY])

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        from sklearn.utils import class_weight
        class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y)
        class_weight_dict = dict(enumerate(class_weight))

        onehot_y = np.zeros((len(y), len(le.classes_)), dtype="float16")
        for i, label_index in enumerate(y):
            onehot_y[i, label_index] = 1.

        y = onehot_y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test, le.classes_, class_weight_dict

    def package_from_dataframe(self, df, label_key, image_path_key, point_x_key, point_y_key, category_limit, save_path=None):
        X_train, X_val, X_test, y_train, y_val, y_test, classes, class_weight_dict = self.train_val_test(df,
                                                                                                    label_key=label_key,
                                                                                                    point_x_key=point_x_key,
                                                                                                    point_y_key=point_y_key,
                                                                                                    image_path_key=image_path_key,
                                                                                                    limit=category_limit)

        if save_path is not None:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            self.pickle_objects(save_path,
                           [X_train, X_val, X_test, y_train, y_val, y_test, classes, class_weight_dict],
                           ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "classes", "class_weight_dict"])
            self.save_labels(save_path, classes)

        return X_train, X_val, X_test, y_train, y_val, y_test, classes, class_weight_dict

    def package_from_csv(self, csv_file, label_key, image_path_key, point_x_key, point_y_key, category_limit, save_path=None):
        all_photos = pd.read_csv(csv_file)
        return self.package_from_dataframe(all_photos, label_key=label_key, image_path_key=image_path_key, point_x_key=point_x_key, point_y_key=point_y_key,
                                      category_limit=category_limit, save_path=save_path)

    def package_from_sqlite(self, sqlite_file, query, label_key, image_path_key, point_x_key, point_y_key, category_limit, save_path=None):
        con = sqlite3.connect(sqlite_file)
        all_photos = pd.read_sql_query(query, con)
        con.close()

        return self.package_from_dataframe(all_photos, label_key=label_key, image_path_key=image_path_key, point_x_key=point_x_key, point_y_key=point_y_key,
                                      category_limit=category_limit, save_path=save_path)

    def pickle_objects(self, destination_path, objects_to_save, filenames):
        for index, item_to_save in enumerate(objects_to_save):
            pickle.dump(item_to_save, open(os.path.join(destination_path, str(filenames[index]) + ".p"), "wb"))

    def save_labels(self, destination_path, classes):
        with open(os.path.join(destination_path, "labels.txt"), 'w') as file_handler:
            for item in classes:
                file_handler.write("{}\n".format(item))

    def load_saved_data(self, path):
        X_train = pickle.load(open(os.path.join(path ,"X_train.p"), "rb"))
        X_val = pickle.load(open(os.path.join(path, "X_val.p"), "rb"))
        X_test = None#pickle.load(open(os.path.join(path, "X_test.p"), "rb"))
        y_train = pickle.load(open(os.path.join(path, "y_train.p"), "rb"))
        y_val = pickle.load(open(os.path.join(path, "y_val.p"), "rb"))
        y_test = None #pickle.load(open(os.path.join(path, "y_test.p"), "rb"))
        classes = pickle.load(open(os.path.join(path, "classes.p"), "rb"))
        class_weight_dict = pickle.load(open(os.path.join(path, "class_weight_dict.p"), "rb"))

        return X_train, X_val, X_test, y_train, y_val, y_test, classes, class_weight_dict

    def the_generator(self, X_train, y_train, batch_size, cropping_function):
        nb_train_samples = len(X_train)
        while True:
            for start in range(0, nb_train_samples, batch_size):
                x_batch = []
                y_batch = []

                end = min(start + batch_size, nb_train_samples)
                for index in range(start, end):
                    patch = cropping_function(X_train[index])
                    y = y_train[index]

                    x_batch.append(patch)
                    y_batch.append(y)

                yield (np.array(x_batch), np.array(y_batch))

    def make_train_generator(self, X_train, y_train, patch_sizes):

        train_generator = FullImagePointCroppingLoader(X_train, y_train,
        #train_generator = self.the_generator(X_train, y_train,
                                                       self.BATCH_SIZE,
                                                       self.cropping_function)

        return train_generator

    def cropping_function(self, image_dict):
        patch = utils.load_image_and_crop_o(image_dict["image_path"], image_dict["point_x"], image_dict["point_y"], 256, 256)
        patch = tf.keras.preprocessing.image.img_to_array(patch)
        patch = self.preprocess_img(patch)

        return patch

    '''
    def cropping_function(self, image_dict):
        # randomly select crop ratio
        height_ratio = random.choice([7, 7.5, 8, 8.5, 9])

        # randomly jitter crop center
        jitter = random.choice([-0.01, -0.02, -0.03, -0.04, -0.05, 0.05, 0.04, 0.03, 0.02, 0.01])
        point_x = image_dict["point_x"] + jitter
        point_y = image_dict["point_y"] + jitter

        #patch = utils.load_image_and_crop_o(image_dict["image_path"], image_dict["point_x"], image_dict["point_y"], 256, 256)
        patch = utils.load_image_and_crop_ratio(image_dict["image_path"], point_x, point_y, 256, 256, height_ratio)
        patch = tf.keras.preprocessing.image.img_to_array(patch)
        patch = self.preprocess_img(patch)

        return patch
    '''

    def make_val_generator(self, X_val, y_val, patch_sizes):

        val_generator = FullImagePointCroppingLoader(X_val, y_val,
        #val_generator = self.the_generator(X_val, y_val,
                                                       self.BATCH_SIZE,
                                                       self.cropping_function)

        return val_generator

    def calculate_mean_image(self, X_train):
        mean_image = utils.calculate_mean_image_from_crop_file_list(sample(X_train, 100))
        im = Image.fromarray(mean_image)
        im.save(os.path.join(self.save_path, "mean_image.jpg"))
        return np.array(im, dtype=np.float)

    def load_mean_image(self, filepath):
        mean_image = Image.open(os.path.join(filepath, "mean_image.jpg"))
        return mean_image

    def preprocess_img(self, img):
        img -= self.mean_image

        #img = ImFeelingLucky().white_balance_pil_image(keras.preprocessing.image.array_to_img(img))
        #img = ImFeelingLucky().white_balance_img_array(img)

        # normalise for faster training times
        img /= 255.0



        return img
