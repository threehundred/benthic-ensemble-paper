import tensorflow as tf

from datagenerators import KerasDataset

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
tf.compat.v1.disable_eager_execution()
print("----------------------------->>>>>",  tf.executing_eagerly())
tf.config.threading.set_intra_op_parallelism_threads(1)

from PIL import Image, ImageFilter
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import load_model
import os.path as osp
import utils
from tensorflow.keras.utils import Sequence
import sys

class ImageLoader(Sequence):

    def __init__(self, x_set, y_set, batch_size, load_image_func):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.load_image_func = load_image_func

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        (idx * self.batch_size, (idx + 1) * self.batch_size, int(np.ceil(len(self.x) / float(self.batch_size))))

        indexes = []
        for index in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            indexes.append(index)

        patches = [self.load_image_func(dict_item) for dict_item in batch_x]

        return np.array(patches), np.array(batch_y)

def load_model_weights(weights_path, mean_image_path, labels_path):

    mean_image = Image.open(mean_image_path)
    mean_image = tf.keras.preprocessing.image.img_to_array(mean_image)

    with open(labels_path) as f:
        labels = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    labels = [x.strip() for x in labels]

    model = load_model(weights_path)

    return model, mean_image, labels


def load_val_data(val_data_x_path, val_data_y_path):
    X = pickle.load(open(val_data_x_path, "rb"))
    y = pickle.load(open(val_data_y_path, "rb"))

    return X, y

def load_data(data_file_path, query):

    # multi patch config
    PATCH_SIZES = []  # [128, 64]
    LABEL_KEY = "label"
    IMG_PATH_KEY = "path"
    #POINT_X_KEY = "POINT_X"
    #POINT_Y_KEY = "POINT_Y"
    DATA_FILE = data_file_path
    CATEGORY_LIMIT = 100000

    dataset = KerasDataset(DATA_FILE,
                                    LABEL_KEY,
                                    IMG_PATH_KEY,
                                    img_width=256,
                                    img_height=256,
                                    batch_size=48,
                                    save_path="../logs/atemplogs/",
                                    patch_sizes=PATCH_SIZES,
                                    category_limit=CATEGORY_LIMIT)

    return dataset.X, dataset.y


def blur_aug(patch, aug_number):
    patch_blur = tf.keras.preprocessing.image.array_to_img(patch)
    patch_blur = patch_blur.filter(ImageFilter.GaussianBlur(radius=(aug_number * 0.5)))
    patch_blur = tf.keras.preprocessing.image.img_to_array(patch_blur)
    return patch_blur


def colour_aug(patch, aug_number):
    # patch = tf.keras.preprocessing.image.array_to_img(patch)
    # patch_col = np.asarray(patch, dtype=np.float)

    patch_col = np.copy(patch)
    patch_col[..., 0] *= (1 - (aug_number * 0.1))
    #print(1 - (aug_number * 0.1))
    aug_img = tf.keras.preprocessing.image.img_to_array(patch_col)
    return aug_img


def inference(model, X, y, mean_image, augmentation, aug_number):

    def cropping_function(image_path):

        patch = Image.open(image_path)
        patch = tf.keras.preprocessing.image.img_to_array(patch)

        patch -= mean_image

        if augmentation == 'colour':
            patch = colour_aug(patch, aug_number)

        if augmentation == 'blur':
            patch = colour_aug(patch, aug_number)

        patch /= 255.0
        return patch

    results = []

    print("calling val gen")
    val_generator = ImageLoader(X, y,
                                 96,
                                 cropping_function)


    for i in range(15):
        print("setting up predict gen")
        predictions = model.predict(val_generator,
                                    steps=(len(y) // 96) + 1,
                                    verbose=1,
                                    max_queue_size=10,
                                    workers=10,
                                    use_multiprocessing=False)

        # print(predictions)
        print(len(predictions), len(X), len(y))

        '''
        predictions = np.array(predictions).round(2)
        for index, item in enumerate(predictions):
            predicted_index = np.argmax(predictions[index])
            y_index = np.argmax(y[index])
            prediction_score = predictions[index][predicted_index]
            prediction_label = labels[predicted_index]
            true_label = labels[y_index]

            print(prediction_label, true_label, prediction_score)
        '''
        print("appending")
        results.append(predictions)

    return results

def save(X_out_filename, y_out_filename, X_result, y_result):
    pickle.dump(X_result, open(X_out_filename, "wb"))
    pickle.dump(y_result, open(y_out_filename, "wb"))


if __name__ == '__main__':

    if len(sys.argv) < 10:
        print("Error! No run name or data path specified!")
        print("Usage: %s run_name data_path" % sys.argv[0])
        sys.exit(1)

    weights_path = sys.argv[1]
    mean_image_path = sys.argv[2]
    labels_path = sys.argv[3]

    val_data_x_path = sys.argv[4]
    val_data_y_path = sys.argv[5]

    data_file_path = sys.argv[6]
    query = sys.argv[7]

    augmentation = sys.argv[8]
    aug_number = int(sys.argv[9])

    X_out_filename = sys.argv[10]
    y_out_filename = sys.argv[11]

    print(weights_path,
          mean_image_path,
          labels_path,
          val_data_x_path,
          val_data_y_path,
          data_file_path,
          query,
          augmentation,
          aug_number,
          X_out_filename,
          y_out_filename)


    model, mean_image, labels = load_model_weights(weights_path, mean_image_path, labels_path)

    if data_file_path != "" and data_file_path != None:
        X, y = load_data(data_file_path, query)
    else:
        X, y = load_val_data(val_data_x_path, val_data_y_path)

    results = inference(model, X, y, mean_image, augmentation, aug_number)
    save(X_out_filename, y_out_filename, results, y)

    print(weights_path,
          mean_image_path,
          labels_path,
          val_data_x_path,
          val_data_y_path,
          data_file_path,
          query,
          X_out_filename,
          y_out_filename)