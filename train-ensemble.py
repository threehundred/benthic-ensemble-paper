import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print(gpu_devices)
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

import sys
import gc
import os
import numpy as np


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint
import models
from datagenerators import KerasDataset, KerasFullImageDataset

#to hold all config variables
class Config:
    BATCH_SIZE = 32
    IMG_WIDTH, IMG_HEIGHT = 256, 256
    IMG_CHANNELS = 3
    DATA_AUGMENTATION = True
    LEARNING_RATE = 0.01
    EPOCHS = 500

# set up some folder structures
def init_project(run_name):

    # run name unique for each new run
    RUN_NAME = run_name

    # make logs file
    Config.LOGS = './logs/' + RUN_NAME + "/"

    if not os.path.isdir(Config.LOGS):
        os.makedirs(Config.LOGS)
    else:
        print("Run name exists - use another one.")
        exit(0)

    # make checkpoints path
    Config.CHECKPOINTS_FOLDER = Config.LOGS + "/checkpoints-%s/" % RUN_NAME
    Config.CHECKPOINTS = Config.CHECKPOINTS_FOLDER + "weights.best.hdf5"

    if not os.path.isdir(Config.CHECKPOINTS_FOLDER):
        os.mkdir(Config.CHECKPOINTS_FOLDER)

    # static settings
    Config.MEAN_IMAGE_PATH = Config.LOGS + "mean_image.jpg"


def get_existing_data(path):

    dataset = KerasDataset(path,
                            Config.LABEL_KEY,
                            Config.IMG_PATH_KEY,
                            img_width=Config.IMG_WIDTH,
                            img_height=Config.IMG_HEIGHT,
                            batch_size=Config.BATCH_SIZE,
                            save_path=Config.LOGS,
                            patch_sizes=Config.PATCH_SIZES,
                            category_limit=Config.CATEGORY_LIMIT)

    Config.NB_TRAIN_SAMPLES = len(dataset.X_train)
    Config.NB_VALIDATION_SAMPLES = len(dataset.X_val)
    Config.NB_CLASSES = len(dataset.classes)

    return dataset


def get_data():

    dataset = KerasDataset(Config.DATA_FILE,
                    Config.LABEL_KEY,
                    Config.IMG_PATH_KEY,
                    query=Config.QUERY,
                    img_width=Config.IMG_WIDTH,
                    img_height=Config.IMG_HEIGHT,
                    batch_size=Config.BATCH_SIZE,
                    save_path=Config.LOGS,
                    patch_sizes=Config.PATCH_SIZES,
                    category_limit=Config.CATEGORY_LIMIT)

    Config.NB_TRAIN_SAMPLES = len(dataset.X_train)
    Config.NB_VALIDATION_SAMPLES = len(dataset.X_val)
    Config.NB_CLASSES = len(dataset.classes)

    return dataset


def get_full_image_data():
    """
    Seprate function to get the data in full image format with the point locations in a dict, so that we
    can dynamically crop the images on the fly to generate multiple fields of view
    :return:
    """

    dataset = KerasFullImageDataset(Config.DATA_FILE,
                    Config.LABEL_KEY,
                    Config.IMG_PATH_KEY,
                    Config.POINT_X_KEY,
                    Config.POINT_Y_KEY,
                    query=Config.QUERY,
                    img_width=Config.IMG_WIDTH,
                    img_height=Config.IMG_HEIGHT,
                    batch_size=Config.BATCH_SIZE,
                    save_path=Config.LOGS,
                    patch_sizes=Config.PATCH_SIZES,
                    category_limit=Config.CATEGORY_LIMIT)

    Config.NB_TRAIN_SAMPLES = len(dataset.X_train)
    Config.NB_VALIDATION_SAMPLES = len(dataset.X_val)
    Config.NB_CLASSES = len(dataset.classes)

    return dataset


def trip1_ensemble(run_name, data_path):
    RUN_NAME = run_name
    CUSTOM_WEIGHTS = "./weights/weights.best.hdf5"

    # multi patch config
    Config.PATCH_SIZES = []  # [128, 64]
    Config.LABEL_KEY = "label"
    Config.IMG_PATH_KEY = "path"
    #Config.POINT_X_KEY = "POINT_X"
    #Config.POINT_Y_KEY = "POINT_Y"
    Config.DATA_FILE = data_path
    Config.QUERY = ""
    Config.CATEGORY_LIMIT = 100000

    init_project(RUN_NAME)
    dataset = get_existing_data(data_path)

    model = models.densenet(Config.NB_CLASSES, Config.IMG_WIDTH, Config.IMG_HEIGHT, Config.IMG_CHANNELS,
                            custom_weights=CUSTOM_WEIGHTS)

    return dataset, model

def make_dataset(run_name):
    RUN_NAME = run_name
    Config.PATCH_SIZES = []  # [128, 64]
    Config.LABEL_KEY = "label"
    Config.IMG_PATH_KEY = "path"
    #Config.POINT_X_KEY = "POINT_X"
    #Config.POINT_Y_KEY = "POINT_Y"
    Config.DATA_FILE = "./data/data.sqlite"
    Config.QUERY = "SELECT * from 'benthic-ensemble-paper'"
    Config.CATEGORY_LIMIT = 100000

    init_project(run_name)
    dataset = get_data()


if __name__ == "__main__":

    #make_dataset("Trip1-GROUP_DESC-ensemble-data")
    #exit(0)

    if len(sys.argv) < 3:
        print("Error! No run name or data path specified!")
        print("Usage: %s run_name data_path" % sys.argv[0])
        sys.exit(1)

    run_name = sys.argv[1]
    data_path = sys.argv[2]

    # train
    sgd = tf.keras.optimizers.SGD(lr=Config.LEARNING_RATE)

    dataset, model = trip1_ensemble(run_name, data_path)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'], )
    # options=run_options, run_metadata=run_metadata)

    tb = tf.keras.callbacks.TensorBoard(log_dir=Config.LOGS, histogram_freq=0, batch_size=Config.BATCH_SIZE,
                                        write_graph=True,
                                        write_grads=True, write_images=True)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10)
    csv_logger = CSVLogger(Config.LOGS + '/csvlogs.csv')
    checkpoint = ModelCheckpoint(Config.CHECKPOINTS, monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')

    model.fit_generator(
        dataset.training,
        steps_per_epoch=Config.NB_TRAIN_SAMPLES // Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=dataset.validation,
        validation_steps=Config.NB_VALIDATION_SAMPLES // Config.BATCH_SIZE,
        callbacks=[lr_reducer, csv_logger, tb, checkpoint, early_stopper],
        class_weight=dataset.class_weight_dict,
        verbose=1,
        # use_multiprocessing=True,
        # max_queue_size=3,
        workers=10)
