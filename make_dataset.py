#to hold all config variables
import os

from datagenerators import KerasDataset

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
    make_dataset("Trip1-GROUP_DESC-ensemble-data")