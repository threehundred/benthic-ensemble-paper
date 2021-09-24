from tensorflow.keras.utils import Sequence
import numpy as np
import concurrent.futures
import time

class FullImagePointCroppingLoader(Sequence):

    def __init__(self, x_set, y_set, batch_size, load_image_func):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.load_image_func = load_image_func

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        indexes = []
        for index in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            indexes.append(index)



        #start = time.time()
        '''
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:

            # build the list of images to process
            future_to_patches = {executor.submit(self.load_image_func, dict_item): dict_item for dict_item in batch_x}

            loaded_patches = []

            for future in concurrent.futures.as_completed(future_to_patches):
                path = future_to_patches[future]
                try:
                    patch = future.result()
                    loaded_patches.append(patch)

                except Exception as exc:
                    print('%r generated an exception: %s' % (path, exc))
                #else:
                #    print('%r path is %d bytes' % (path, len(patch)))

            end = time.time()
            print("batch took", end - start)
            return np.array(loaded_patches), np.array(batch_y)
        '''
        patches = [self.load_image_func(dict_item) for dict_item in batch_x]
        #end = time.time()
        #print("batch took", end - start)

        return np.array(patches), np.array(batch_y)#, indexes

        #return np.array([
        #    resize(imread(file_name), (200, 200))
        #    for file_name in batch_x]), np.array(batch_y)


class FullImagePointCroppingSegmentationLoader(Sequence):

    def __init__(self, x_set, y_set, batch_size, img_width, img_height, load_image_func):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.load_image_func = load_image_func
        self.img_width = img_width
        self.img_height = img_height

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def generate_masks(self, width, height, batch_y):
        masks = []

        for y in batch_y:
            mask = [[y] * width] * height
            #masks.append( np.full((width, height), y) )
            masks.append(mask)

        return np.array(masks)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        indexes = []
        for index in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            indexes.append(index)

        patches = [self.load_image_func(dict_item) for dict_item in batch_x]
        masks = self.generate_masks(self.img_width, self.img_height, batch_y)

        return np.array(patches), masks