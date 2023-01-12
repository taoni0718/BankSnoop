import random
import numpy as np
import glob
import cv2 as cv
import tensorflow as tf


class MAMLDataLoaderMag:

    def __init__(self, data_path, batch_size, n_way, k_shot, q_query):
        """
        MAML data loader
        :param data_path: data path, classes in different categories
        :param batch_size: number of tasks
        :param n_way: number of classes in a task
        :param k_shot: number of samples for inner loop training in one class
        :param q_query: number of samples for outer loop training in one class
        """
        #self.file_list = [f for f in glob.glob(data_path + "**/character*", recursive=True)]
        self.file_list = data_path
        self.steps = len(self.file_list) // batch_size

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.meta_batch_size = batch_size

    def __len__(self):
        return self.steps
    
    def get_one_task_data_mag(self):
        """
        obtain a task: n_way classes, each class has k_shot for inner training, q_query for outer training.
        return: support_data, query_data
        """
        mag_dirs = random.sample(self.file_list, self.n_way)
        support_data = []
        query_data = []

        support_mag = []
        support_label = []
        query_mag = []
        query_label = []
        #print(mag_dirs)
        
        for label, mag_dir in enumerate(mag_dirs):
            mag_list = [i for i in idx_dict[mag_dir]]
            mags = random.sample(mag_list,  min(len(mag_list), self.k_shot + self.q_query))

            #print(mag_dir)
            #print(mags)
            #print(label)
            # Read support set
            for mag_idx in mags[:self.k_shot]:
                magdata = get_magdata_idx(Xtu, mag_idx)
                #image = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                #image = image / 255.0
                #image = np.reshape(image, (28,28,1))
                #image = image.reshape((28, 28, 1))
                #image = cv.resize(image, (28, 28))
                #image = np.expand_dims(image, axis=-1).astype('float32')
                support_data.append((magdata, label))

            # Read query set
            for mag_idx in mags[self.k_shot:]:
                magdata = get_magdata_idx(Xtu, mag_idx)
                #image = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                #image = image / 255.0
                #image = image.reshape((28, 28, 1))
                #image = cv.resize(image, (28, 28))
                #image = np.expand_dims(image, axis=-1).astype('float32')
                #image = np.expand_dims(image, axis=-1).astype('float32')
                query_data.append((magdata, label))

        # shuffle support set
        random.shuffle(support_data)
        for data in support_data:
            support_mag.append(data[0])
            support_label.append(data[1])

        # shuffle query set
        random.shuffle(query_data)
        for data in query_data:
            query_mag.append(data[0])
            query_label.append(data[1])
            
        return np.array(support_mag), np.array(support_label), np.array(query_mag), np.array(query_label)
    
    def get_one_batch_mag(self):
        """
        get samples from one batch
        :return: k_shot_data, q_query_data
        """
        while True:
            batch_support_mag = []
            batch_support_label = []
            batch_query_mag = []
            batch_query_label = []

            for _ in range(self.meta_batch_size):
                support_mag, support_label, query_mag, query_label = self.get_one_task_data_mag()
                batch_support_mag.append(support_mag)
                batch_support_label.append(support_label)
                batch_query_mag.append(query_mag)
                batch_query_label.append(query_label)

            yield np.array(batch_support_mag), np.array(batch_support_label), np.array(batch_query_mag), np.array(batch_query_label)
        

    def get_one_task_data(self):
        """
        get one task, which contains n_way classes, where k_shot for inner training, q_query for outer training
        :return: support_data, query_data
        """
        img_dirs = random.sample(self.file_list, self.n_way)
        support_data = []
        query_data = []

        support_image = []
        support_label = []
        query_image = []
        query_label = []
        #print(img_dirs)

        for label, img_dir in enumerate(img_dirs):
            img_list = [f for f in glob.glob(img_dir + "**/*.png", recursive=True)]
            #img_list = img_dirs
            images = random.sample(img_list, self.k_shot + self.q_query)

            #print(img_dir)
            #print(images)
            #print(label)
            # Read support set
            for img_path in images[:self.k_shot]:
                image = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                image = image / 255.0
                #image = np.reshape(image, (28,28,1))
                #image = image.reshape((28, 28, 1))
                image = cv.resize(image, (28, 28))
                image = np.expand_dims(image, axis=-1).astype('float32')
                support_data.append((image, label))

            # Read query set
            for img_path in images[self.k_shot:]:
                image = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                image = image / 255.0
                #image = image.reshape((28, 28, 1))
                image = cv.resize(image, (28, 28))
                image = np.expand_dims(image, axis=-1).astype('float32')
                #image = np.expand_dims(image, axis=-1).astype('float32')
                query_data.append((image, label))

        # shuffle support set
        random.shuffle(support_data)
        for data in support_data:
            support_image.append(data[0])
            support_label.append(data[1])

        # shuffle query set
        random.shuffle(query_data)
        for data in query_data:
            query_image.append(data[0])
            query_label.append(data[1])
            
        return np.array(support_image), np.array(support_label), np.array(query_image), np.array(query_label)

    def get_one_batch(self):
        """
        get samples from one batch
        :return: k_shot_data, q_query_data
        """

        while True:
            batch_support_image = []
            batch_support_label = []
            batch_query_image = []
            batch_query_label = []

            for _ in range(self.meta_batch_size):
                support_image, support_label, query_image, query_label = self.get_one_task_data()
                batch_support_image.append(support_image)
                batch_support_label.append(support_label)
                batch_query_image.append(query_image)
                batch_query_label.append(query_label)

            yield np.array(batch_support_image), np.array(batch_support_label), \
                  np.array(batch_query_image), np.array(batch_query_label)