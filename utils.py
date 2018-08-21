# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# utils.py
# Created on: 2018-08-13
# Author : Charles Tousignant
# Project : -
# Description : Utility functions
# ---------------------------------------------------------------------------
import os
import pickle
from os import listdir
from os.path import isfile, join
from random import shuffle
from shutil import copyfile
import cv2
import numpy as np
import tensorflow as tf


#######################################
# Helper funtions for data management #
#######################################
def separateData(data_dir):
    for filename in listdir(data_dir):
        if isfile(join(data_dir, filename)):
            tokens = filename.split('.')
            if tokens[-1] == 'jpg':
                image_path = join(data_dir, filename)
                if not os.path.exists(join(data_dir, tokens[0])):
                    os.makedirs(join(data_dir, tokens[0]))
                copyfile(image_path, join(join(data_dir, tokens[0]), filename))
                os.remove(image_path)


def resizeAndPad(img, size):
    h, w = img.shape[:2]
    sh, sw = size
    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC
    # aspect ratio of image
    aspect = w / h
    # padding
    if aspect > 1:  # horizontal image
        new_shape = list(img.shape)
        new_shape[0] = w
        new_shape[1] = w
        new_shape = tuple(new_shape)
        new_img = np.zeros(new_shape, dtype=np.uint8)
        h_offset = int((w - h) / 2)
        new_img[h_offset:h_offset + h, :, :] = img.copy()
    elif aspect < 1:  # vertical image
        new_shape = list(img.shape)
        new_shape[0] = h
        new_shape[1] = h
        new_shape = tuple(new_shape)
        new_img = np.zeros(new_shape, dtype=np.uint8)
        w_offset = int((h - w) / 2)
        new_img[:, w_offset:w_offset + w, :] = img.copy()
    else:
        new_img = img.copy()
    # scale and pad
    scaled_img = cv2.resize(new_img, size, interpolation=interp)
    return scaled_img


class DataSetGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_labels = self.get_data_labels()
        self.data_info = self.get_data_paths()

    def get_data_labels(self):
        data_labels = []
        for filename in listdir(self.data_dir):
            if not isfile(join(self.data_dir, filename)):
                data_labels.append(filename)
        return data_labels

    def get_data_paths(self):
        data_paths = []
        for label in self.data_labels:
            img_lists = []
            path = join(self.data_dir, label)
            for filename in listdir(path):
                tokens = filename.split('.')
                if tokens[-1] == 'jpg':
                    image_path = join(path, filename)
                    img_lists.append(image_path)
            shuffle(img_lists)
            data_paths.append(img_lists)
        return data_paths

    # to save the labels its optional in case you want to restore the names from the ids
    # and you forgot the names or the order it was generated
    def save_labels(self, path):
        pickle.dump(self.data_labels, open(path, "wb"))

    def get_mini_batches(self, batch_size=10, image_size=(200, 200), allchannel=True):
        images = []
        labels = []
        empty = False
        counter = 0
        each_batch_size = int(batch_size / len(self.data_info))
        while True:
            for i in range(len(self.data_labels)):
                label = np.zeros(len(self.data_labels), dtype=int)
                label[i] = 1
                if len(self.data_info[i]) < counter + 1:
                    empty = True
                    continue
                empty = False
                img = cv2.imread(self.data_info[i][counter])
                img = resizeAndPad(img, image_size)
                if not allchannel:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                images.append(img)
                labels.append(label)
            counter += 1

            if empty:
                break
            # if the iterator is multiple of batch size return the mini batch
            if counter % each_batch_size == 0:
                yield np.array(images, dtype=np.uint8), np.array(labels, dtype=np.uint8)
                del images
                del labels
                images = []
                labels = []


############################
# Helper Functions for CNN #
############################
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

# if __name__ == "__main__":
#     separateData("./train")
