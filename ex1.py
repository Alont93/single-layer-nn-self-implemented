import numpy as np
import pandas as pd

from dataloader import load_data, display_face
from pca import *

EMOTIONS = ['h','ht','m','s','f','a','d','n']
PCA_NUMBER_OF_COMPONENT = 6
TRAINING_PERCENTAGE = 0.6
VALIDATION_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.2


def simplify_labels(filename):
    labels = find_between(filename, '_', '.')
    return labels[:-1]

def find_between(s, first, last):
    start = s.index(first) + len(first)
    end = s.index(last, start)
    return s[start:end]

def import_data():
    images, labels = load_data()
    for i in range(len(labels)):
        labels[i] = simplify_labels(labels[i])

    return np.array(images), np.array(labels)

def different_emotions_figure():
    images, labels = import_data()

    for emotion in EMOTIONS:
        first_image_index = labels.index(emotion)
        emotion_image = images[first_image_index]
        display_face(emotion_image)

def display_pca_conponents():
    images, labels = import_data()

    pca = PCA(PCA_NUMBER_OF_COMPONENT)
    pca.fit(images)

    pca_images = np.empty(((images.size, 1, PCA_NUMBER_OF_COMPONENT)))
    transformed_images = np.empty(images.shape)

    for i in range(images.shape[0]):
        pca_images[i] =  pca.transform(images[i])
        transformed_images[i] = pca.inverse_transform(pca_images[i])

    pca.display('')


def arrange_data_set_for_emotions(emo1, emo0):
    images, labels = import_data()

    indices_of_emo1 = np.where(labels == emo1)[0]
    indices_of_emo0 = np.where(labels == emo0)[0]

    np.random.shuffle(indices_of_emo1)
    np.random.shuffle(indices_of_emo0)

    indices_of_wanted_emotions = np.vstack((indices_of_emo1,indices_of_emo0)).ravel('F')

    training_ratio = int(indices_of_wanted_emotions.size * TRAINING_PERCENTAGE)
    validation_ratio = int(indices_of_wanted_emotions.size * (TRAINING_PERCENTAGE + VALIDATION_PERCENTAGE))
    # test_ratio = indices_of_wanted_emotions.size * (TRAINING_PERCENTAGE + VALIDATION_PERCENTAGE+ TEST_PERCENTAGE)

    relevant_images = images[indices_of_wanted_emotions]
    relevant_labels = labels[indices_of_wanted_emotions]

    relevant_labels[relevant_labels == emo1] = 1
    relevant_labels[relevant_labels == emo0] = 0

    training_images = relevant_images[:training_ratio]
    training_labels = relevant_labels[:training_ratio]

    validation_images = relevant_images[training_ratio: validation_ratio]
    validation_labels = relevant_labels[training_ratio: validation_ratio]

    test_images = relevant_images[validation_ratio:]
    test_labels = relevant_labels[validation_ratio:]

    return training_images, validation_images, test_images, training_labels, validation_labels, test_labels


def gaussian_pdf(sample, mean, std):
    var = std ** 2
    denom = (2 * np.pi * var) ** .5
    num = np.exp(-(sample - mean) ** 2 / (2 * var))
    return num/denom


def logistic_regression(image, weights, mean1, std1, mean0, std0, data_ratio):
    a = np.log((gaussian_pdf(image, mean1, std1)) * data_ratio / (gaussian_pdf(image, mean0, std0)) * (1- data_ratio))
    return 1 / (1 + np.exp(-a))


def train_data():
    training_images, validation_images, test_images, training_labels, validation_labels, test_labels = arrage_data_set_for_emotions('h','m')
