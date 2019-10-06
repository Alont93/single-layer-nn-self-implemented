import numpy as np
import pandas as pd

from dataloader import load_data, display_face
from pca import *

EMOTIONS = ['h','ht','m','s','f','a','d','n']
PCA_NUMBER_OF_COMPONENT = 6
LEARNING_RATE = 1
GRADIENT_DECENT_ITERATION = 10
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

    relevant_labels = relevant_labels.astype(np.int)

    training_images = relevant_images[:training_ratio]
    training_labels = relevant_labels[:training_ratio]

    validation_images = relevant_images[training_ratio: validation_ratio]
    validation_labels = relevant_labels[training_ratio: validation_ratio]

    test_images = relevant_images[validation_ratio:]
    test_labels = relevant_labels[validation_ratio:]

    return training_images, validation_images, test_images, training_labels, validation_labels, test_labels


def logistic_regression(samples, weights):
    a = samples @ weights
    p = 1 / (1 + np.exp(-a))

    return p


def batch_gradient_decent(samples, validation_samples, validation_labels, labels):
    current_weights = np.zeros(PCA_NUMBER_OF_COMPONENT + 1)
    best_loss = get_weights_loss(validation_samples, validation_labels, current_weights)
    best_weights = current_weights

    for t in range(GRADIENT_DECENT_ITERATION):
        logistic_results = logistic_regression(samples, current_weights)
        loss_gradient = (labels - logistic_results) @ samples
        current_weights = current_weights + LEARNING_RATE * loss_gradient

        current_loss = get_weights_loss(validation_samples, validation_labels, current_weights)
        if best_loss > current_loss:
            best_loss = current_loss
            best_weights = current_weights

    return current_weights


def run_pca_on_samples(pca, images):
    number_of_images = images.shape[0]
    pca_images = np.empty(((number_of_images, 1, PCA_NUMBER_OF_COMPONENT)))

    for i in range(number_of_images):
        pca_images[i] = pca.transform(images[i])

    return pca_images


def add_bias_coordinate(pca, pca_images):
    number_of_images = pca_images.shape[0]
    pca_images = pca_images.reshape((number_of_images, PCA_NUMBER_OF_COMPONENT))
    bias_coordinates = np.ones((number_of_images, 1))
    return np.hstack((pca_images,bias_coordinates))

def regression_loss(labels, prediction):
    return -np.mean(labels * np.log(prediction) + (1-labels) * np.log(1-prediction))

def train_data():
    training_images, validation_images, test_images, \
    training_labels, validation_labels, test_labels = arrange_data_set_for_emotions('h','m')
    
    pca = PCA(PCA_NUMBER_OF_COMPONENT)
    pca.fit(training_images)

    training_pca_images = run_pca_on_samples(pca, training_images)
    validation_pca_images = run_pca_on_samples(pca, validation_images)
    test_pca_images = run_pca_on_samples(pca, test_images)

    training_pca_images = add_bias_coordinate(pca, training_pca_images)
    validation_pca_images = add_bias_coordinate(pca, validation_pca_images)
    test_pca_images = add_bias_coordinate(pca, test_pca_images)
    weights = batch_gradient_decent(training_pca_images, validation_pca_images, validation_labels, training_labels)

    loss = get_weights_loss(test_pca_images, test_labels, weights)
    x = 1


def get_weights_loss(pca_images, labels, weights):
    prediction = logistic_regression(pca_images, weights)
    return regression_loss(labels, prediction)


train_data()