import numpy as np
import pandas as pd

from dataloader import load_data, display_face
from pca import *
from ex1_two_class import *

RELEVANT_EMOTIONS = ['h', 'm', 's', 'f', 'a', 'd']
UNRELEVANT_EMOTIONS = ['ht', 'n']
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


def arrange_data_set_for_emotions():
    images, labels = import_data()


    data_table = pd.DataFrame({'emotion': labels})
    data_table = data_table[~data_table['emotion'].isin(UNRELEVANT_EMOTIONS)]
    data_table = pd.concat((data_table,pd.get_dummies(data_table.emotion)),1)
    # remove linear independent variable
    data_table.drop(data_table.columns[[-1,]], axis=1, inplace=True)


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


arrange_data_set_for_emotions()