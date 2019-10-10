import numpy as np
import pandas as pd

from dataloader import load_data, display_face
from pca import *

EMOTIONS = ['h','ht','m','s','f','a','d','n']
PCA_NUMBER_OF_COMPONENT = 6
LEARNING_RATE = 1
EPOCHS = 10
TRAINING_PERCENTAGE = 0.6
VALIDATION_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.2
NUMBER_OF_EMOTIONS = 2
NUMBER_OF_SUBJECTS = 10
NUMBER_OF_RUNS = 5

EMOTION1 = 'h'
EMOTION0 = 'm'

EPOCHS_TO_INCLUDE_STD = [2, 4, 8, 10]

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
        pca_images[i] = pca.transform(images[i])
        transformed_images[i] = pca.inverse_transform(pca_images[i])

    pca.display('')


def arrange_data_set_for_emotions(emo1, emo0):
    images, labels = import_data()

    subjects_indices = get_shuffled_subject_indicies()

    relevant_images, relevant_labels = filter_only_two_emotions_by_subject_order(emo0, emo1, images, labels,
                                                                                 subjects_indices)
    # Break down the data into training, testing and validation
    training_ratio = int(subjects_indices.size * TRAINING_PERCENTAGE)
    validation_ratio = int(subjects_indices.size * (TRAINING_PERCENTAGE + VALIDATION_PERCENTAGE))

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


def filter_only_two_emotions_by_subject_order(emo0, emo1, images, labels, subjects_indices):
    indices_of_emo1 = np.where(labels == emo1)[0]
    indices_of_emo0 = np.where(labels == emo0)[0]
    indices_of_wanted_emotions = np.vstack((indices_of_emo1, indices_of_emo0)).ravel('F')
    relevant_images = images[indices_of_wanted_emotions][subjects_indices]
    relevant_labels = labels[indices_of_wanted_emotions][subjects_indices]
    return relevant_images, relevant_labels


def get_shuffled_subject_indicies():
    subjects_indices = np.arange(NUMBER_OF_SUBJECTS)
    np.random.shuffle(subjects_indices)
    subjects_indices = np.repeat(subjects_indices, NUMBER_OF_EMOTIONS)
    subjects_indices *= 2
    subjects_indices[1::2] += 1
    return subjects_indices


def logistic_regression(samples, weights):
    a = samples @ weights
    p = 1 / (1 + np.exp(-a))

    return p


def batch_gradient_decent(samples, labels, validation_samples, validation_labels):
    current_weights = np.zeros(PCA_NUMBER_OF_COMPONENT + 1)
    first_train_loss = get_weights_loss(samples, labels, current_weights)
    best_validation_loss = get_weights_loss(validation_samples, validation_labels, current_weights)
    best_weights = current_weights

    validation_errors = [best_validation_loss]
    training_errors = [first_train_loss]

    for t in range(EPOCHS):
        logistic_results = logistic_regression(samples, current_weights)
        loss_gradient = (labels - logistic_results) @ samples
        current_weights = current_weights + LEARNING_RATE * loss_gradient

        current_validation_loss = get_weights_loss(validation_samples, validation_labels, current_weights)
        current_train_loss = get_weights_loss(samples, labels, current_weights)
        validation_errors.append(current_validation_loss)
        training_errors.append(current_train_loss)
        if best_validation_loss > current_validation_loss:
            best_validation_loss = current_validation_loss
            best_weights = current_weights

    return best_weights, np.array(training_errors), np.array(validation_errors)


def run_pca_on_samples(pca, images):
    number_of_images = images.shape[0]
    pca_images = np.empty(((number_of_images, 1, PCA_NUMBER_OF_COMPONENT)))

    for i in range(number_of_images):
        pca_images[i] = pca.transform(images[i])

    return pca_images


def add_bias_coordinate(pca_images):
    number_of_images = pca_images.shape[0]
    pca_images = pca_images.reshape((number_of_images, PCA_NUMBER_OF_COMPONENT))
    bias_coordinates = np.ones((number_of_images, 1))
    return np.hstack((pca_images,bias_coordinates))


def regression_loss(labels, prediction):
    return -np.mean(labels * np.log(prediction) + (1-labels) * np.log(1-prediction))


def train_data(principle_component_number):
    training_images, validation_images, test_images, \
    training_labels, validation_labels, test_labels = arrange_data_set_for_emotions(EMOTION1, EMOTION0)
    
    pca = PCA(principle_component_number)
    pca.fit(training_images)

    training_pca_images = run_pca_on_samples(pca, training_images)
    validation_pca_images = run_pca_on_samples(pca, validation_images)
    test_pca_images = run_pca_on_samples(pca, test_images)

    training_pca_images = add_bias_coordinate(training_pca_images)
    validation_pca_images = add_bias_coordinate(validation_pca_images)
    test_pca_images = add_bias_coordinate(test_pca_images)
    weights, training_errors, validation_errors = batch_gradient_decent(training_pca_images, training_labels,
                                                                        validation_pca_images, validation_labels)

    return training_errors, validation_errors




def train_n_times_for_k_principle_components(n, k):
    all_training_errors = np.zeros((n, EPOCHS + 1))
    all_validation_errors = np.zeros((n, EPOCHS + 1))

    for i in range(n):
        training_errors, validation_errors = train_data(k)
        all_training_errors[i] = training_errors
        all_validation_errors[i] = validation_errors

    avg_training_errors = np.average(all_training_errors, axis=0)
    avg_validation_errors = np.average(all_validation_errors, axis=0)
    std_training_errors = np.std(all_training_errors, axis=0)
    std_validation_errors = np.std(all_validation_errors, axis=0)

    plt.plot(avg_training_errors, label="training error")
    plt.plot(avg_validation_errors, label="validation error")
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.title("Average errors as a function of number of epochs")
    plt.legend()
    plt.show()

    print()
    print("STDs:")
    for i in EPOCHS_TO_INCLUDE_STD:
        print("Epoch #" + str(i) + " std for training errors is " + str(std_training_errors[i]))
        print("Epoch #" + str(i) + " std for validation errors is " + str(std_validation_errors[i]))

    x = 1


def get_weights_loss(pca_images, labels, weights):
    prediction = logistic_regression(pca_images, weights)
    return regression_loss(labels, prediction)


train_n_times_for_k_principle_components(NUMBER_OF_RUNS, PCA_NUMBER_OF_COMPONENT)