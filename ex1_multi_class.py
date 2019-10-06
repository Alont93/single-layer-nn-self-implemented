import numpy as np
import pandas as pd

from dataloader import load_data, display_face
from pca import *
from ex1_two_class import *

EMOTIONS = ['h','ht','m','s','f','a','d','n']
PCA_NUMBER_OF_COMPONENT = 6
LEARNING_RATE = 1
GRADIENT_DECENT_ITERATION = 10
TRAINING_PERCENTAGE = 0.6
VALIDATION_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.2