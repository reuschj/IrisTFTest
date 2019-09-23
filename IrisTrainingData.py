# Imports
# --------------------------------------

# Project imports
from utility import read_training_file

# OS import
import os

# Math and numpy imports
import math
import numpy as np

# --------------------------------------

# Extracts and holds information from a iris training file and stores in format ready for use with an iris model
class IrisTrainingData(object):

    """Extracts and holds information from a iris training file and stores in format ready for use with an iris model"""

    def __init__(self, training_file, is_relative_path=True):
        current_directory = os.path.dirname(__file__)
        full_path_of_training_file = os.path.join(current_directory, training_file) if is_relative_path else training_file
        t_file = read_training_file(full_path_of_training_file)
        ###
        size = t_file[0]
        class_names = t_file[1]
        data = t_file[2]
        labels = t_file[3]
        ###
        self.data = np.float32(data)
        self.labels = np.asarray(labels, dtype=int)
        self.feature_size = size[1]
        self.sample_size = size[0]
        self.class_names = class_names
        self.class_count = len(class_names)