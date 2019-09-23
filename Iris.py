# Imports
# --------------------------------------

# Misc imports
from __future__ import absolute_import, division, print_function, unicode_literals

# Project imports
from utility import *

# OS import
import os

# Math and numpy imports
import math
import numpy as np


# TensorFlow imports
import tensorflow as tf
from tensorflow.compat.v1.math import softmax
from tensorflow.compat.v1.math import argmax

# --------------------------------------

# Holds information about a single iris flower and allows classification as a particular species with a trained iris model
class Iris(object):

    """Holds information about a single iris flower and allows classification as a particular species with a trained iris model"""

    def __init__(self, sepal_length, sepal_width, petal_length, petal_width):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.labels = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
        self.predition = None
        self.classification = None
        self.confidence = 0

    def __str__(self):
        classification = self.classification if self.classification != None else 'Not classified'
        return '%s (Sepal length: %f, Sepal width: %f, Petal length: %f, Petal width: %f)' % (classification, self.sepal_length, self.sepal_width, self.petal_length, self.petal_width)

    # Classifies as a particular species of iris
    def classify(self, model):
        values = [[self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]]
        data = np.float32(values)
        result = model.predict(data, batch_size=32)
        probabilities = softmax(result)
        all_probabilities = probabilities[0].numpy()
        print('')
        print('-' * 30)
        for i in range(len(self.labels)):
            print ('%s: %s confidence' % (self.labels[i], to_percent_str(all_probabilities[i])))
        print('-' * 30)
        prediciton = argmax(result, axis=1)
        self.prediction = prediciton[0].numpy()
        self.classification = translate_label(self.prediction, self.labels)
        self.confidence = probabilities[0][self.prediction].numpy()
        print ('Classification: %s (%s confidence)' % (self.classification, to_percent_str(self.confidence)))
        print('')
        return (self.prediction, self.classification, self.confidence)
