# Imports
# --------------------------------------

# Misc imports
from __future__ import absolute_import, division, print_function, unicode_literals

# Project imports
from IrisModel import IrisModel
from IrisTrainingData import IrisTrainingData
from Iris import Iris
from utility import evaluate_model

# OS import
import os

# Math and numpy imports
import math
import numpy as np

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.compat.v2.keras.optimizers import SGD
from tensorflow.compat.v1.math import softmax
from tensorflow.compat.v1.math import argmax

# Train the model
# --------------------------------------

# Training file
training_data = IrisTrainingData('iris_training.csv')

# Test Model
model = IrisModel(hidden_size=10, feature_size=training_data.feature_size, class_count=training_data.class_count)

model.compile(optimizer=AdamOptimizer(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_data.data, training_data.labels, epochs=500, batch_size=32)

# Evaluate model
# --------------------------------------

evaluate_model(model, training_data)

# Evaluate a few single iris samples
# --------------------------------------

# This is from test data. It should be an iris virginica (2)
print('\n\nTest iris 1:')
print('(This is from test data. It should be an iris virginica)')
test_iris = Iris(sepal_length=4.9, sepal_width=2.5, petal_length=4.5, petal_width=1.7)
test_iris.classify(model=model)
print(test_iris)

# Made up sample to classiy
print('\n\nTest iris 2:')
test_iris_2 = Iris(sepal_length=5.4, sepal_width=3.1, petal_length=2.5, petal_width=2.3)
test_iris_2.classify(model=model)
print(test_iris_2)
