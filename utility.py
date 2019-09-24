# Imports
# --------------------------------------

# Math and numpy imports
import math
import numpy as np

# TensorFlow imports
import tensorflow as tf
from tensorflow.compat.v1.math import softmax
from tensorflow.compat.v1.math import argmax

# Function Defs
# --------------------------------------

# Makes a percent string from a percentage float
def to_percent_str(input, places=0):
    multiply_by = 100 * (10 ** places)
    devide_by = multiply_by / 100
    rounded = round(input * multiply_by) / devide_by
    rounded = int(rounded) if places == 0 else rounded
    return str(rounded) + '%'

# --------------------------------------

# Translates numbered label into it's string equivalent
def translate_label(label_index, labels):
    try:
        return str(labels[label_index])
    except:
        return None

# --------------------------------------

# Reads a training file and outputs a tuple with
# 0 -> shape - tuple(sample size, feature size)
# 1 -> labels names - list
# 2 -> data - list
# 3 -> labels - list
def read_training_file(training_file):
    file = open(training_file)

    # Init variables
    sample_size = 0
    feature_size = 0
    class_names = []
    data = []
    labels = []

    line_count = 0
    for line in file:
        if (line_count == 0):
            header = line.strip().split(",")
            sample_size = int(header[0])
            feature_size = int(header[1])
            for f in range(2, len(header)):
                class_names.append(str(header[f]))
        else:
            sample_line = line.strip().split(",")
            sample_values = []
            for v in range(feature_size):
                sample_values.append(float(sample_line[v]))
            data.append(sample_values)
            labels.append(int(sample_line[feature_size]))
        line_count += 1
    file.close()

    return ((sample_size, feature_size), class_names, data, labels)

# --------------------------------------

# Tries the model out on the training data, then compares to labels to check accuracy
def evaluate_model(model, training_data):
    result = model.predict(training_data.data, batch_size=32)

    probabilities = softmax(result)
    predictions = argmax(result, axis=1)

    print('-' * 30)
    print(probabilities)
    print('-' * 30)
    print(predictions)
    print('-' * 30)

    ## Run through results, comparing the predicted result with the acutal
    correct_count = 0
    incorrect_count = 0
    for i in range(len(predictions)):
        predicted_label = translate_label(predictions[i].numpy(), training_data.class_names)
        actual_label = translate_label(training_data.labels[i], training_data.class_names)
        is_correct = predicted_label == actual_label
        if is_correct:
            correct_count += 1
            print('%i: %s' % (i, predicted_label))
        else:
            incorrect_count += 1
            print('%i: INCORRECT! -> %s (Actual: %s)' % (i, predicted_label, actual_label))

    # Result summary
    print('-' * 30)
    print('Correct:', correct_count)
    print('Incorrect:', incorrect_count)
    print('Total:', len(predictions))
    percent_correct = float(correct_count) / float(len(predictions))
    print('Score: %s' % (to_percent_str(percent_correct, places=2)))