# -*- coding: utf-8 -*-
#Kaggle Titanic

# Utilities

# Imports


import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from google.colab import drive

"""Mount the notebook to my drive"""

drive.mount('/drive')

"""Instantiate file paths"""

TRAIN_PATH = "/drive/My Drive/Projects/Kaggle Titanic/train.csv"
TEST_PATH = "/drive/My Drive/Projects/Kaggle Titanic/test.csv"
RESULTS_PATH = "/drive/My Drive/Projects/Kaggle Titanic/results.csv"

"""Function to read data from a csv file"""

def read_from_file(filename):

  return pd.read_csv(filename)

  # contents = []
  # with open(filename) as csvfile:
  #   reader = csv.reader(csvfile, delimiter = ',')
  #   for row in reader:
  #     contents.append(row)
  
  # return contents

"""# Code
Utility functions for preparing the data
"""

# Drop specified columns from dataframe
def dropFields(frame, fields):
  for field in fields:
    frame.drop(field, inplace = True, axis = 1)

def tokeniseFields(frame, fields):
  for field in fields:
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(frame[field])
    
    # normalise the values
    for value in tokenizer.word_index:
      tokenizer.word_index[value] -= 1

    frame[field] = [tokenizer.word_index[item.lower()] for item in frame[field]]

"""Get the datasets and prepare them for the network"""

# Get the data sets
testData = read_from_file(TEST_PATH)
trainData = read_from_file(TRAIN_PATH)

# Replace NaN age field in age
ageMean = trainData['Age'].mean()
trainData['Age'].fillna(value = ageMean, inplace = True)

# Replace Nan in Embarked
trainData['Embarked'].fillna(value = 's', inplace = True)

# Get the title from the name field
# TODO - THIS!!!!
f = lambda x : 5
trainData["Name"] = trainData["Name"].apply(f)

# Drop unwanted fields
fields = ['PassengerId', 'Cabin', 'Ticket']
dropFields(trainData, fields)
dropFields(testData, fields[1:])

# Get the IDs from testData - used for displaying results
passengerIDs = np.array(testData.pop('PassengerId'))

# Tokenise relevant fields
fields = ['Sex', 'Embarked']
tokeniseFields(trainData, fields)
tokeniseFields(testData, fields)

# Retrieve the labels from the training data
labels = trainData.pop('Survived')
trainData = trainData.astype(float)

# Print the headers of the data that will be going in to the network
print(f"Test data header {list(testData)}")
print(f"Train data header {list(trainData)}\n")

print(f"Training data has shape {trainData.shape}\n")
print(f"Testing data has shape {testData.shape}\n")

"""Functions to create a model"""

# Decaying Learning Rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=10e-5,
    decay_steps=400 * 40,
    decay_rate=0.9)

# stop when training reaches a set accuracy
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.86 and epoch > 250):
      self.model.stop_training = True
      print('Training stopped!!')

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr

def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(500, input_shape = [8], activation = "relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(200, input_shape = [7], activation = "relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, input_shape = [7], activation = "relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
  ])

  # loss = tf.keras.losses.BinaryCrossentropy()
  loss = tf.keras.losses.Huber()

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  lr_metric = get_lr_metric(optimizer)

  model.compile(loss = loss,
                optimizer = optimizer,
                metrics = 'accuracy',)

  model.summary()

  return model

"""Compile and train the model"""

model = create_model()

# train the model
# split = 890
split = 800
history = model.fit(trainData.iloc[:split,:],
                    labels[:split], 
                    epochs = 500,
                    verbose = 2,
                    callbacks = myCallback())

from matplotlib.image import pil_to_array

def plotGraph(history, start = 0, end = None, losses = 'no'):
  xs = [x for x in range(len(history.history['accuracy']))]
  if losses == 'yes':
    plt.plot(xs[start:end], history.history['loss'][start:end])
    # plt.plot(xs[start:end], history.history['val_loss'][start:end])
  else:
    plt.plot(xs[start:end], history.history['accuracy'][start:end])
    # plt.plot(xs[start:end], history.history['val_accuracy'][start:end])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

  
# display the training results
plotGraph(history, losses = 'no')
plotGraph(history, losses = 'yes')

# Test the results file has been written properly
file = open(RESULTS_PATH, 'r')
fr = csv.reader(file)

for i, row in enumerate(fr):
  if i != 0:
    print(type(row[0]))
    print(int(row[1]))
  else:
    print(row)
  if i == 1:
    break
