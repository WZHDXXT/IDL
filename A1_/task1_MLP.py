# -*- coding: utf-8 -*-
"""task1_MLP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_4L69rew1Tpt7-L3m45V6VSMXOtPvEsK
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras import layers
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""loading dataset"""

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

train_images, val_images, train_labels, val_labels = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42)

"""Firstly we set all parameters to fixed values, and then proceed with parameter tuning. We chose 13 hyperparametwes to train our model."""

#Training Hyperparameters
num_classes = 10
regularization = "l2"
dropout_rate = 0.3
dropout = "TRUE"
num_classes = 10
learning_rate = 0.0001
Regularization_rate = 0.0001
dropout_rate = 0.3
lay_num = 3
dense_units = 128
epochs = 20
batch_size = 64
activation = 'relu'
optimizer = 'adam'
kernel_initialize = "glorot_uniform"

"""# Different learning_rate for MLP

 We set different learning_rate like 0.001,0.0001, 0.00001 etc that is common in deep learning and we fix other parameters. As a result, we will find best learning_rate to fit this model.

"""

learning_rate_histories = {}
learning_rates = [0.001, 0.0001, 0.00001, 0.000001, 0.01]

for learning_rate in learning_rates:
    if optimizer.lower() == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer_instance = keras.optimizers.SGD(learning_rate, momentum=0.9)
    elif optimizer.lower() == 'adagrad':
        optimizer_instance = keras.optimizers.Adagrad(learning_rate)
    elif optimizer.lower() == 'rmsprop':
        optimizer_instance = keras.optimizers.RMSprop(learning_rate)
    elif optimizer.lower() == 'adamax':
        optimizer_instance = keras.optimizers.Adamax(learning_rate)
    elif optimizer.lower() == 'nadam':
        optimizer_instance = keras.optimizers.Nadam(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    if kernel_initialize == "random_normal":
        kernel_initializer = keras.initializers.RandomNormal()
    elif kernel_initialize == "glorot_normal":
        kernel_initializer = keras.initializers.GlorotNormal()
    elif kernel_initialize == "he_normal":
        kernel_initializer = keras.initializers.HeNormal()
    elif kernel_initialize == "glorot_uniform":
        kernel_initializer = keras.initializers.GlorotUniform()
    elif kernel_initialize == "he_uniform":
        kernel_initializer = keras.initializers.HeUniform()
    elif kernel_initialize == "random_uniform":
        kernel_initializer = keras.initializers.RandomUniform()
    else:
        raise ValueError(f"Unsupported initializer: {kernel_initialize}")
    if regularization == "l1":
        kernel_regularizer = keras.regularizers.l1(Regularization_rate)
    elif regularization == "l2":
        kernel_regularizer = keras.regularizers.l2(Regularization_rate)
    elif regularization == "l1_l2":
        kernel_regularizer = keras.regularizers.l1_l2(l1=Regularization_rate, l2=Regularization_rate)
    else:
        kernel_regularizer = None
    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28]))
    for _ in range(lay_num):
        if dropout == "TRUE":
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(dense_units, activation=activation, kernel_initializer=kernel_initialize, kernel_regularizer=kernel_regularizer))

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer_instance,
                  metrics=["accuracy"])

    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_images, val_labels), verbose=1)
    learning_rate_histories[learning_rate] = {
        'accuracy': history.history['accuracy'],
        'loss': history.history['loss'],
        'val_accuracy': history.history['val_accuracy'],
        'val_loss': history.history['val_loss']
    }

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def plot_learning_rate_histories(histories):

    plt.subplot(1, 2, 1)
    for lr, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_accuracy']) + 1), metrics['val_accuracy'], label=f'Learning Rate: {lr}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.grid()
    plt.subplot(1, 2, 2)
    for lr, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_loss']) + 1), metrics['val_loss'], label=f'Learning Rate: {lr}')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'.rstrip('0').rstrip('.')))
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='small')
    plt.grid()

    plt.tight_layout()
    plt.show()

plot_learning_rate_histories(learning_rate_histories)

"""# Different regularization types For MLP

In this part, we set 3 values for regularization_types like l1,l2,l1_l2 to identify which one perform best.
"""

# Define different regularization types
regularization_types = ["l1", "l2", "l1_l2"]
regularization_histories = {}
for regularization in regularization_types:
    if optimizer.lower() == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer_instance = keras.optimizers.SGD(learning_rate, momentum=0.9)
    elif optimizer.lower() == 'adagrad':
        optimizer_instance = keras.optimizers.Adagrad(learning_rate)
    elif optimizer.lower() == 'rmsprop':
        optimizer_instance = keras.optimizers.RMSprop(learning_rate)
    elif optimizer.lower() == 'adamax':
        optimizer_instance = keras.optimizers.Adamax(learning_rate)
    elif optimizer.lower() == 'nadam':
        optimizer_instance = keras.optimizers.Nadam(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    if kernel_initialize == "random_normal":
        kernel_initializer = keras.initializers.RandomNormal()
    elif kernel_initialize == "glorot_normal":
        kernel_initializer = keras.initializers.GlorotNormal()
    elif kernel_initialize == "he_normal":
        kernel_initializer = keras.initializers.HeNormal()
    elif kernel_initialize == "glorot_uniform":
        kernel_initializer = keras.initializers.GlorotUniform()
    elif kernel_initialize == "he_uniform":
        kernel_initializer = keras.initializers.HeUniform()
    elif kernel_initialize == "random_uniform":
        kernel_initializer = keras.initializers.RandomUniform()
    else:
        raise ValueError(f"Unsupported initializer: {kernel_initialize}")
    if regularization == "l1":
        kernel_regularizer = keras.regularizers.l1(Regularization_rate)
    elif regularization == "l2":
        kernel_regularizer = keras.regularizers.l2(Regularization_rate)
    elif regularization == "l1_l2":
        kernel_regularizer = keras.regularizers.l1_l2(l1=Regularization_rate, l2=Regularization_rate)
    else:
        kernel_regularizer = None
    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28]))

    for _ in range(lay_num):
        if dropout == "TRUE":
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(dense_units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))

    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer_instance,
                  metrics=["accuracy"])
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_images, val_labels), verbose=1)

    regularization_histories[regularization if regularization else "None"] = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    data_dict = {
        'regularization': regularization if regularization else "None",
        'epochs': epochs,
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }
    df = pd.DataFrame([data_dict])
    csv_file = 'training_results.csv'
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

def plot_regularization_histories(histories):
    plt.subplot(1, 2, 1)
    for reg_type, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_accuracy']) + 1), metrics['val_accuracy'], label=f'Regularization: {reg_type}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.grid()
    plt.subplot(1, 2, 2)
    for reg_type, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_loss']) + 1), metrics['val_loss'], label=f'Regularization: {reg_type}')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='small')
    plt.grid()

    plt.tight_layout()
    plt.show()
plot_regularization_histories(regularization_histories)

"""# Different regularization rates to test with L2 regularization

We set 4 values which are common in regularization rates. we find the best one is 0.0001

"""

# Define different regularization rates to test with L2 regularization
regularization_rates = [0.01, 0.001, 0.0001, 0.00001]
regularization_histories = {}
for reg_rate in regularization_rates:
    if optimizer.lower() == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer_instance = keras.optimizers.SGD(learning_rate, momentum=0.9)
    elif optimizer.lower() == 'adagrad':
        optimizer_instance = keras.optimizers.Adagrad(learning_rate)
    elif optimizer.lower() == 'rmsprop':
        optimizer_instance = keras.optimizers.RMSprop(learning_rate)
    elif optimizer.lower() == 'adamax':
        optimizer_instance = keras.optimizers.Adamax(learning_rate)
    elif optimizer.lower() == 'nadam':
        optimizer_instance = keras.optimizers.Nadam(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    if kernel_initialize == "random_normal":
        kernel_initializer = keras.initializers.RandomNormal()
    elif kernel_initialize == "glorot_normal":
        kernel_initializer = keras.initializers.GlorotNormal()
    elif kernel_initialize == "he_normal":
        kernel_initializer = keras.initializers.HeNormal()
    elif kernel_initialize == "glorot_uniform":
        kernel_initializer = keras.initializers.GlorotUniform()
    elif kernel_initialize == "he_uniform":
        kernel_initializer = keras.initializers.HeUniform()
    elif kernel_initialize == "random_uniform":
        kernel_initializer = keras.initializers.RandomUniform()
    else:
        raise ValueError(f"Unsupported initializer: {kernel_initialize}")
    kernel_regularizer = keras.regularizers.l2(reg_rate)
    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28]))

    for _ in range(lay_num):
        if dropout == "TRUE":
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(dense_units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))

    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer_instance,
                  metrics=["accuracy"])
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_images, val_labels), verbose=1)
    regularization_histories[reg_rate] = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }

    data_dict = {
        'regularization_type': 'l2',
        'regularization_rate': reg_rate,
        'epochs': epochs,
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }

    df = pd.DataFrame([data_dict])
    csv_file = 'regularization_rates.csv'
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

import matplotlib.pyplot as plt
def plot_regularization_histories(histories):
    plt.subplot(1, 2, 1)
    for reg_rate, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_accuracy']) + 1), metrics['val_accuracy'], label=f'Regularization Rate: {reg_rate}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.grid()

    plt.subplot(1, 2, 2)
    for reg_rate, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_loss']) + 1), metrics['val_loss'], label=f'Regularization Rate: {reg_rate}')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='small')
    plt.grid()

    plt.tight_layout()
    plt.show()

plot_regularization_histories(regularization_histories)

"""# Differnt dropout_rate

In this part, we aslo chose 5 dropout_rate to train this model while keeping all other parameters constant.
"""

# differnt dropout
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
dropout_histories = {}
for dropout_rate in dropout_rates:
    if optimizer.lower() == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer_instance = keras.optimizers.SGD(learning_rate, momentum=0.9)
    elif optimizer.lower() == 'adagrad':
        optimizer_instance = keras.optimizers.Adagrad(learning_rate)
    elif optimizer.lower() == 'rmsprop':
        optimizer_instance = keras.optimizers.RMSprop(learning_rate)
    elif optimizer.lower() == 'adamax':
        optimizer_instance = keras.optimizers.Adamax(learning_rate)
    elif optimizer.lower() == 'nadam':
        optimizer_instance = keras.optimizers.Nadam(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    if kernel_initialize == "random_normal":
        kernel_initializer = keras.initializers.RandomNormal()
    elif kernel_initialize == "glorot_normal":
        kernel_initializer = keras.initializers.GlorotNormal()
    elif kernel_initialize == "he_normal":
        kernel_initializer = keras.initializers.HeNormal()
    elif kernel_initialize == "glorot_uniform":
        kernel_initializer = keras.initializers.GlorotUniform()
    elif kernel_initialize == "he_uniform":
        kernel_initializer = keras.initializers.HeUniform()
    elif kernel_initialize == "random_uniform":
        kernel_initializer = keras.initializers.RandomUniform()
    else:
        raise ValueError(f"Unsupported initializer: {kernel_initialize}")

    if regularization == "l1":
        kernel_regularizer = keras.regularizers.l1(Regularization_rate)
    elif regularization == "l2":
        kernel_regularizer = keras.regularizers.l2(Regularization_rate)
    elif regularization == "l1_l2":
        kernel_regularizer = keras.regularizers.l1_l2(l1=Regularization_rate, l2=Regularization_rate)
    else:
        kernel_regularizer = None
    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28]))
    for _ in range(lay_num):
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(dense_units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))

    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer_instance,
                  metrics=["accuracy"])

    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_images, val_labels), verbose=1)
    dropout_histories[dropout_rate] = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    data_dict = {
        'dropout_rate': dropout_rate,
        'epochs': epochs,
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }

    df = pd.DataFrame([data_dict])
    csv_file = 'dropout_results.csv'
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

import matplotlib.pyplot as plt
def plot_dropout_histories(histories):
    plt.subplot(1, 2, 1)
    for rate, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_accuracy']) + 1), metrics['val_accuracy'], label=f'Dropout Rate: {rate}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.grid()
    plt.subplot(1, 2, 2)
    for rate, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_loss']) + 1), metrics['val_loss'], label=f'Dropout Rate: {rate}')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.show()
plot_dropout_histories(dropout_histories)

"""# Different kernel initializers"""

# Define different kernel initializers
kernel_initializers = ["random_normal", "glorot_normal", "he_normal", "glorot_uniform", "he_uniform", "random_uniform"]
initializer_histories = {}
for initializer in kernel_initializers:
    if optimizer.lower() == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer_instance = keras.optimizers.SGD(learning_rate, momentum=0.9)
    elif optimizer.lower() == 'adagrad':
        optimizer_instance = keras.optimizers.Adagrad(learning_rate)
    elif optimizer.lower() == 'rmsprop':
        optimizer_instance = keras.optimizers.RMSprop(learning_rate)
    elif optimizer.lower() == 'adamax':
        optimizer_instance = keras.optimizers.Adamax(learning_rate)
    elif optimizer.lower() == 'nadam':
        optimizer_instance = keras.optimizers.Nadam(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    if initializer == "random_normal":
        kernel_initializer = keras.initializers.RandomNormal()
    elif initializer == "glorot_normal":
        kernel_initializer = keras.initializers.GlorotNormal()
    elif initializer == "he_normal":
        kernel_initializer = keras.initializers.HeNormal()
    elif initializer == "glorot_uniform":
        kernel_initializer = keras.initializers.GlorotUniform()
    elif initializer == "he_uniform":
        kernel_initializer = keras.initializers.HeUniform()
    elif initializer == "random_uniform":
        kernel_initializer = keras.initializers.RandomUniform()
    else:
        raise ValueError(f"Unsupported initializer: {initializer}")

    kernel_regularizer = keras.regularizers.l2(Regularization_rate)

    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28]))

    for _ in range(lay_num):
        if dropout == "TRUE":
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(dense_units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))

    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer_instance,
                  metrics=["accuracy"])
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_images, val_labels), verbose=1)

    initializer_histories[initializer] = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }

    data_dict = {
        'initializer': initializer,
        'epochs': epochs,
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }
    df = pd.DataFrame([data_dict])
    csv_file = 'initializer_results.csv'
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

import matplotlib.pyplot as plt

def plot_initializer_histories(histories):
    plt.subplot(1, 2, 1)
    for initializer, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_accuracy']) + 1), metrics['val_accuracy'], label=f'Initializer: {initializer}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.grid()
    plt.subplot(1, 2, 2)
    for initializer, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_loss']) + 1), metrics['val_loss'], label=f'Initializer: {initializer}')
    plt.yscale('log')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.show()
plot_initializer_histories(initializer_histories)

"""# Different optimizers for MLP

We chose 6 values like 'adam','adagrad' to train model.

"""

# Define different optimizers to test
optimizers = ["adam", "sgd", "adagrad", "rmsprop", "adamax", "nadam"]
optimizer_histories = {}
for optimizer_name in optimizers:
    if optimizer_name.lower() == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer_instance = keras.optimizers.SGD(learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'adagrad':
        optimizer_instance = keras.optimizers.Adagrad(learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer_instance = keras.optimizers.RMSprop(learning_rate)
    elif optimizer_name.lower() == 'adamax':
        optimizer_instance = keras.optimizers.Adamax(learning_rate)
    elif optimizer_name.lower() == 'nadam':
        optimizer_instance = keras.optimizers.Nadam(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    kernel_initializer = keras.initializers.GlorotNormal()
    kernel_regularizer = keras.regularizers.l2(Regularization_rate)
    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28]))

    for _ in range(lay_num):
        if dropout == "TRUE":
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(dense_units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))

    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer_instance,
                  metrics=["accuracy"])

    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_images, val_labels), verbose=1)
    optimizer_histories[optimizer_name] = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    data_dict = {
        'optimizer': optimizer_name,
        'epochs': epochs,
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }
    df = pd.DataFrame([data_dict])
    csv_file = 'optimizer_results.csv'
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

import matplotlib.pyplot as plt

def plot_optimizer_histories(histories):
    plt.subplot(1, 2, 1)
    for optimizer, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_accuracy']) + 1), metrics['val_accuracy'], label=f'Optimizer: {optimizer}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.grid()
    plt.subplot(1, 2, 2)
    for optimizer, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_loss']) + 1), metrics['val_loss'], label=f'Optimizer: {optimizer}')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.show()
plot_optimizer_histories(optimizer_histories)

"""# Different batch sizes

we use a list of batch sizes to trian this model. It trains a new model for each zise,logs the training history and save the results into a csv file for comparison.
"""

# Define different batch sizes to test
batch_sizes = [32, 64, 128, 256]
batch_size_histories = {}
for batch_size in batch_sizes:
    if optimizer.lower() == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer_instance = keras.optimizers.SGD(learning_rate, momentum=0.9)
    elif optimizer.lower() == 'adagrad':
        optimizer_instance = keras.optimizers.Adagrad(learning_rate)
    elif optimizer.lower() == 'rmsprop':
        optimizer_instance = keras.optimizers.RMSprop(learning_rate)
    elif optimizer.lower() == 'adamax':
        optimizer_instance = keras.optimizers.Adamax(learning_rate)
    elif optimizer.lower() == 'nadam':
        optimizer_instance = keras.optimizers.Nadam(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    kernel_initializer = keras.initializers.GlorotNormal()
    kernel_regularizer = keras.regularizers.l2(Regularization_rate)
    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28]))

    for _ in range(lay_num):
        if dropout == "TRUE":
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(dense_units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))

    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer_instance,
                  metrics=["accuracy"])

    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_images, val_labels), verbose=1)
    batch_size_histories[batch_size] = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    data_dict = {
        'batch_size': batch_size,
        'epochs': epochs,
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }
    df = pd.DataFrame([data_dict])
    csv_file = 'batch_size_results.csv'
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

import matplotlib.pyplot as plt

def plot_batch_size_histories(histories):
    plt.subplot(1, 2, 1)
    for batch_size, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_accuracy']) + 1), metrics['val_accuracy'], label=f'Batch Size: {batch_size}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.grid()

    plt.subplot(1, 2, 2)
    for batch_size, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_loss']) + 1), metrics['val_loss'], label=f'Batch Size: {batch_size}')
    plt.yscale('log')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.show()
plot_batch_size_histories(batch_size_histories)

"""# Different lay_nums

We use various numbers of lay-numbers. For each model confuguration, it recordes the model's training history amd store the results in a dictionary.
"""

lay_nums = [1, 2, 3, 4,5,6,7,8,9,10]
layer_histories = {}
for num_layers in lay_nums:
    if optimizer.lower() == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer_instance = keras.optimizers.SGD(learning_rate, momentum=0.9)
    elif optimizer.lower() == 'adagrad':
        optimizer_instance = keras.optimizers.Adagrad(learning_rate)
    elif optimizer.lower() == 'rmsprop':
        optimizer_instance = keras.optimizers.RMSprop(learning_rate)
    elif optimizer.lower() == 'adamax':
        optimizer_instance = keras.optimizers.Adamax(learning_rate)
    elif optimizer.lower() == 'nadam':
        optimizer_instance = keras.optimizers.Nadam(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    kernel_initializer = keras.initializers.GlorotNormal()
    kernel_regularizer = keras.regularizers.l2(Regularization_rate)
    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28]))
    for _ in range(num_layers):
        if dropout == "TRUE":
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(dense_units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))

    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer_instance,
                  metrics=["accuracy"])
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_images, val_labels), verbose=1)

    layer_histories[num_layers] = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }

    data_dict = {
        'num_layers': num_layers,
        'epochs': epochs,
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }

    df = pd.DataFrame([data_dict])
    csv_file = 'layer_results.csv'
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

import matplotlib.pyplot as plt

def plot_layer_histories(histories):
    plt.subplot(1, 2, 1)
    for num_layers, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_accuracy']) + 1), metrics['val_accuracy'], label=f'Layers: {num_layers}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.grid()
    plt.subplot(1, 2, 2)
    for num_layers, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_loss']) + 1), metrics['val_loss'], label=f'Layers: {num_layers}')
    plt.yscale('log')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.show()
plot_layer_histories(layer_histories)

"""# different dense_units_list

we set different dense layer configurations to identify the model's perforance. It trains a separate netural network model for each number of units in the list and store the training histories and results in both a dictionary and a csv file.
"""

dense_units_list = [32, 64, 128, 256]
dense_units_histories = {}
for dense_units in dense_units_list:
    if optimizer.lower() == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate)
    elif optimizer.lower() == 'sgd':
        optimizer_instance = keras.optimizers.SGD(learning_rate, momentum=0.9)
    elif optimizer.lower() == 'adagrad':
        optimizer_instance = keras.optimizers.Adagrad(learning_rate)
    elif optimizer.lower() == 'rmsprop':
        optimizer_instance = keras.optimizers.RMSprop(learning_rate)
    elif optimizer.lower() == 'adamax':
        optimizer_instance = keras.optimizers.Adamax(learning_rate)
    elif optimizer.lower() == 'nadam':
        optimizer_instance = keras.optimizers.Nadam(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    kernel_initializer = keras.initializers.GlorotNormal()
    kernel_regularizer = keras.regularizers.l2(Regularization_rate)
    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=[28, 28]))
    for _ in range(lay_num):
        if dropout == "TRUE":
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(dense_units, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))

    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer_instance,
                  metrics=["accuracy"])
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_images, val_labels), verbose=1)
    dense_units_histories[dense_units] = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }

    data_dict = {
        'dense_units': dense_units,
        'epochs': epochs,
        'accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }
    df = pd.DataFrame([data_dict])
    csv_file = 'dense_units_results.csv'
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

import matplotlib.pyplot as plt
def plot_dense_units_histories(histories):
    plt.subplot(1, 2, 1)
    for units, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_accuracy']) + 1), metrics['val_accuracy'], label=f'Dense Units: {units}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.grid()
    plt.subplot(1, 2, 2)
    for units, metrics in histories.items():
        plt.plot(range(1, len(metrics['val_loss']) + 1), metrics['val_loss'], label=f'Dense Units: {units}')
    plt.yscale('log')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.show()
plot_dense_units_histories(dense_units_histories)