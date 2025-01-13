#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ConfigSpace')
import ConfigSpace
import matplotlib.pyplot as plt
import numpy as np
import keras
import time
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF, WhiteKernel,ConstantKernel as C


# # 1.Random Search on MLP

# After comparing different hyperparameters, we use random search to find 3 best-performing hyperparameter sets by constraining hyperparameters within a certain range.

# Build up configuration space.

# In[ ]:


def get_hyperparameter_search_space():

    cs = ConfigSpace.ConfigurationSpace()
    learning_rate = ConfigSpace.UniformFloatHyperparameter("learning_rate", 1e-5, 0.0001, log=True, default_value=0.0001)
    dropout = ConfigSpace.CategoricalHyperparameter(name='dropout', choices=['t', 'f'], default_value='f')
    keep_prop_rate = ConfigSpace.UniformFloatHyperparameter('keep_prop_rate', lower=0.1, upper=0.5, default_value=0.1, log=False)
    regularization = ConfigSpace.CategoricalHyperparameter(name='regularization', choices=['t', 'f'], default_value='f')
    regularization_param = ConfigSpace.UniformFloatHyperparameter('regularization_param',0.0001 , 0.001, log=True, default_value=0.0001)

    batch_size = ConfigSpace.UniformIntegerHyperparameter('batch_size', 16, 128, default_value=128)
    layer_num = ConfigSpace.UniformIntegerHyperparameter('layer_num', 1, 10, default_value=3)
    dense_num = ConfigSpace.UniformIntegerHyperparameter('dense_num', 10, 500, default_value=300)
    #lay_num = ConfigSpace.UniformIntegerHyperparameter('lay_num', 2, 10, default_value=3)
    kernel_initializer = ConfigSpace.CategoricalHyperparameter('kernel_initializer', ['he_uniform', 'he_normal', 'glorot_uniform'], default_value='he_uniform')
    optimizer = ConfigSpace.CategoricalHyperparameter('optimizer', ['rmsprop', 'sgd', 'adam'], default_value='adam')

    cs.add([batch_size, dense_num, layer_num, learning_rate, regularization_param, dropout,
            keep_prop_rate, regularization, kernel_initializer, optimizer])

    re_depends_on_regularization = ConfigSpace.EqualsCondition(regularization_param, regularization, 't')
    rate_depends_on_dropout = ConfigSpace.EqualsCondition(keep_prop_rate, dropout, 't')
    cs.add(re_depends_on_regularization)
    cs.add(rate_depends_on_dropout)


    return cs


# Build up model.

# In[ ]:


class Model():
    def __init__(self, config, X_train, y_train, X_valid, y_valid, X_test, y_test):
        self.config = config
        self.num = 10
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.optimizer = None
        self.model = self.compute()
    def evaluate(self):
        performance = self.model.evaluate(self.X_test, self.y_test)[1]
        return performance
    def compute(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dropout(self.config['keep_prop_rate']))
        model.add(keras.layers.Flatten(input_shape=[28, 28]))
        for _ in range(self.config['layer_num']):
            if self.config['dropout'] == 't':
                model.add(keras.layers.Dropout(self.config['keep_prop_rate']))
            if self.config['regularization'] == 't':
                dense_num = int(np.random.random()*self.config['dense_num'])
                model.add(keras.layers.Dense(dense_num,
                                             activation='relu',
                                             kernel_initializer=self.config['kernel_initializer'],
                                             kernel_regularizer=keras.regularizers.l2(self.config['regularization_param'])))
            else:
                dense_num = int(np.random.random()*self.config['dense_num'])
                model.add(keras.layers.Dense(dense_num,
                                             activation='relu',
                                             kernel_initializer=self.config['kernel_initializer']))
        model.add(keras.layers.Dropout(self.config['keep_prop_rate']))
        model.add(keras.layers.Dense(10, activation="softmax"))
        if self.config['optimizer']=='rmsprop':
            self.optimizer = keras.optimizers.RMSprop(learning_rate=self.config['learning_rate'])
        if self.config['optimizer']=='adam':
            self.optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        else:
            self.optimizer = keras.optimizers.SGD(learning_rate=self.config['learning_rate'])

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, batch_size=self.config['batch_size'],
                  epochs=50, validation_data=(self.X_valid, self.y_valid))
        return model
def default(cs, configs):
      for hp in list(cs.values()):
            if hp.name not in configs.keys():
                configs[hp.name] = hp.default_value


# Randomly sample configuration to train the model.

# In[ ]:


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1)

cs = get_hyperparameter_search_space()

best_configs = []
best_performances = []
best_model = []
iteration_num = 50
for i in range(3):
    best_performance = -np.inf
    best_config = {}
    for j in range(iteration_num):
    
        config = dict(cs.sample_configuration(size=None))
        default(cs, config)
        model = Model(config, X_train, y_train, X_valid, y_valid, X_test, y_test)
        performance = model.evaluate()
        if performance>best_performance:
            best_config = config
            best_performance = performance
    best_model.append(model)
    best_performances.append(best_performance)
    best_configs.append(best_config)
for i in range(len(best_configs)):
    print(best_configs[i])
print(best_performances)


# {'batch_size': 81, 'dense_num': 497, 'dropout': 't', 'kernel_initializer': 'he_uniform', 'layer_num': 7, 'learning_rate': 7.53568583e-05, 'optimizer': 'adam', 'regularization': 't', 'keep_prop_rate': 0.1180346539385, 'regularization_param': 0.0001208036847}
# 
# 
# 
# {'batch_size': 45, 'dense_num': 254, 'dropout': 'f', 'kernel_initializer': 'he_uniform', 'layer_num': 6, 'learning_rate': 8.17295463e-05, 'optimizer': 'adam', 'regularization': 'f', 'keep_prop_rate': 0.1, 'regularization_param': 0.0001}
# 
# 
# 
# {'batch_size': 104, 'dense_num': 213, 'dropout': 'f', 'kernel_initializer': 'glorot_uniform', 'layer_num': 10, 'learning_rate': 5.04320814e-05, 'optimizer': 'adam', 'regularization': 't', 'regularization_param': 0.0002345882678, 'keep_prop_rate': 0.1}
# 
# 
# 
# [0.8888000249862671, 0.8906999826431274, 0.8892999887466431]


# Next, using these three configurations to train new models on the CIFAR-10 dataset

# 1. import CIFAR-10 dataset.

# In[5]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32')/255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# Split into validation set

# In[6]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[7]:


y_train = keras.utils.to_categorical(y_train, 10)
y_val= keras.utils.to_categorical(y_val, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(y_train.shape)


# Set model to the best configurations.

# In[8]:


config_data =[
    {'batch_size': 81, 'dense_num': 497, 'dropout': 't', 'kernel_initializer': 'he_uniform', 'layer_num': 7, 'learning_rate': 7.53568583e-05, 'optimizer': 'adam', 'regularization': 't', 'keep_prop_rate': 0.1180346539385, 'regularization_param': 0.0001208036847},
    {'batch_size': 45, 'dense_num': 254, 'dropout': 'f', 'kernel_initializer': 'he_uniform', 'layer_num': 6, 'learning_rate': 8.17295463e-05, 'optimizer': 'adam', 'regularization': 'f', 'keep_prop_rate': 0.1, 'regularization_param': 0.0001},
    {'batch_size': 104, 'dense_num': 213, 'dropout': 'f', 'kernel_initializer': 'glorot_uniform', 'layer_num': 10, 'learning_rate': 5.04320814e-05, 'optimizer': 'adam', 'regularization': 't', 'regularization_param': 0.0002345882678, 'keep_prop_rate': 0.1}
]


# Build up model.

# In[ ]:


history={}
for i, config in enumerate(config_data):
  model = keras.models.Sequential()
  model.add(keras.layers.Dropout(config['keep_prop_rate']))
  model.add(keras.layers.Flatten(input_shape=[32,32,3]))
  for _ in range(config['layer_num']):
      if config['dropout'] == 't':
          model.add(keras.layers.Dropout(config['keep_prop_rate']))
      if config['regularization'] == 't':
          dense_num = int(np.random.random()*config['dense_num'])
          model.add(keras.layers.Dense(dense_num,activation='relu',kernel_initializer=config['kernel_initializer'],
              kernel_regularizer=keras.regularizers.l2(config['regularization_param'])))
      else:
          dense_num = int(np.random.random()*config['dense_num'])
          model.add(keras.layers.Dense(dense_num,activation='relu', kernel_initializer=config['kernel_initializer']))
  model.add(keras.layers.Dropout(config['keep_prop_rate']))
  model.add(keras.layers.Dense(10, activation="softmax"))
  if config['optimizer']=='rmsprop':
    optimizer = keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
  if config['optimizer']=='adam':
    optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
  else:
    optimizer = keras.optimizers.SGD(learning_rate=config['learning_rate'])

  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  history[i]=model.fit(x_train, y_train, batch_size=config['batch_size'],
            epochs=20, validation_data=(x_val, y_val))


# In[16]:


plt.figure(figsize=(5, 5)) 
color=['red', 'blue', 'green']
for i, hist in history.items():
    epochs = range(1, len(hist.history['loss']) + 1)
    
    plt.plot(epochs, hist.history['loss'], label=f'Model {i+1} Training Loss', color=color[i])
    plt.plot(epochs, hist.history['val_loss'], label=f'Model {i+1} Validation Loss', linestyle='--', color=color[i])

plt.title('Training and Validation Loss of All Models')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(fontsize='small')
plt.grid()
plt.show()


plt.figure(figsize=(5, 5)) 

for i, hist in history.items():
    epochs = range(1, len(hist.history['accuracy']) + 1)
    
    plt.plot(epochs, hist.history['accuracy'], label=f'Model {i+1} Training Accuracy', color=color[i])
    plt.plot(epochs, hist.history['val_accuracy'], label=f'Model {i+1} Validation Accuracy', linestyle='--', color=color[i])

plt.title('Training and Validation Accuracy of All Models')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(fontsize='small')
plt.grid()
plt.show()


# # 2.Random Search on CNN

# After comparing different hyperparameters, we use random search to find 3 best-performing hyperparameter sets.

# Build up configuration space.

# In[10]:


def get_hyperparameter_search_space():
    cs = ConfigSpace.ConfigurationSpace()
    dropout = ConfigSpace.CategoricalHyperparameter(name='dropout', choices=['t', 'f'], default_value='t')
    keep_prop_rate = ConfigSpace.UniformFloatHyperparameter('keep_prop_rate', lower=0.2, upper=0.4, default_value=0.2)
    regularization_param = ConfigSpace.UniformFloatHyperparameter('regularization_param', 0.0001, 0.001, log=True, default_value=0.0001)
    batch_size = ConfigSpace.UniformIntegerHyperparameter('batch_size', 16, 128, default_value=32)
    dense_num = ConfigSpace.UniformIntegerHyperparameter('dense_num', 64, 512, default_value=64)
    lay_num = ConfigSpace.UniformIntegerHyperparameter('lay_num', 1, 5, default_value=3)
    kernel_initializer = ConfigSpace.CategoricalHyperparameter('kernel_initializer', ['he_uniform', 'he_normal', 'glorot_uniform'], default_value='glorot_uniform')
    optimizer = ConfigSpace.CategoricalHyperparameter('optimizer', ['rmsprop', 'sgd', 'adam'], default_value='adam')
    conv_filters = ConfigSpace.UniformIntegerHyperparameter('conv_filters', lower=16, upper=64, default_value=32)
    kernel_size = ConfigSpace.UniformIntegerHyperparameter('kernel_size', lower=3, upper=5, default_value=3)
    conv_layer_num = ConfigSpace.UniformIntegerHyperparameter('conv_layer_num', lower=1, upper=3, default_value=2)
    learning_rate = ConfigSpace.UniformFloatHyperparameter('learning_rate', lower=0.0001, upper=0.01, log=True, default_value=0.001)


    cs.add([batch_size, dense_num, lay_num, regularization_param, dropout, keep_prop_rate,
                            kernel_initializer, optimizer, conv_filters, kernel_size, conv_layer_num, learning_rate])

    rate_depends_on_dropout = ConfigSpace.EqualsCondition(keep_prop_rate, dropout, 't')
    cs.add(rate_depends_on_dropout)

    return cs


# CNN model

# In[18]:


import numpy as np
import time
import keras

class CNNModel:
    def __init__(self, config, X_train, y_train, X_valid, y_valid, X_test, y_test):
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.optimizer = None
        self.model = self.build_model()


    def evaluate(self):
        performance = self.model.evaluate(self.X_test, self.y_test)[1]
        return performance


    def build_model(self):
        start_time = time.time()
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(
            filters=self.config['conv_filters'],
            kernel_size=self.config['kernel_size'],
            activation='relu',
            input_shape=(28, 28, 1)
        ))

        for _ in range(self.config['conv_layer_num']):
            model.add(keras.layers.Conv2D(
                filters=self.config['conv_filters'],
                kernel_size=self.config['kernel_size'],
                activation='relu'
            ))

        model.add(keras.layers.Flatten())
        for _ in range(self.config['lay_num']):
            dense_num = int(np.random.random() * self.config['dense_num'])
            model.add(keras.layers.Dense(dense_num, activation='relu',
                                         kernel_initializer=self.config['kernel_initializer'],
                                         kernel_regularizer=keras.regularizers.l2(self.config['regularization_param'])))
            if self.config['dropout'] == 't':
                model.add(keras.layers.Dropout(self.config['keep_prop_rate']))

        model.add(keras.layers.Dense(10, activation='softmax'))


        if self.config['optimizer'] == 'rmsprop':
            self.optimizer = keras.optimizers.RMSprop(learning_rate=self.config['learning_rate'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        else:
            self.optimizer = keras.optimizers.SGD(learning_rate=self.config['learning_rate'])


        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['accuracy'])


        model.fit(self.X_train, self.y_train,
                  batch_size=self.config['batch_size'],
                  epochs=30,
                  validation_data=(self.X_valid, self.y_valid))

        return model

def default(cs, config):
    for hp in list(cs.values()):
        if hp.name not in config.keys():
            config[hp.name] = hp.default_value


# In[19]:


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1)

cs = get_hyperparameter_search_space()

best_configs = []
best_performances = []
best_model = []
iteration_num = 10
for i in range(3):
    best_performance = -np.inf
    best_config = {}
    for j in range(iteration_num):
        config = dict(cs.sample_configuration(size=None))
        default(cs, config)
        model = CNNModel(config, X_train, y_train, X_valid, y_valid, X_test, y_test)
        performance = model.evaluate()
        if performance>best_performance:
            best_config = config
            best_performance = performance
    best_model.append(model)
    best_performances.append(best_performance)
    best_configs.append(best_config)

for i in range(len(best_configs)):
    print(best_configs[i])
print(best_performances)


# {'batch_size': 86, 'conv_filters': 58, 'conv_layer_num': 3, 'dense_num': 182, 'dropout': 't', 'kernel_initializer': 'he_uniform', 'kernel_size': 4, 'lay_num': 4, 'learning_rate': 0.0011153481085, 'optimizer': 'rmsprop', 'regularization_param': 0.0004285305778, 'keep_prop_rate': 0.2881900532343}
# 
# 
# 
# 
# 
# {'batch_size': 114, 'conv_filters': 41, 'conv_layer_num': 2, 'dense_num': 439, 'dropout': 't', 'kernel_initializer': 'he_normal', 'kernel_size': 4, 'lay_num': 2, 'learning_rate': 0.0021711460824, 'optimizer': 'rmsprop', 'regularization_param': 0.0002546832188, 'keep_prop_rate': 0.2161377468806}
# 
# 
# 
# 
# 
# {'batch_size': 98, 'conv_filters': 32, 'conv_layer_num': 1, 'dense_num': 256, 'dropout': 'f', 'kernel_initializer': 'he_normal', 'kernel_size': 3, 'lay_num': 4, 'learning_rate': 0.0003705821721, 'optimizer': 'rmsprop', 'regularization_param': 0.000481956921, 'keep_prop_rate': 0.2}
# 
# 
# 
# #### are three best-performing hyperparameter sets, with accuracies of
# 
# 
# 
# [0.9153000116348267, 0.9057999849319458, 0.9099000096321106]

# take the 3 best ones and train new models on the CIFAR-10 dataset.

# 1. import data

# In[18]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32')/255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[20]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[21]:


y_train = keras.utils.to_categorical(y_train, 10)
y_val= keras.utils.to_categorical(y_val, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(y_train.shape)


# 2. Use best-performing configuration.

# In[22]:


config_data = [
    {'batch_size': 86, 'conv_filters': 58, 'conv_layer_num': 3, 'dense_num': 182, 'dropout': 't', 'kernel_initializer': 'he_uniform', 'kernel_size': 4, 'lay_num': 4, 'learning_rate': 0.0011153481085, 'optimizer': 'rmsprop', 'regularization_param': 0.0004285305778, 'keep_prop_rate': 0.2881900532343},
    {'batch_size': 114, 'conv_filters': 41, 'conv_layer_num': 2, 'dense_num': 439, 'dropout': 't', 'kernel_initializer': 'he_normal', 'kernel_size': 4, 'lay_num': 2, 'learning_rate': 0.0021711460824, 'optimizer': 'rmsprop', 'regularization_param': 0.0002546832188, 'keep_prop_rate': 0.2161377468806},
    {'batch_size': 98, 'conv_filters': 32, 'conv_layer_num': 1, 'dense_num': 256, 'dropout': 'f', 'kernel_initializer': 'he_normal', 'kernel_size': 3, 'lay_num': 4, 'learning_rate': 0.0003705821721, 'optimizer': 'rmsprop', 'regularization_param': 0.000481956921, 'keep_prop_rate': 0.2}
]


# Build up model

# In[23]:


history={}
for i, config in enumerate(config_data):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(
        filters=config['conv_filters'],
        kernel_size=(config['kernel_size'], config['kernel_size']),
        activation='relu',
        input_shape=(32, 32, 3)
    ))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    for _ in range(config['conv_layer_num']):
        model.add(keras.layers.Conv2D(
            filters=config['conv_filters'],
            kernel_size=(config['kernel_size'], config['kernel_size']),
            padding='same',
            activation='relu'
        ))

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
         
    model.add(keras.layers.Flatten())
    for _ in range(config['lay_num']):
        dense_num = int(np.random.random() * config['dense_num'])
        model.add(keras.layers.Dense(dense_num, activation='relu',
                                     kernel_initializer=config['kernel_initializer'],
                                     kernel_regularizer=keras.regularizers.l2(config['regularization_param'])))

        if config['dropout'] == 't':
            model.add(keras.layers.Dropout(config['keep_prop_rate']))
    model.add(keras.layers.Dense(10, activation='softmax'))
    if config['optimizer'] == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    else:
        optimizer = keras.optimizers.SGD(learning_rate=config['learning_rate'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history[i] = model.fit(
        x_train, y_train,
        batch_size=config['batch_size'],
        epochs=20,
        verbose=1,
        validation_data=(x_val, y_val)

    )


# In[24]:


plt.figure(figsize=(5, 5)) 
color=['red', 'blue', 'green']
for i, hist in history.items():
    epochs = range(1, len(hist.history['loss']) + 1)
    
    plt.plot(epochs, hist.history['loss'], label=f'Model {i+1} Training Loss', color=color[i])
    plt.plot(epochs, hist.history['val_loss'], label=f'Model {i+1} Validation Loss', linestyle='--', color=color[i])

plt.title('Training and Validation Loss of All Models')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(fontsize='small')
plt.grid()
plt.show()


plt.figure(figsize=(5, 5)) 

for i, hist in history.items():
    epochs = range(1, len(hist.history['accuracy']) + 1)
    
    plt.plot(epochs, hist.history['accuracy'], label=f'Model {i+1} Training Accuracy', color=color[i])
    plt.plot(epochs, hist.history['val_accuracy'], label=f'Model {i+1} Validation Accuracy', linestyle='--', color=color[i])

plt.title('Training and Validation Accuracy of All Models')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(fontsize='small')
plt.grid()
plt.show()

