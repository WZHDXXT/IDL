#!/usr/bin/env python
# coding: utf-8

# ### Task 2: Develop a “Tell-the-time” network.

# Directly applying CNN structure to larger dataset cause the problem of overfitting, therefore we change some parts of CNN structure and train on large dataset again to see their performance.
# 
# To apply to larger dataset, we changed some parts of CNN structure to avoid overfitting and to capture more image information,

# # 1. multi-class classification problem

# import packages

# In[3]:


import tensorflow as tf
import keras 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import Image


# In[5]:


X_original = np.load('/kaggle/input/clocks150/images.npy')
y_original = np.load('/kaggle/input/clocks150/labels.npy')


# ## 1.1 Using 24 multi-class classification

# Data preprocessing

# 
# After loading the data, we classify the labels into 24 classes, then use one-hot encoding for transformation. 
#     
# We split data into 80/10/10% for training/validation and test sets respectively. 
#     
# The train_test_split splits the data out of order with setting the value of shuffle to be true as default. 
# 
#     

# In[4]:


y = np.zeros(y_original.shape[0])
for i, y_ in enumerate(y_original):
    y[i] = ((y_[0]%12)* 60 + y_[1])//30

# one-hot encoding
y = keras.utils.to_categorical(y, num_classes=24)
X = X_original.astype('float32') / 255.0
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Then we start to build CNN. 
# 
# The output layer has 24 neurons, using the softmax activation function which outputs a probability distribution over 24 classes.
# 
# The model is compiled with the Adam optimizer.
# 
# The loss function is categorical crossentropy, which is ideal for multi-class classification problems.
# </font>

# We customize loss function to avoid penalize predictions near boundaries like 0 and 11 heavily.

# In[3]:


def get_custom_loss(T):
    def custom_circular_loss(y_true, y_pred):
        y_true_class = tf.argmax(y_true, axis=1)
        y_pred_class = tf.argmax(y_pred, axis=1)

        abs_diff = tf.abs(y_true_class - y_pred_class)
        
        circular_diff = tf.minimum(abs_diff, T - abs_diff)
        
        cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        penalty = tf.cast(circular_diff, tf.float32) / T
        custom_loss = cce_loss + penalty  
        
        return custom_loss
    return custom_circular_loss


# In[6]:


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(128, (3, 3), padding='SAME', strides=2, activation='relu', input_shape=(150, 150, 1)))
model.add(keras.layers.MaxPooling2D((3, 3)))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, (3, 3), padding='SAME', strides=2, activation='relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(24, activation='softmax'))
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=get_custom_loss(24), metrics=['accuracy'])


# Visualize the structure of model.

# In[7]:


# keras.utils.plot_model 
keras.utils.plot_model(model, to_file='24class.png', show_shapes=True, show_layer_names=True)
Image(filename='24class.png')


# In[8]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("24class_model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, 
                                                  restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=300,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("24class_model.keras")


# Let's plot to see the results.

# In[9]:


# model = keras.models.load_model("24class_model.keras")
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplots_adjust(wspace=0.4)  


plt.show()


# See the result values and the accuracy.

# In[10]:


y_pred = model.predict(X_test)
print(np.argmax(y_pred, axis=1))
print(np.argmax(y_test, axis=1))
print(np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))/len(y_test))


# After evaluating on test set, the accuracy is 79.2%.

# Encapsulate it to use the same structure.

# In[11]:


def encapsulation(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(128, (3, 3), padding='SAME', strides=2, activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, (3, 3), padding='SAME', strides=2, activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.5))
    return model


# ## 1.2 Using 48 multi-class classification
# 

# Grouping all the samples into 48 categories with 15-minute interval, and then applying the CNN on them to see how it works.

# In[12]:


y = np.zeros(y_original.shape[0])
for i, y_ in enumerate(y_original):
    y[i] = ((y_[0]%12)* 60 + y_[1])/15

y = keras.utils.to_categorical(y, num_classes=48)
X = X_original.astype('float32') / 255.0
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=24)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=24)


# In[13]:


input_shape = (150, 150, 1)
model = encapsulation(input_shape)
model.add(keras.layers.Dense(48, activation='softmax'))

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=get_custom_loss(48), metrics=['accuracy'])


# In[14]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("48class_model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, 
                                                  restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=300,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("48class_model.keras")


# Plot to see the result.

# In[15]:


# model = keras.models.load_model("48class_model.keras")
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplots_adjust(wspace=0.4) 

plt.show()


# See the results.

# In[16]:


y_pred = model.predict(X_test)
print(np.argmax(y_pred, axis=1))
print(np.argmax(y_test, axis=1))
print(np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))/len(y_test))


# The accuracy of prediction is 81.7%, which is not bad. Then we can try 720 classes using the same architecture to see how it performs.

# ## 1.3 Using 720 multi-class classification
# 

# Grouping all the samples into 720 categories, and then applying the CNN on them to see how it works.

# First, customize a new loss function with 10 minutes tolerance, which means if the difference between the predict value and the actual value is less than 10 minutes, there will be no penalty to the loss function.

# 1. custom loss function

# In[4]:


def get_custom_circular_loss_t(T, tolerance):
    def custom_circular_loss(y_true, y_pred):
        y_true_class = tf.argmax(y_true, axis=1)
        y_pred_class = tf.argmax(y_pred, axis=1)

        abs_diff = tf.abs(y_true_class - y_pred_class)
        
        circular_diff = tf.minimum(abs_diff, T - abs_diff)
        
        cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        penalty = tf.where(
            circular_diff <= tolerance, 
            0.0,  
            tf.cast(circular_diff, tf.float32) / T  
        )
        custom_loss = cce_loss + penalty  
        
        return custom_loss
    return custom_circular_loss


# In[18]:


y = np.zeros(y_original.shape[0])
for i, y_ in enumerate(y_original):
    y[i] = ((y_[0]%12)* 60 + y_[1])

y = keras.utils.to_categorical(y, num_classes=720)
X = X_original.astype('float32') / 255.0
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=24)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=24)


# In[19]:


input_shape = (150, 150, 1)
model = encapsulation(input_shape)

model.add(keras.layers.Dense(720, activation='softmax'))
optimizer = keras.optimizers.Adam(learning_rate=0.0002)
model.compile(optimizer=optimizer, loss=get_custom_circular_loss_t(720, 10), metrics=['accuracy'])


# In[20]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("720class_model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, 
                                                  restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=300,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("720class_model.keras")


# See the results.

# In[21]:


# model = keras.models.load_model("720class_model.keras")
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplots_adjust(wspace=0.4) 

plt.show()


# In[22]:


y_pred = model.predict(X_test)
print(np.argmax(y_pred, axis=1))
print(np.argmax(y_test, axis=1))
print(np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))/len(y_test))


# The acutal correct ratio is pretty low, now calculate it again considering the tolerance.

# In[ ]:


def tolerance_rate(y_pred, y_test, T, tolerance):
    count = 0
    for i in range(len(y_test)):
        abs_diff = np.abs(np.argmax(y_pred[i]) - np.argmax(y_test[i]))
        min_diff = min(T - abs_diff, abs_diff)
        if min_diff <= tolerance:
            count += 1
    return count/len(y_test)


# In[24]:


# within error tolerance
y_pred = model.predict(X_test)
rate = tolerance_rate(y_pred, y_test, 720, 10)
print(rate)


# The same structure doesn't work. There is no improvement in the accuracy on validation set.
# 
# The vast number of classes means that small differences in the model’s output can lead to significant misclassifications.
# 
# Treating each minute as a distinct class ignores the relationship between adjacent minutes.As the model overfit the training data to try to learn the nuances of all 720 classes. 
# This results in poor generalization to unseen data. Even considering the 10 minutes tolerance.
# 

# # 2. Regression problem

# ## 2.1.First method

# To transform recognition into a regression problem, we transform categorical labels of hours and minutes to continuous value. Decimal form, 3:30 to be 3.5.

# In[65]:


y = np.zeros(y_original.shape[0])
for i, y_ in enumerate(y_original):
    y[i] = (y_[0]%12 + y_[1]/60)

X = X_original.astype('float32') / 255.0
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=24)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=24)


# Custom “common sense” error measure, common sense error of below 10 minutes is achievable

# In[6]:


def get_custom_regression_loss(T, tolerance):
    def custom_mae(y_true, y_pred):
        y_pred = abs(y_pred)
        y_pred = tf.math.floormod(y_pred, T)
        y_true = tf.math.floormod(y_true, T)
        abs_diff = tf.math.abs(y_true - y_pred)
        min_diff = tf.minimum(T - abs_diff, abs_diff)
        loss = tf.where(min_diff <= tolerance, 0.0, min_diff)

        # loss = min_diff ** 2
        custom_loss = tf.reduce_mean(loss)
        return custom_loss
    return custom_mae


# In[7]:


def encapsulation_regression(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(128, (3, 3), padding='SAME', strides=2, activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, (5, 5), padding='SAME', strides=2, activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))

    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.5))
    return model


# In[67]:


model = encapsulation_regression((150, 150, 1))

model.add(keras.layers.Dense(1, activation='linear'))
optimizer = keras.optimizers.Adam(learning_rate=0.0008)
model.compile(optimizer=optimizer, loss=get_custom_regression_loss(12, float(1/6)))


# In[68]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("regression_model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=300,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("regression_model.keras")


# Visualize the model.

# In[69]:


keras.utils.plot_model(model, to_file='regression.png', show_shapes=True, show_layer_names=True)
Image(filename='regression.png')


# In[71]:


# model = keras.models.load_model("regression_model.keras")
plt.subplot(1, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Calculate the ratio of clock that has been recognized correctly within acceptable error range.

# In[7]:


def tolerance_regression_rate(y_pred, y_test, T, tolerance):
    count = 0
    y_pred = abs(y_pred)
    for i in range(len(y_test)):
        abs_diff = np.abs(y_pred[i] - y_test[i])
        min_diff = min(T - abs_diff, abs_diff)
        if min_diff <= tolerance:
            count += 1
    return count/len(y_test)


# In[73]:


y_pred = model.predict(X_test)
rate = tolerance_regression_rate(y_pred, y_test, 12, float(1/6))
print(rate)


# ## 2.2 second method

# The second method is to transform labels by computing the total minutes.

# In[74]:


y = np.zeros(y_original.shape[0])
for i, y_ in enumerate(y_original):
    y[i] = ((y_[0]%12)*60 + y_[1])

X = X_original.astype('float32') / 255.0
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=24)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=24)


# In[77]:


model = encapsulation_regression((150, 150, 1))

model.add(keras.layers.Dense(1, activation='linear'))
optimizer = keras.optimizers.Adam(learning_rate=0.0008)
model.compile(optimizer=optimizer, loss=get_custom_regression_loss(720, 10))


# In[78]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("regression_minutes_model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=300,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("regression_minutes_model.keras")


# In[79]:


keras.utils.plot_model(model, to_file='regression_minutes.png', show_shapes=True, show_layer_names=True)
Image(filename='regression_minutes.png')


# In[80]:


# model = keras.models.load_model("regression_model_minutes.keras")
plt.subplot(1, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[81]:


y_pred = model.predict(X_test)
rate = tolerance_regression_rate(y_pred, y_test, 720, 10)
print(rate)


# # 2. Multi-head problem

# Cutomize common error function, which transform labels to total minutes and count inconsistency considering 10 minutes' tolerance.

# In[20]:


def common_error(y_pred, y_true):
    count = 0
    for i in range(len(y_pred)):
        true_hour, true_minute = y_true[i]
        pred_hour, pred_minute = abs(y_pred[i])
        
        true_total_minutes = (true_hour % 12) * 60 + true_minute
        pred_total_minutes = (pred_hour % 12) * 60 + pred_minute
        
        abs_diff = np.abs(true_total_minutes - pred_total_minutes)
        minute_diff = min(abs_diff, 720 - abs_diff)  
        
        if minute_diff <= 10:
            count += 1
    
    return count / len(y_true)


# ## 2.1 regression

# The hour and minute are numerical output of the two heads using linear activation function.

# In[174]:


X = X_original.astype('float32') / 255.0
y = y_original

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=24)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=24)


# In[175]:


inputs = keras.Input(shape=(150, 150, 1), name="inputs")

conv1 = keras.layers.Conv2D(64, (3, 3), padding='SAME', strides=2, activation='relu', name="conv1")(inputs)
bn1 = keras.layers.BatchNormalization()(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), name="pool1")(bn1)

conv2 = keras.layers.Conv2D(32, (3, 3), padding='SAME', strides=2, activation='relu', name="conv2")(pool1)
bn2 = keras.layers.BatchNormalization()(conv2)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(bn2)

flatten = keras.layers.Flatten(name="flatten")(pool2)

dense_hours = keras.layers.Dense(32, activation='relu', name="dense_hours", kernel_regularizer=keras.regularizers.l2(0.0001))(flatten)
dropout_hours = keras.layers.Dropout(0.5, name="dropout_hours")(dense_hours)
output_hours = keras.layers.Dense(1, activation='linear', name="output_hours")(dropout_hours)

conv3 = keras.layers.Conv2D(32, (3, 3), padding='SAME', activation='relu', name="conv3")(pool2)
bn3 = keras.layers.BatchNormalization()(conv3)
pool3 = keras.layers.MaxPooling2D(pool_size=(3, 3), name="pool3")(bn3)

conv4 = keras.layers.Conv2D(16, (3, 3), padding='SAME', activation='relu', name="conv4")(pool3)
bn4 = keras.layers.BatchNormalization()(conv4)
pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(bn4)

flatten_minutes = keras.layers.Flatten(name="flatten_minutes")(pool4)

dense_minutes = keras.layers.Dense(32, activation='relu', name="dense_minutes", kernel_regularizer=keras.regularizers.l2(0.0001))(flatten_minutes)
dropout_minutes = keras.layers.Dropout(0.5, name="dropout_minutes")(dense_minutes)
bn_dense1 = keras.layers.BatchNormalization()(dropout_minutes)

output_minutes = keras.layers.Dense(1, activation='linear', name="output_minutes")(bn_dense1)

model = keras.Model(inputs=inputs, outputs=[output_hours, output_minutes])
optimizer = keras.optimizers.Adam(learning_rate=0.001) 
model.compile(loss={"output_hours": get_custom_regression_loss(12, float(1/6)), "output_minutes": get_custom_regression_loss(60, 10)}, loss_weights={"output_hours": 0.5, "output_minutes": 0.5}, optimizer=optimizer)


# In[176]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("multihead_regression_model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(
    X_train, [y_train[:, 0], y_train[:, 1]], epochs=300,
    validation_data=(X_valid, [y_valid[:, 0], y_valid[:, 1]]),
    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("multihead_regression_model.keras")


# Visualize the structure.

# In[177]:


keras.utils.plot_model(model, to_file='regression_multihead.png', show_shapes=True, show_layer_names=True)
Image(filename='regression_multihead.png')


# In[178]:


plt.subplot(1, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[179]:


hour_pred, minute_pred = model.predict(X_test)
print(np.hstack((hour_pred, minute_pred)))
print(y_test)


# Calculate the accuracy using common error function, which transform labels to total minutes and count inconsistency considering 10 minutes' tolerance.

# In[180]:


print(common_error(np.hstack((hour_pred, minute_pred)), y_test))


# ## 2.2 classification 

# Applying classification as the outputs, one head for predicting hours and the other head for predicting minutes.

# In[9]:


X = X_original.astype('float32') / 255.0
y = y_original
y0 = y[:, 0].reshape(-1, 1)
y1 = y[:, 1].reshape(-1, 1)
hour_labels = keras.utils.to_categorical(y0, num_classes=12)
minute_labels = keras.utils.to_categorical(y1, num_classes=60)

x_train, x_temp, hour_train, hour_test, minute_train, minute_test = train_test_split(X, hour_labels, minute_labels, test_size=0.2)
x_val, x_test, hour_val, hour_test, minute_val, minute_test = train_test_split(x_temp, hour_test, minute_test, test_size=0.5)


# In[10]:


inputs = keras.Input(shape=(150, 150, 1), name="inputs")

conv1 = keras.layers.Conv2D(16, (3, 3), padding='SAME', strides=2, activation='relu', name="conv1")(inputs)
bn1 = keras.layers.BatchNormalization()(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), name="pool1")(bn1)

conv2 = keras.layers.Conv2D(16, (3, 3), padding='SAME', strides=2, activation='relu', name="conv2")(pool1)
bn2 = keras.layers.BatchNormalization()(conv2)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(bn2)

flatten = keras.layers.Flatten(name="flatten")(pool2)

dense_hours = keras.layers.Dense(128, activation='relu', name="dense_hours", kernel_regularizer=keras.regularizers.l2(0.0001))(flatten)
dropout_hours = keras.layers.Dropout(0.5, name="dropout_hours")(dense_hours)
output_hours = keras.layers.Dense(12, activation='softmax', name="output_hours")(dropout_hours)

conv3 = keras.layers.Conv2D(16, (3, 3), padding='SAME', activation='relu', name="conv3")(pool2)
bn3 = keras.layers.BatchNormalization()(conv3)
pool3 = keras.layers.MaxPooling2D(pool_size=(3, 3), name="pool3")(bn3)

conv4 = keras.layers.Conv2D(32, (3, 3), padding='SAME', activation='relu', name="conv4")(pool3)
bn4 = keras.layers.BatchNormalization()(conv4)
pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(bn4)

flatten_minutes = keras.layers.Flatten(name="flatten_minutes")(pool4)

dense_minutes = keras.layers.Dense(64, activation='relu', name="dense_minutes", kernel_regularizer=keras.regularizers.l2(0.001))(flatten_minutes)
dropout_minutes = keras.layers.Dropout(0.5, name="dropout_minutes")(dense_minutes)
bn_dense1 = keras.layers.BatchNormalization()(dropout_minutes)

output_minutes = keras.layers.Dense(60, activation='softmax', name="output_minutes")(bn_dense1)


# In[11]:


model = keras.Model(inputs=inputs, outputs=[output_hours, output_minutes])
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss={'output_hours':get_custom_loss(12), 'output_minutes':get_custom_circular_loss_t(60, 10)}, loss_weights={"output_hours": 0.8, "output_minutes": 0.5}, optimizer=optimizer, metrics=['accuracy', 'accuracy'])


# In[12]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("multihead_classification_model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
history = model.fit(
    x_train, [hour_train, minute_train], epochs=300,
    validation_data=(x_val, [hour_val, minute_val]),
    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("multihead_classification_model.keras")


# In[13]:


# model = keras.models.load_model("multihead_classification_model.keras")
plt.subplot(1, 2, 1)
plt.plot(history.history['output_hours_accuracy'], label='Hour Accuracy')
plt.plot(history.history['val_output_hours_accuracy'], label='Validation Hour Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['output_minutes_accuracy'], label='Minute Accuracy')
plt.plot(history.history['val_output_minutes_accuracy'], label='Validation Minute Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplots_adjust(wspace=0.4) 

plt.show()


# See the results of hour and minute respectively.

# In[15]:


hour_pred, minute_pred = model.predict(x_test)
print(np.argmax(minute_pred, axis=1))
print(np.argmax(minute_test, axis=1))
print(np.sum((np.argmax(minute_pred, axis=1)== np.argmax(minute_test, axis=1)))/len(minute_test))

print(np.argmax(hour_test, axis=1))
print(np.sum(np.argmax(hour_pred, axis=1) == np.argmax(hour_test, axis=1))/len(hour_test))


# Now calculate considering common sense error.

# In[16]:


y_pred = np.vstack((np.argmax(hour_pred, axis=1), np.argmax(minute_pred, axis=1))).T
y_test = np.vstack((np.argmax(hour_test, axis=1), np.argmax(minute_test, axis=1))).T
print(y_pred)
print(y_test)
print(common_error(y_pred, y_test))


# ## 2.3 combination 

# Using combination of the two methods, we initially transform the problem of the hour prediction as a regression problem and the minute prediction as a classification problem. Then swap the approach, treating the hour prediction as a classification problem and the minute prediction as a regression problem.

# 1. the hour prediction as a regression problem and the minute prediction as a classification

# In[19]:


X = X_original.astype('float32') / 255.0
y = y_original
y1 = y[:, 1].reshape(-1, 1)
hour_labels = y[:, 0].reshape(-1, 1)
minute_labels = keras.utils.to_categorical(y1, num_classes=60)

x_train, x_temp, hour_train, hour_test, minute_train, minute_test = train_test_split(X, hour_labels, minute_labels, test_size=0.2)
x_val, x_test, hour_val, hour_test, minute_val, minute_test = train_test_split(x_temp, hour_test, minute_test, test_size=0.5)


# In[20]:


inputs = keras.Input(shape=(150, 150, 1), name="inputs")

conv1 = keras.layers.Conv2D(32, (3, 3), padding='SAME', strides=2, activation='relu', name="conv1")(inputs)
bn1 = keras.layers.BatchNormalization()(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), name="pool1")(bn1)

conv2 = keras.layers.Conv2D(16, (3, 3), padding='SAME', strides=2, activation='relu', name="conv2")(pool1)
bn2 = keras.layers.BatchNormalization()(conv2)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(bn2)

flatten = keras.layers.Flatten(name="flatten")(pool2)

dense_hours = keras.layers.Dense(32, activation='relu', name="dense_hours", kernel_regularizer=keras.regularizers.l2(0.001))(flatten)
dropout_hours = keras.layers.Dropout(0.5, name="dropout_hours")(dense_hours)
output_hours = keras.layers.Dense(1, activation='linear', name="output_hours")(dropout_hours)

conv3 = keras.layers.Conv2D(16, (3, 3), padding='SAME', activation='relu', name="conv3")(pool2)
bn3 = keras.layers.BatchNormalization()(conv3)
pool3 = keras.layers.MaxPooling2D(pool_size=(3, 3), name="pool3")(bn3)

conv4 = keras.layers.Conv2D(32, (3, 3), padding='SAME', activation='relu', name="conv4")(pool3)
bn4 = keras.layers.BatchNormalization()(conv4)
pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(bn4)

flatten_minutes = keras.layers.Flatten(name="flatten_minutes")(pool4)

dense_minutes = keras.layers.Dense(32, activation='relu', name="dense_minutes", kernel_regularizer=keras.regularizers.l2(0.001))(flatten_minutes)
dropout_minutes = keras.layers.Dropout(0.5, name="dropout_minutes")(dense_minutes)
bn_dense1 = keras.layers.BatchNormalization()(dropout_minutes)

output_minutes = keras.layers.Dense(60, activation='softmax', name="output_minutes")(bn_dense1)


# In[21]:


model = keras.Model(inputs=inputs, outputs=[output_hours, output_minutes])
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss={'output_hours':get_custom_regression_loss(12, float(1/6)), 'output_minutes':get_custom_circular_loss_t(60, 10)}, loss_weights={"output_hours": 0.5, "output_minutes": 0.5}, optimizer=optimizer, metrics={'output_hours': ['mae'], 'output_minutes': ['accuracy']})


# In[22]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("multihead_combination_model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
history = model.fit(
    x_train, [hour_train, minute_train], epochs=300,
    validation_data=(x_val, [hour_val, minute_val]),
    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("multihead_combination_model.keras")


# In[23]:


# model = keras.models.load_model("multihead_combination_model.keras")
plt.subplot(1, 2, 1)
plt.plot(history.history['output_hours_mae'], label='Hour Loss')
plt.plot(history.history['val_output_hours_mae'], label='Validation Hour Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['output_minutes_accuracy'], label='Minute Accuracy')
plt.plot(history.history['val_output_minutes_accuracy'], label='Validation Minute Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplots_adjust(wspace=0.4) 

plt.show()


# Hour accuracy

# In[29]:


hour_pred, minute_pred = model.predict(x_test)

print(hour_pred)
print(hour_test)
print(tolerance_regression_rate(hour_pred, hour_test, 12, float(1/6)))


# minute accuracy

# In[30]:


rate = tolerance_rate(minute_pred, minute_test, 60, 10)
print(rate)


# In[54]:


hour_pred, minute_pred = model.predict(x_test)

minute_test = np.argmax(minute_test, axis=1).reshape(-1, 1)
minute_pred = np.argmax(minute_pred, axis=1).reshape(-1, 1)
print(common_error(np.hstack((hour_pred, minute_pred)), np.hstack((hour_test, minute_test))))


# 2. the hour prediction as a classification problem and the minute prediction as a regression problem.

# In[148]:


X = X_original.astype('float32') / 255.0
y = y_original
y0 = y[:, 0].reshape(-1, 1)
minute_labels = y[:, 1].reshape(-1, 1)
hour_labels = keras.utils.to_categorical(y0, num_classes=12)

x_train, x_temp, hour_train, hour_test, minute_train, minute_test = train_test_split(X, hour_labels, minute_labels, test_size=0.2)
x_val, x_test, hour_val, hour_test, minute_val, minute_test = train_test_split(x_temp, hour_test, minute_test, test_size=0.5)


# In[136]:


inputs = keras.Input(shape=(150, 150, 1), name="inputs")

conv1 = keras.layers.Conv2D(32, (3, 3), padding='SAME', strides=2, activation='relu', name="conv1")(inputs)
bn1 = keras.layers.BatchNormalization()(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), name="pool1")(bn1)

conv2 = keras.layers.Conv2D(16, (3, 3), padding='SAME', strides=2, activation='relu', name="conv2")(pool1)
bn2 = keras.layers.BatchNormalization()(conv2)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(bn2)

flatten = keras.layers.Flatten(name="flatten")(pool2)

dense_hours = keras.layers.Dense(64, activation='relu', name="dense_hours", kernel_regularizer=keras.regularizers.l2(0.0001))(flatten)
dropout_hours = keras.layers.Dropout(0.5, name="dropout_hours")(dense_hours)
output_hours = keras.layers.Dense(12, activation='softmax', name="output_hours")(dropout_hours)

conv3 = keras.layers.Conv2D(16, (3, 3), padding='SAME', activation='relu', name="conv3")(pool2)
bn3 = keras.layers.BatchNormalization()(conv3)
pool3 = keras.layers.MaxPooling2D(pool_size=(3, 3), name="pool3")(bn3)

conv4 = keras.layers.Conv2D(8, (3, 3), padding='SAME', activation='relu', name="conv4")(pool3)
bn4 = keras.layers.BatchNormalization()(conv4)
pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(bn4)

flatten_minutes = keras.layers.Flatten(name="flatten_minutes")(pool4)

dense_minutes = keras.layers.Dense(32, activation='relu', name="dense_minutes", kernel_regularizer=keras.regularizers.l2(0.0001))(flatten_minutes)
dropout_minutes = keras.layers.Dropout(0.5, name="dropout_minutes")(dense_minutes)
bn_dense1 = keras.layers.BatchNormalization()(dropout_minutes)

output_minutes = keras.layers.Dense(1, activation='linear', name="output_minutes")(bn_dense1)


# In[137]:


model = keras.Model(inputs=inputs, outputs=[output_hours, output_minutes])
optimizer = keras.optimizers.Adam(learning_rate=0.0002)
model.compile(loss={'output_hours':get_custom_loss(12), 'output_minutes':get_custom_regression_loss(60, 10)}, loss_weights={"output_hours": 0.5, "output_minutes": 0.5}, optimizer=optimizer, metrics={'output_hours': ['accuracy'], 'output_minutes': ['mae']})


# In[138]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("multihead_combination_model2.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
history = model.fit(
    x_train, [hour_train, minute_train], epochs=300,
    validation_data=(x_val, [hour_val, minute_val]),
    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("multihead_combination_model2.keras")


# In[139]:


plt.subplot(1, 2, 1)
plt.plot(history.history['output_hours_accuracy'], label='Hour Accuracy')
plt.plot(history.history['val_output_hours_accuracy'], label='Validation Hour Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['output_minutes_mae'], label='Minute Loss')
plt.plot(history.history['val_output_minutes_mae'], label='Validation Minute Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplots_adjust(wspace=0.4) 

plt.show()


# In[149]:


hour_pred, minute_pred = model.predict(x_test)
print(hour_test)

hour_pred = np.argmax(hour_pred, axis=1).reshape(-1, 1)
hour_test = np.argmax(hour_test, axis=1).reshape(-1, 1)

print(np.hstack((hour_pred, minute_pred)))
print(np.hstack((hour_test, minute_test)))
print(common_error(np.hstack((hour_pred, minute_pred)), np.hstack((hour_test, minute_test))))


# # 4. Label Transform

# To implement label transform, we use sine and cosine functions to represent the angles on the unit circle.
# First, apply sine and cosine functions with total minutes.

# In[62]:


X_train_full = X_original.astype('float32') / 255.0
y_train_full = y_original.astype(np.float64)
T = 720

for y in y_train_full:
    A = y[0]%12 * 60 + y[1]
    theta = 2 * np.pi * (A / T)
    y[0] = np.sin(theta)
    y[1] = np.cos(theta)
    
    
print(X_train_full.shape)
print(y_train_full.shape)

X_train, X_test_full, y_train, y_test_full = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=24)
X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, test_size=0.5, random_state=24)


# Customize loss function.

# In[63]:


def custom_mse_loss(y_true, y_pred):
    sin_true, cos_true = y_true[:, 0], y_true[:, 1]
    sin_pred, cos_pred = y_pred[:, 0], y_pred[:, 1]
    
    sin_diff = sin_true - sin_pred
    cos_diff = cos_true - cos_pred
    
    loss = tf.reduce_mean(tf.square(sin_diff) + tf.square(cos_diff))
    return loss


# In[64]:


def EP(input_shape):
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.5))

    return model

model = EP((150, 150, 1))
model.add(keras.layers.Dense(2))
    
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=custom_mse_loss, metrics=[custom_mse_loss])    

# history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_valid, y_valid))


# In[65]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("transform_model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=300,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("transform_model.keras")


# In[66]:


# model = keras.models.load_model("transform_model.keras")
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['custom_mse_loss'], label='Training mse')
plt.plot(history.history['val_custom_mse_loss'], label='mse')
plt.title('mse')
plt.xlabel('Epochs')
plt.ylabel('mse')
plt.legend()
plt.show()


# In[67]:


keras.utils.plot_model(model, to_file='transform_model.png', show_shapes=True, show_layer_names=True)
Image(filename='transform_model.png')


# Transform angels back to time and calculate common error.

# In[68]:


def sin_cos_to_time(sin_val, cos_val, T=720):
    theta = np.arctan2(sin_val, cos_val)
    if theta < 0:
        theta += 2 * np.pi
    A = theta * T / (2 * np.pi)
    total_minutes = A
    hours = int((total_minutes // 60) % 12)
    minutes = int(total_minutes % 60)
    return hours, minutes


# In[69]:


y_pred = model.predict(X_test)

pred_times = np.array([sin_cos_to_time(sin, cos) for sin, cos in y_pred])
print(pred_times[30:50])
true_times = np.array([sin_cos_to_time(sin, cos) for sin, cos in y_test])
print(true_times[30:50])

common_error(pred_times, true_times)


# Use tanh as the activation function.

# In[23]:


model = EP((150, 150, 1))
model.add(keras.layers.Dense(2, activation='tanh'))
    
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=custom_mse_loss, metrics=[custom_mse_loss])    


# In[24]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("transform_tanh_model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=300,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("transform_tanh_model.keras")


# In[25]:


plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['custom_mse_loss'], label='Training mse')
plt.plot(history.history['val_custom_mse_loss'], label='mse')
plt.title('mse')
plt.xlabel('Epochs')
plt.ylabel('mse')
plt.legend()
plt.show()


# In[26]:


pred_times = np.array([sin_cos_to_time(sin, cos) for sin, cos in y_pred])
print(pred_times[30:50])
true_times = np.array([sin_cos_to_time(sin, cos) for sin, cos in y_test])
print(true_times[30:50])

common_error(pred_times, true_times)


# Secondly, set hour and minute apart, mapping each to an angle.

# In[55]:


X_train_full = X_original.astype('float32') / 255.0
y_train_full = y_original.astype(np.float64)

y_t = np.zeros((y_train_full.shape[0], 4))

for i, y in enumerate(y_train_full):
    theta_hours = 2 * np.pi * (y[0] % 12) / 12
    y_t[i, 0] = np.sin(theta_hours)
    y_t[i, 1] = np.cos(theta_hours)
    
    # Convert minutes to sine and cosine
    theta_minutes = 2 * np.pi * y[1] / 60
    y_t[i, 2] = np.sin(theta_minutes)
    y_t[i, 3] = np.cos(theta_minutes)
    

X_train, X_test_full, y_train, y_test_full = train_test_split(X_train_full, y_t, test_size=0.2, random_state=24)
X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, test_size=0.5, random_state=24)


# In[56]:


def custom_mse_loss(y_true, y_pred):
    sin_true, cos_true = y_true[:, 0:2], y_true[:, 2:4]
    sin_pred, cos_pred = y_pred[:, 0:2], y_pred[:, 2:4]
    
    sin_diff = sin_true - sin_pred
    cos_diff = cos_true - cos_pred
    
    loss = tf.reduce_mean(tf.square(sin_diff) + tf.square(cos_diff))
    return loss


# In[57]:


model = EP((150, 150, 1))
model.add(keras.layers.Dense(4))
    
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=custom_mse_loss, metrics=[custom_mse_loss])    


# In[58]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("transform_h_model.keras",
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=300,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

model.save("transform_h_model.keras")


# In[43]:


plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['custom_mse_loss'], label='Training mse')
plt.plot(history.history['val_custom_mse_loss'], label='mse')
plt.title('mse')
plt.xlabel('Epochs')
plt.ylabel('mse')
plt.legend()
plt.show()


# In[44]:


keras.utils.plot_model(model, to_file='transform_h_model.png', show_shapes=True, show_layer_names=True)
Image(filename='transform_h_model.png')


# In[49]:


def sin_cos_to_time(sin_hour, cos_hour, sin_minute, cos_minute):
    theta_hour = np.arctan2(sin_hour, cos_hour)
    if theta_hour < 0:
        theta_hour += 2 * np.pi
    A = theta_hour * 12 / (2 * np.pi)
    hours = int(A % 12)
    
    theta_minute = np.arctan2(sin_minute, cos_minute)
    if theta_minute < 0:
        theta_minute += 2 * np.pi
    A = theta_minute * 60 / (2 * np.pi)
    minutes = int(A % 60)
    return hours, minutes


# In[46]:


y_pred = model.predict(X_test)
print(y_pred)

pred_times = np.array([sin_cos_to_time(sin_hour, cos_hour, sin_minute, cos_minute) for sin_hour, cos_hour, sin_minute, cos_minute in y_pred])
true_times = np.array([sin_cos_to_time(sin_hour, cos_hour, sin_minute, cos_minute) for sin_hour, cos_hour, sin_minute, cos_minute in  y_test])
print(common_error(pred_times, true_times))

