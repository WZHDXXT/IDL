#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np

import tensorflow as tf
import os
import matplotlib.pyplot as plt

import keras
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, GlobalAveragePooling2D, Dense, LeakyReLU, Reshape, BatchNormalization, Dropout, Activation


# # Train a VAE model with cartoon face dataset

# Firstly, we train a VAE model with cartoon face dataset.

# Load dataset and plot the first 9 pictures

# In[2]:


def load_real_samples(scale=False, img_size=(64, 64), limit=10000):
    image_paths = glob('/kaggle/input/cartoon-faces-googles-cartoon-set/cartoonset100k_jpg/1/*')
    image_paths = image_paths[:limit]
    
    images = []
    for path in image_paths:
        img = load_img(path, target_size=img_size)
        img_array = img_to_array(img)
        images.append(img_array)
    X = np.array(images, dtype=np.float32)
    if scale:
        X = (X - 127.5) / 127.5  # Scale to [-1, 1]
    else:
        X = X / 255.0 
    return X


# In[3]:


def grid_plot(images, epoch='', name='', n=3, save=False, scale=False):

    if scale:
        images = (images + 1) / 2.0
    
    if hasattr(images, "numpy"): 
        images = images.numpy()
    
    plt.figure(figsize=(n * 2, n * 2)) 
    for index in range(n * n):
        plt.subplot(n, n, index + 1)
        plt.axis('off')
        plt.imshow(images[index].astype('float32')) 

    fig = plt.gcf()
    fig.suptitle(name + '  ' + str(epoch), fontsize=14)
    
    if save:
        filename = f'results/generated_plot_e{epoch:03d}_{name}.png'
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


# In[4]:


dataset = load_real_samples()


# In[5]:


seed = 42 
np.random.seed(seed)
random_indices = np.random.choice(dataset.shape[0], size=9, replace=False)
selected_images = dataset[random_indices]
grid_plot(selected_images, name='annotated-anime-faces dataset 64*64*3', n=3)


# build encoder model
# 
# 

# In[6]:


def build_conv_net(in_shape, out_shape, conv_structure):
    """
    Build a convolutional network based on a customizable structure.
    """
    input = tf.keras.Input(shape=in_shape)
    x = input

    for i, layer_args in enumerate(conv_structure):
        x = Conv2D(**layer_args, name=f'enc_conv_{i}')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = Flatten()(x)
    # x = GlobalAveragePooling2D()(x)
    x = Dense(out_shape, activation='sigmoid', name='enc_output')(x)

    model = tf.keras.Model(inputs=input, outputs=x, name='Encoder')
    model.summary()
    return model


# In[7]:


def build_deconv_net(latent_dim, conv_structure, target_shape=(64, 64, 3)):
    input = tf.keras.Input(shape=(latent_dim,))
    
    reversed_structure = conv_structure[::-1]
    num = len(conv_structure)
    target_height, target_width, target_channels = target_shape
    initial_height = target_height // (2 ** num)  # Downscaled size after num upsampling layers
    initial_width = target_width // (2 ** num)

    last_conv_shape = reversed_structure[0].get('filters', 128)  # Use the last filter count from the reversed structure
    x = Dense(initial_height * initial_width * last_conv_shape, activation='relu', name='dec_input')(input)
    x = Reshape((initial_height, initial_width, last_conv_shape))(x)  # Start with calculated spatial dimension
    x = Dropout(0.2)(x)

    for i, layer_args in enumerate(reversed_structure):
        x = Conv2DTranspose(
            name=f'dec_deconv_{i}',
            filters=layer_args.get('filters', 64), 
            kernel_size=layer_args.get('kernel_size', (3, 3)),
            strides=layer_args.get('strides', (2, 2)),  
            padding=layer_args.get('padding', 'same'),
            activation=layer_args.get('activation', 'relu'),
        )(x)
    # x = BatchNormalization()(x)
    x = Conv2DTranspose(
        filters=target_channels,  # Match the target channels
        kernel_size=(3, 3),
        strides=(1, 1),  # No further upsampling
        padding='same',
        activation='tanh',  
        name='dec_output'
    )(x)

    model = tf.keras.Model(inputs=input, outputs=x, name='Decoder')
    model.summary()
    return model


# build VAE model

# In[8]:


class Sampling(tf.keras.layers.Layer):
    """
    Custom layer for the variational autoencoder.
    """
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_var) * epsilon


def build_vae(data_shape, latent_dim, conv_structure):
    encoder = build_conv_net(data_shape, latent_dim * 2, conv_structure)

    # Add sampling layer
    z_mean = Dense(latent_dim, name='z_mean')(encoder.output)
    z_var = Dense(latent_dim, name='z_var')(encoder.output)
    z = Sampling()([z_mean, z_var])

    encoder = tf.keras.Model(inputs=encoder.input, outputs=z)

    # Build decoder using the reversed conv_structure
    decoder = build_deconv_net(latent_dim, conv_structure)

    # Connect encoder and decoder
    vae = tf.keras.Model(inputs=encoder.input, outputs=decoder(z))

    # Define KL loss layer
    class KLLossLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_var = inputs
            kl_loss = -0.5 * tf.reduce_sum(z_var - tf.square(z_mean) - tf.exp(z_var) + 1)
            self.add_loss(kl_loss / tf.cast(tf.keras.backend.prod(data_shape), tf.float32))
            return inputs

    # Apply KL loss layer
    _, _ = KLLossLayer()([z_mean, z_var])

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')

    return encoder, decoder, vae


# To better generate images, we design custome convolutional architecture, then build up VAE model.

# In[9]:


latent_dim = 64 
conv_structure = [
    {'filters': 64, 'kernel_size': (5, 5), 'strides': (2, 2), 'padding': 'same', 'activation': 'relu'},
    {'filters': 64, 'kernel_size': (5, 5), 'strides': (2, 2), 'padding': 'same', 'activation': 'relu'},
    # {'filters': 64, 'kernel_size': (5, 5), 'strides': (2, 2), 'padding': 'same', 'activation': 'relu'},
]

encoder, decoder, vae = build_vae(dataset.shape[1:], latent_dim, conv_structure)


# visualize the structure of encoder

# In[10]:


keras.utils.plot_model(encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True)


# visualize the structure of decoder.

# In[11]:


keras.utils.plot_model(decoder, to_file='decoder.png', show_shapes=True, show_layer_names=True)


# Then we start training VAE model with early stopping

# In[33]:


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='loss',  
    patience=10,  
    restore_best_weights=True  
)

losses = []
for epoch in range(50): 
    history = vae.fit(
        x=dataset, 
        y=dataset, 
        epochs=1, 
        batch_size=16, 
        callbacks=[early_stopping] 
    )
    epoch_loss = history.history['loss'][0]
    losses.append(epoch_loss)

    latent_vectors = np.random.randn(9, latent_dim) / 6 
    images = decoder(latent_vectors)
    grid_plot(images, epoch, name='VAE generated images (randomly sampled from the latent space)', n=3, save=False)


# we can visualize the loss vallue to see how it changes in the process

# In[34]:


plt.plot(range(50), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()


# #  linearly interpolating for VAE

# In[35]:


def generate_latent_pairs(latent_dim, num_pairs, scale=0.5, seed=2024):
    if seed is not None:
        np.random.seed(seed)
    return [(np.random.randn(1, latent_dim) * scale, np.random.randn(1, latent_dim) * scale) for _ in range(num_pairs)]

def interpolate_latent_vectors(latent_pairs, steps):
    all_interpolations = []
    for z1, z2 in latent_pairs:
        alphas = np.linspace(0, 1, steps)
        interpolation_vectors = (1 - alphas[:, None]) * z1 + alphas[:, None] * z2
        all_interpolations.append(interpolation_vectors)
    return all_interpolations

def decode_and_visualize(decoder, all_interpolations, num_transitions, steps, title="Interpolation Grid"):
    decoded_images = []
    for interpolation in all_interpolations:
        decoded_images.append(decoder.predict(interpolation))

    fig, axes = plt.subplots(num_transitions, steps, figsize=(steps * 2, num_transitions * 2))
    for i, row_images in enumerate(decoded_images):
        for j, img in enumerate(row_images):
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()


# The latent_dim defines the dimension of the latent space. 
# 
# Two random points, z1 and z2 in latent_pairs are generated in the latent space as the start and end points for interpolation. 
# 
# The steps parameter determines the number of interpolation steps
# 
# The decoder is used to decode the interpolation vectors into images. 
# 
# the gradual transition can be shown.

# In[36]:


latent_dim = 64
seed = 2024
num_transitions = 4
steps = 5
latent_pairs = generate_latent_pairs(latent_dim, num_transitions, seed=seed)
all_interpolations = interpolate_latent_vectors(latent_pairs, steps)
decode_and_visualize(decoder, all_interpolations, num_transitions, steps, title="Latent Space Interpolation")


# # Train a VAE model with dog face dataset

# In[ ]:


def load_real_samples(scale=False, img_size=(64, 64), limit=10000):
    image_paths = glob('/kaggle/input/animal-faces/afhq/train/dog/*')
    image_paths = image_paths[:limit]
    
    images = []
    for path in image_paths:
        img = load_img(path, target_size=img_size)
        img_array = img_to_array(img)
        images.append(img_array)
    X = np.array(images, dtype=np.float32)
    if scale:
        X = (X - 127.5) / 127.5  # Scale to [-1, 1]
    else:
        X = X / 255.0 
    return X


# In[ ]:


def grid_plot(images, epoch='', name='', n=3, save=False, scale=False):

    if scale:
        images = (images + 1) / 2.0
    
    if hasattr(images, "numpy"): 
        images = images.numpy()
    
    plt.figure(figsize=(n * 2, n * 2)) 
    for index in range(n * n):
        plt.subplot(n, n, index + 1)
        plt.axis('off')
        plt.imshow(images[index].astype('float32')) 

    fig = plt.gcf()
    fig.suptitle(name + '  ' + str(epoch), fontsize=14)
    
    if save:
        filename = f'results/generated_plot_e{epoch:03d}_{name}.png'
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


# In[ ]:


dataset = load_real_samples()


# In[ ]:


seed = 42 
np.random.seed(seed)
random_indices = np.random.choice(dataset.shape[0], size=9, replace=False)
selected_images = dataset[random_indices]
grid_plot(selected_images, name='annotated-anime-faces dataset 64*64*3', n=3)


# In[ ]:


latent_dim = 64 
conv_structure = [
    {'filters': 64, 'kernel_size': (5, 5), 'strides': (2, 2), 'padding': 'same', 'activation': 'relu'},
    {'filters': 64, 'kernel_size': (5, 5), 'strides': (2, 2), 'padding': 'same', 'activation': 'relu'},
    # {'filters': 64, 'kernel_size': (5, 5), 'strides': (2, 2), 'padding': 'same', 'activation': 'relu'},
]

encoder, decoder, vae = build_vae(dataset.shape[1:], latent_dim, conv_structure)


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='loss',  
    patience=10,  
    restore_best_weights=True  
)

losses = []
for epoch in range(50): 
    history = vae.fit(
        x=dataset, 
        y=dataset, 
        epochs=1, 
        batch_size=16, 
        callbacks=[early_stopping] 
    )
    epoch_loss = history.history['loss'][0]
    losses.append(epoch_loss)

    latent_vectors = np.random.randn(9, latent_dim) / 6 
    images = decoder(latent_vectors)
    grid_plot(images, epoch, name='VAE generated images (randomly sampled from the latent space)', n=3, save=False)


# # visualize the change of loss

# In[ ]:


plt.plot(range(50), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

