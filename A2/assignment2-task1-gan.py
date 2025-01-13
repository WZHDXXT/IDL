#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tf_keras==2.17.0')


# In[2]:


import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
from glob import glob
from tensorflow.keras.utils import img_to_array, load_img
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


# # Train a GAN model with cartoon face dataset

# Load dataset and plot the first 9 pictures

# In[3]:


def load_real_samples(scale=False, img_size=(64, 64), limit=20000):
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

# We will use this function to display the output of our models throughout this notebook
def grid_plot(images, epoch='', name='', n=3, save=False, scale=False):
    if scale:
        images = (images + 1) / 2.0
    for index in range(n * n):
        plt.subplot(n, n, 1 + index)
        plt.axis('off')
        plt.imshow(images[index])
    fig = plt.gcf()
    fig.suptitle(name + '  '+ str(epoch), fontsize=14)
    if save:
        filename = 'results/generated_plot_e%03d_f.png' % (epoch+1)
        plt.savefig(filename)
        plt.close()
    plt.show()


dataset = load_real_samples()
grid_plot(dataset[np.random.randint(0, 1000, 9)], name='Fliqr dataset (64x64x3)', n=3)


# build generator and discriminator models
# 
# 

# In[4]:


from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape

def build_conv_net(in_shape, out_shape, n_downsampling_layers=4, filters=128, out_activation='sigmoid'):
    """
    Build a basic convolutional network
    """
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    input = tf.keras.Input(shape=in_shape)
    x = Conv2D(filters=filters, name='enc_input', **default_args)(input)

    for _ in range(n_downsampling_layers):
        x = Conv2D(**default_args, filters=filters)(x)

    x = Flatten()(x)
    x = Dense(out_shape, activation=out_activation, name='enc_output')(x)

    model = tf.keras.Model(inputs=input, outputs=x, name='Encoder')

    model.summary()
    return model


def build_deconv_net(latent_dim, n_upsampling_layers=4, filters=128, activation_out='sigmoid'):
    """
    Build a deconvolutional network for decoding/upscaling latent vectors

    When building the deconvolutional architecture, usually it is best to use the same layer sizes that
    were used in the downsampling network and the Conv2DTranspose layers are used instead of Conv2D layers.
    Using identical layers and hyperparameters ensures that the dimensionality of our output matches the
    shape of our input images.
    """
    input = tf.keras.Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 64, input_dim=latent_dim, name='dec_input')(input)
    x = Reshape((4, 4, 64))(x) # This matches the output size of the downsampling architecture

    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    for i in range(n_upsampling_layers):
        x = Conv2DTranspose(filters=filters, **default_args)(x)

    # This last convolutional layer converts back to 3 channel RGB image
    x = Conv2D(filters=3, kernel_size=(3,3), padding='same', activation=activation_out, name='dec_output')(x)

    model = tf.keras.Model(inputs=input, outputs=x, name='Decoder')
    model.summary()
    return model


# build up GAN model

# In[5]:


from tensorflow.keras.optimizers.legacy import Adam

def build_gan(data_shape, latent_dim, filters=128, lr=0.0002, beta_1=0.5):
    optimizer = Adam(learning_rate=lr, beta_1=beta_1)

    # Usually thew GAN generator has tanh activation function in the output layer
    generator = build_deconv_net(latent_dim, activation_out='tanh', filters=filters)

    # Build and compile the discriminator
    discriminator = build_conv_net(in_shape=data_shape, out_shape=1, filters=filters) # Single output for binary classification
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # End-to-end GAN model for training the generator
    discriminator.trainable = False
    true_fake_prediction = discriminator(generator.output)
    GAN = tf.keras.Model(inputs=generator.input, outputs=true_fake_prediction)
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer)

    return discriminator, generator, GAN


# In[6]:


def get_batch(generator, dataset, batch_size=64):
    """
    Fetches one batch of data and ensures no memory leaks by using TensorFlow operations.
    """
    half_batch = batch_size // 2

    # Generate fake images
    latent_vectors = tf.random.normal(shape=(half_batch, latent_dim))
    fake_data = generator(latent_vectors, training=False)

    # Select real images
    idx = np.random.randint(0, dataset.shape[0], half_batch)
    real_data = dataset[idx]

    # Combine
    X = tf.concat([real_data, fake_data], axis=0)
    y = tf.concat([tf.ones((half_batch, 1)), tf.zeros((half_batch, 1))], axis=0)

    return X, y


def train_gan(generator, discriminator, gan, dataset, latent_dim, n_epochs=20, batch_size=64):
    """
    Train the GAN with memory-efficient updates and clear session management.
    """
    batches_per_epoch = dataset.shape[0] // batch_size

    for epoch in range(n_epochs):
        for batch in tqdm(range(batches_per_epoch)):
            # Train Discriminator
            X, y = get_batch(generator, dataset, batch_size)
            discriminator_loss = discriminator.train_on_batch(X, y)

            # Train Generator
            latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
            y_gan = tf.ones((batch_size, 1))
            generator_loss = gan.train_on_batch(latent_vectors, y_gan)

        # Generate and visualize after each epoch
        noise = tf.random.normal(shape=(16, latent_dim))
        generated_images = generator(noise, training=False)
        grid_plot(generated_images.numpy(), epoch, name='Generated Images', n=3)

        # Clear backend session to free memory
        tf.keras.backend.clear_session()


# We first set dimension of latent space to be 512 to see the result then change it to 64 to see difference.

# In[10]:


latent_dim = 512
discriminator, generator, gan = build_gan(dataset.shape[1:], latent_dim, filters=128)
dataset_scaled = load_real_samples(scale=True)

train_gan(generator, discriminator, gan, dataset_scaled, latent_dim, n_epochs=50)


# # linear interpolation

# In[11]:


def generate_latent_pairs(latent_dim, num_pairs, scale=0.8, seed=2024):
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


# In[12]:


latent_dim = 512
seed = 2024
num_transitions = 5
steps = 5
latent_pairs = generate_latent_pairs(latent_dim, num_transitions, seed=seed)
all_interpolations = interpolate_latent_vectors(latent_pairs, steps)
decode_and_visualize(generator, all_interpolations, num_transitions, steps, title="Latent Space Interpolation")


# Set latent space dimension to be 64

# In[15]:


latent_dim = 64
discriminator, generator, gan = build_gan(dataset.shape[1:], latent_dim, filters=128)
dataset_scaled = load_real_samples(scale=True)

train_gan(generator, discriminator, gan, dataset_scaled, latent_dim, n_epochs=50)


# In[16]:


latent_dim = 64
seed = 2024
num_transitions = 5
steps = 5
latent_pairs = generate_latent_pairs(latent_dim, num_transitions, seed=seed)
all_interpolations = interpolate_latent_vectors(latent_pairs, steps)
decode_and_visualize(generator, all_interpolations, num_transitions, steps, title="Latent Space Interpolation")


# # Train a GAN model with dog face dataset

# In[58]:


def load_real_samples(scale=False, img_size=(64, 64), limit=20000):
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


# As images of dog faces are more complicate, we set dimension of latent space to be 1024 expecting better performance.

# In[62]:


latent_dim = 1024
discriminator, generator, gan = build_gan(dataset.shape[1:], latent_dim, filters=128)
dataset_scaled = load_real_samples(scale=True)

train_gan(generator, discriminator, gan, dataset_scaled, latent_dim, n_epochs=50)

