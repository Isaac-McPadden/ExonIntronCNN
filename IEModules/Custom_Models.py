import time
import sys
import os
import glob
import math
import threading
import concurrent.futures as cf
import random
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, layers, metrics, losses, callbacks, optimizers, models, utils
from keras import backend as K
import gc
import keras_tuner as kt
from pyfaidx import Fasta

import sys
import os

@utils.register_keras_serializable()
def tile_to_batch(z):
    pe, x = z
    return tf.tile(pe, [tf.shape(x)[0], 1, 1])

@utils.register_keras_serializable()
class LocalMaskLayer(layers.Layer):
    def __init__(self, radius=3, **kwargs):
        """
        Args:
            radius (int): The allowed distance for positions (default 3).
        """
        super(LocalMaskLayer, self).__init__(**kwargs)
        self.radius = radius

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]
        mask = tf.cast(tf.abs(i - j) <= self.radius, tf.float32)
        mask = tf.expand_dims(mask, 0)  # shape: (1, seq_len, seq_len)
        mask = tf.tile(mask, [tf.shape(inputs)[0], 1, 1])  # shape: (batch, seq_len, seq_len)
        return mask

    def get_config(self):
        config = super(LocalMaskLayer, self).get_config()
        config.update({'radius': self.radius})
        return config


# Helper fn to adjust dilation rates.
@utils.register_keras_serializable()
def adjust_dilation_rate(rate, dilation_multiplier):
    # Multiply by dilation_multiplier, round to nearest int, and make sure it is at least 1.
    new_rate = int(round(rate * dilation_multiplier))
    return new_rate if new_rate >= 1 else 1


@utils.register_keras_serializable()
def create_modular_dcnn_model(
    input_dim=5,
    sequence_length=5000,
    num_classes=5,
    use_local_attention=True,
    use_long_range_attention=True,
    use_final_attention=True,
    dilation_multiplier=1.0  # New parameter to scale the dilation rates
):
    

    inputs = Input(shape=(sequence_length, input_dim))
    
    # Condensed positional encoding block
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    pos_encoding = layers.Embedding(input_dim=sequence_length, output_dim=num_classes)(positions)
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)
    pos_encoding = layers.Lambda(tile_to_batch)([pos_encoding, inputs])
    
    concat_input = layers.Concatenate(axis=-1)([inputs, pos_encoding])
    
    # Option 1: Local Masked Attention after positional encoding
    if use_local_attention:
        local_mask = LocalMaskLayer()(concat_input)
        local_attn = layers.Attention(use_scale=True)([concat_input, concat_input], attention_mask=local_mask)
        # Residual connection
        concat_input = layers.Add()([concat_input, local_attn])
    
    # Hyperparameters for dropout
    early_dropout = 0.1
    middle_dropout = 0
    late_dropout = 0.2

    cnn = layers.Conv1D(filters=64, kernel_size=9, activation='relu', padding='same')(concat_input)
    cnn = layers.BatchNormalization()(cnn)
    cnn = layers.Dropout(early_dropout)(cnn)
    
    # Dilated convolution block
    skip = concat_input
    skip = layers.Conv1D(filters=64, kernel_size=1, padding='same')(skip)
    dcnn = layers.Conv1D(filters=64, kernel_size=9, dilation_rate=adjust_dilation_rate(1, dilation_multiplier),  # originally 1
        activation='relu', padding='same')(skip)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(early_dropout)(dcnn)
    low_dcnn = dcnn
    
    dcnn = layers.Conv1D(filters=64, kernel_size=9, dilation_rate=adjust_dilation_rate(2, dilation_multiplier),  # originally 2
        activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(early_dropout)(dcnn)
    dcnn = layers.Add()([dcnn, skip])
    
    skip = dcnn
    skip = layers.Conv1D(filters=160, kernel_size=1, padding='same')(skip)
    dcnn = layers.Conv1D(filters=160, kernel_size=9, dilation_rate=adjust_dilation_rate(4, dilation_multiplier),  # originally 4
        activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    
    dcnn = layers.Conv1D(filters=160, kernel_size=9, dilation_rate=adjust_dilation_rate(8, dilation_multiplier),  # originally 8
        activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    dcnn = layers.Add()([dcnn, skip])
    
    skip = dcnn
    skip = layers.Conv1D(filters=192, kernel_size=1, padding='same')(skip)
    dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=adjust_dilation_rate(16, dilation_multiplier),  # originally 16
        activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    
    dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=adjust_dilation_rate(32, dilation_multiplier),  # originally 32
        activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    dcnn = layers.Add()([dcnn, skip])
    
    skip = dcnn
    skip = layers.Conv1D(filters=192, kernel_size=1, padding='same')(skip)
    dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=adjust_dilation_rate(64, dilation_multiplier),  # originally 64
        activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    
    dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=adjust_dilation_rate(128, dilation_multiplier),  # originally 128
        activation='relu', padding='same')(dcnn)
    dcnn = layers.BatchNormalization()(dcnn)
    dcnn = layers.Dropout(middle_dropout)(dcnn)
    dcnn = layers.Add()([dcnn, skip])
    
    # Concatenate inputs from different paths
    second_concat = layers.Concatenate(axis=-1)([concat_input, cnn, dcnn, low_dcnn])
    
    # Option 2: Long-range Attention after pooling the final convolution outputs
    if use_long_range_attention:
        pool_size = 10  # You can adjust the pooling factor as needed.
        pooled = layers.MaxPooling1D(pool_size=pool_size, padding='same')(second_concat)
        long_attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(pooled, pooled)
        long_attn_upsampled = layers.UpSampling1D(size=pool_size)(long_attn)
        second_concat = layers.Concatenate(axis=-1)([second_concat, long_attn_upsampled])
    
    # Option 3: Final Attention to capture which outputs are most important
    if use_final_attention:
        final_attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(second_concat, second_concat)
        second_concat = layers.Add()([second_concat, final_attn])
    
    # Instead of flattening, use Conv1D (kernel_size=1) as dense layers.
    dense = layers.Conv1D(128, kernel_size=1, activation='relu')(second_concat)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(late_dropout)(dense)
    
    dense = layers.Conv1D(128, kernel_size=1, activation='relu')(dense)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(late_dropout)(dense)
    
    # Final classification layer (applied at every time step)
    outputs = layers.Conv1D(num_classes, kernel_size=1, activation='sigmoid')(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


# @utils.register_keras_serializable()
# def tile_to_batch(z):
#     pe, x = z
#     return tf.tile(pe, [tf.shape(x)[0], 1, 1])

# @utils.register_keras_serializable()
# def create_dcnn_model(
#     input_dim=5,
#     sequence_length=5000,
#     num_classes=5
# ):
#     inputs = Input(shape=(sequence_length, input_dim))
    
#     # Condensed positional encoding block.  See cnn for description
#     positions = tf.range(start=0, limit=sequence_length, delta=1)
#     pos_encoding = layers.Embedding(input_dim=sequence_length, output_dim=num_classes)(positions)
#     pos_encoding = tf.expand_dims(pos_encoding, axis=0)
#     # def tile_to_batch(z):
#     #     pe, x = z
#     #     return tf.tile(pe, [tf.shape(x)[0], 1, 1])
#     pos_encoding = layers.Lambda(tile_to_batch)([pos_encoding, inputs])

#     concat_input = layers.Concatenate(axis=-1)([inputs, pos_encoding])
    
#     '''Initial training hyperparameters'''
#     # early_dropout = 0
#     # middle_dropout = 0.1
#     # late_dropout = 0.2
    
#     '''Reducing false positive distance hyperparameters'''
#     early_dropout = 0.1
#     middle_dropout = 0
#     late_dropout = 0.2

#     cnn = layers.Conv1D(filters=64, kernel_size=9, activation='relu', padding='same')(concat_input)
#     cnn = layers.BatchNormalization()(cnn)
#     cnn = layers.Dropout(early_dropout)(cnn)
#     # We use six layers with increasing dilation rates to capture a wider receptive field.
#     # Dilating convolutional blocks with dropout (pooling is bad because exact sequence matters)
#     skip = concat_input
#     skip = layers.Conv1D(filters=64, kernel_size=1, padding='same')(skip)
#     dcnn = layers.Conv1D(filters=64, kernel_size=9, dilation_rate=1, activation='relu', padding='same')(skip)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(early_dropout)(dcnn)
#     low_dcnn = dcnn
    
#     dcnn = layers.Conv1D(filters=64, kernel_size=9, dilation_rate=2, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(early_dropout)(dcnn)
#     dcnn = layers.Add()([dcnn, skip])
    
#     skip = dcnn
#     skip = layers.Conv1D(filters=160, kernel_size=1, padding='same')(skip)
#     dcnn = layers.Conv1D(filters=160, kernel_size=9, dilation_rate=4, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
    
#     dcnn = layers.Conv1D(filters=160, kernel_size=9, dilation_rate=8, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
#     dcnn = layers.Add()([dcnn, skip])
    
#     skip = dcnn
#     skip = layers.Conv1D(filters=192, kernel_size=1, padding='same')(skip)
#     dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=16, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
    
#     dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=32, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
#     dcnn = layers.Add()([dcnn, skip])
    
#     skip = dcnn
#     skip = layers.Conv1D(filters=192, kernel_size=1, padding='same')(skip)
#     dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=64, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
    
#     dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=128, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
#     dcnn = layers.Add()([dcnn, skip])
    
#     # Tested adding more dilation layers here and got no improvement
    
#     second_concat = layers.Concatenate(axis=-1)([concat_input, cnn, dcnn, low_dcnn])

#     # Instead of flattening, use Conv1D with kernel_size=1 as dense layers:
#     dense = layers.Conv1D(128, kernel_size=1, activation='relu')(second_concat)
#     dense = layers.BatchNormalization()(dense)
#     dense = layers.Dropout(late_dropout)(dense)
    
#     dense = layers.Conv1D(128, kernel_size=1, activation='relu')(dense)
#     dense = layers.BatchNormalization()(dense)
#     dense = layers.Dropout(late_dropout)(dense)

#     # Final classification layer applied at every time step:
#     outputs = layers.Conv1D(num_classes, kernel_size=1, activation='sigmoid')(dense)

#     model = Model(inputs=inputs, outputs=outputs)
#     return model


# # Adding more layers and increasing dense filters to 512 did not improve results
# # skip = dcnn
# # skip = layers.Conv1D(filters=256, kernel_size=1, padding='same')(skip)
# # dcnn = layers.Conv1D(filters=256, kernel_size=9, dilation_rate=256, activation='relu', padding='same')(dcnn)
# # dcnn = layers.BatchNormalization()(dcnn)
# # dcnn = layers.Dropout(middle_dropout)(dcnn)

# # dcnn = layers.Conv1D(filters=256, kernel_size=9, dilation_rate=512, activation='relu', padding='same')(dcnn)
# # dcnn = layers.BatchNormalization()(dcnn)
# # dcnn = layers.Dropout(middle_dropout)(dcnn)
# # dcnn = layers.Add()([dcnn, skip])

# # skip = dcnn
# # skip = layers.Conv1D(filters=256, kernel_size=1, padding='same')(skip)
# # dcnn = layers.Conv1D(filters=256, kernel_size=9, dilation_rate=1024, activation='relu', padding='same')(dcnn)
# # dcnn = layers.BatchNormalization()(dcnn)
# # dcnn = layers.Dropout(middle_dropout)(dcnn)

# # dcnn = layers.Conv1D(filters=256, kernel_size=9, dilation_rate=2048, activation='relu', padding='same')(dcnn)
# # dcnn = layers.BatchNormalization()(dcnn)
# # dcnn = layers.Dropout(middle_dropout)(dcnn)
# # dcnn = layers.Add()([dcnn, skip])

# # @utils.register_keras_serializable()
# # def tile_to_batch(z):
# #     pe, x = z
# #     return tf.tile(pe, [tf.shape(x)[0], 1, 1])

# @utils.register_keras_serializable()
# class LocalMaskLayer(layers.Layer):
#     def __init__(self, radius=3, **kwargs):
#         """
#         Args:
#             radius (int): The allowed distance for positions (default 3).
#         """
#         super(LocalMaskLayer, self).__init__(**kwargs)
#         self.radius = radius

#     def call(self, inputs):
#         seq_len = tf.shape(inputs)[1]
#         i = tf.range(seq_len)[:, None]
#         j = tf.range(seq_len)[None, :]
#         mask = tf.cast(tf.abs(i - j) <= self.radius, tf.float32)
#         mask = tf.expand_dims(mask, 0)  # shape: (1, seq_len, seq_len)
#         mask = tf.tile(mask, [tf.shape(inputs)[0], 1, 1])  # shape: (batch, seq_len, seq_len)
#         return mask

#     def get_config(self):
#         config = super(LocalMaskLayer, self).get_config()
#         config.update({'radius': self.radius})
#         return config


# @utils.register_keras_serializable()
# def create_dcnn_model_with_attention(
#     input_dim=5,
#     sequence_length=5000,
#     num_classes=5,
#     use_local_attention=True,
#     use_long_range_attention=True,
#     use_final_attention=True
# ):
#     inputs = Input(shape=(sequence_length, input_dim))
    
#     # Condensed positional encoding block
#     positions = tf.range(start=0, limit=sequence_length, delta=1)
#     pos_encoding = layers.Embedding(input_dim=sequence_length, output_dim=num_classes)(positions)
#     pos_encoding = tf.expand_dims(pos_encoding, axis=0)
#     pos_encoding = layers.Lambda(tile_to_batch)([pos_encoding, inputs])
    
#     concat_input = layers.Concatenate(axis=-1)([inputs, pos_encoding])
    
#     # Option 1: Local Masked Attention after positional encoding
#     if use_local_attention:
#         local_mask = LocalMaskLayer()(concat_input)
#         local_attn = layers.Attention(use_scale=True)(
#             [concat_input, concat_input],
#             attention_mask=local_mask
#         )
#         # Residual connection
#         concat_input = layers.Add()([concat_input, local_attn])
    
#     # Hyperparameters for dropout
#     early_dropout = 0.1
#     middle_dropout = 0
#     late_dropout = 0.2

#     cnn = layers.Conv1D(filters=64, kernel_size=9, activation='relu', padding='same')(concat_input)
#     cnn = layers.BatchNormalization()(cnn)
#     cnn = layers.Dropout(early_dropout)(cnn)
    
#     # Dilated convolution block
#     skip = concat_input
#     skip = layers.Conv1D(filters=64, kernel_size=1, padding='same')(skip)
#     dcnn = layers.Conv1D(filters=64, kernel_size=9, dilation_rate=1, activation='relu', padding='same')(skip)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(early_dropout)(dcnn)
#     low_dcnn = dcnn
    
#     dcnn = layers.Conv1D(filters=64, kernel_size=9, dilation_rate=2, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(early_dropout)(dcnn)
#     dcnn = layers.Add()([dcnn, skip])
    
#     skip = dcnn
#     skip = layers.Conv1D(filters=160, kernel_size=1, padding='same')(skip)
#     dcnn = layers.Conv1D(filters=160, kernel_size=9, dilation_rate=4, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
    
#     dcnn = layers.Conv1D(filters=160, kernel_size=9, dilation_rate=8, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
#     dcnn = layers.Add()([dcnn, skip])
    
#     skip = dcnn
#     skip = layers.Conv1D(filters=192, kernel_size=1, padding='same')(skip)
#     dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=16, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
    
#     dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=32, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
#     dcnn = layers.Add()([dcnn, skip])
    
#     skip = dcnn
#     skip = layers.Conv1D(filters=192, kernel_size=1, padding='same')(skip)
#     dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=64, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
    
#     dcnn = layers.Conv1D(filters=192, kernel_size=9, dilation_rate=128, activation='relu', padding='same')(dcnn)
#     dcnn = layers.BatchNormalization()(dcnn)
#     dcnn = layers.Dropout(middle_dropout)(dcnn)
#     dcnn = layers.Add()([dcnn, skip])
    
#     # Concatenate inputs from different paths
#     second_concat = layers.Concatenate(axis=-1)([concat_input, cnn, dcnn, low_dcnn])
    
#     # Option 2: Long-range Attention after pooling the final convolution outputs
#     if use_long_range_attention:
#         pool_size = 10  # You can adjust the pooling factor as needed.
#         pooled = layers.MaxPooling1D(pool_size=pool_size, padding='same')(second_concat)
#         long_attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(pooled, pooled)
#         long_attn_upsampled = layers.UpSampling1D(size=pool_size)(long_attn)
#         second_concat = layers.Concatenate(axis=-1)([second_concat, long_attn_upsampled])
    
#     # Option 3: Final Attention to capture which outputs are most important
#     if use_final_attention:
#         final_attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(second_concat, second_concat)
#         second_concat = layers.Add()([second_concat, final_attn])
    
#     # Instead of flattening, use Conv1D (kernel_size=1) as dense layers.
#     dense = layers.Conv1D(128, kernel_size=1, activation='relu')(second_concat)
#     dense = layers.BatchNormalization()(dense)
#     dense = layers.Dropout(late_dropout)(dense)
    
#     dense = layers.Conv1D(128, kernel_size=1, activation='relu')(dense)
#     dense = layers.BatchNormalization()(dense)
#     dense = layers.Dropout(late_dropout)(dense)
    
#     # Final classification layer (applied at every time step)
#     outputs = layers.Conv1D(num_classes, kernel_size=1, activation='sigmoid')(dense)
    
#     model = Model(inputs=inputs, outputs=outputs)
#     return model
