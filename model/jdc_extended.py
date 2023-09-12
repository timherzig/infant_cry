import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from keras import layers

# JDC packages
from model.melodyExtraction_JDC.model import *
from model.melodyExtraction_JDC.featureExtraction import spec_extraction


class Options(object):
    def __init__(self, config) -> None:
        self.num_spec = config.model.jdc.num_spec
        self.input_size = config.model.jdc.input_size
        self.batch_size = config.train.batch_size
        self.resolution = config.model.jdc.resolution
        self.figureON = config.model.jdc.figureON


class JDC(layers.Layer):
    def __init__(self, jdc_model):
        super(JDC, self).__init__()
        self.jdc_model = jdc_model

    def compute_output_shape(self, input_shape):
        return self.jdc_model.compute_output_shape(input_shape)
        # return (None, 31, 2)

    def call(self, input):
        jdc_out = self.jdc_model(input)
        return jdc_out


def jdc(config):
    options = Options(config)

    pitch_range = np.arange(38, 83 + 1.0 / options.resolution, 1.0 / options.resolution)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    # input = layers.Input(shape=(options.input_size, options.num_spec, 1))

    # JDC model
    jdc_model = melody_ResNet_joint_add(options)
    jdc_model.load_weights(os.path.join(os.getcwd(), config.model.jdc.model_path))
    jdc_model.trainable = config.model.jdc.trainable

    jdc_model.summary()

    jdc_model = JDC(jdc_model)

    input = layers.Input(shape=(None, options.input_size, options.num_spec, 1))

    # print("JDC extended model input shape:", input.shape)
    # print(f"type jdc: {type(jdc_model)}")

    embeddings = layers.TimeDistributed(jdc_model)(input)
    x = layers.Dropout(config.model.dropout)(embeddings[0])  # embeddings
    x = tf.reduce_mean(x, axis=1)

    if config.model.bi_lstm.use:
        # x = tf.expand_dims(x, axis=1)
        x = layers.Bidirectional(layers.LSTM(config.model.bi_lstm.units))(x)
        x = layers.Flatten()(x)
    else:
        x = layers.Flatten()(x)
        x = layers.Dense(config.model.dense, activation="relu")(x)

    # Convolutional layer used for visualizing/XAI Tim Impl.
    x = layers.Reshape((x.shape[1], 1, 1))(x)
    x = layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation="relu")(
        x
    )
    # x = layers.MaxPooling2D(pool_size=(2,1))(x)
    x = layers.Flatten()(x)

    x = layers.Dense(config.model.dense, activation="relu")(x)
    predictions = layers.Dense(config.model.output, activation="sigmoid")(x)

    jdc_extended = tf.keras.Model(inputs=input, outputs=predictions)

    return jdc_extended, spec_extraction, options
