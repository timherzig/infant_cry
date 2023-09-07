import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from keras import layers

# JDC packages
from model.melodyExtraction_JDC.model import *
from model.melodyExtraction_JDC.featureExtraction import *


class Options(object):
    def __init__(self, config) -> None:
        self.num_spec = config.jdc.num_spec
        self.input_size = config.jdc.input_size
        self.batch_size = config.train.batch_size
        self.resolution = config.jdc.resolution
        self.figureON = config.jdc.figureON


def jdc(config):
    options = Options(config)

    pitch_range = np.arange(38, 83 + 1.0 / options.resolution, 1.0 / options.resolution)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    spec_extraction = spec_extraction()

    input = layers.Input(shape=(options.input_size, options.num_spec, 1))

    # JDC model
    jdc_model = melody_ResNet_joint_add(options)
    jdc_model.load_weights(os.path.join(os.getcwd, config.jdc.model_path))
    jdc_model.trainable = config.jdc.trainable

    jdc_output = jdc_model(input)
    print(f"jdc output: {jdc_output}")

    embeddings = jdc_output["output"]

    x = layers.Dropout(config.dropout)(embeddings)  # embeddings

    if config.bi_lstm.use:
        x = tf.expand_dims(x, axis=1)
        x = layers.Bidirectional(layers.LSTM(config.bi_lstm.units))(x)
        # x = layers.Flatten()(x)
    else:
        x = layers.Flatten()(x)
        x = layers.Dense(config.dense, activation="relu")(x)

    # Convolutional layer used for visualizing/XAI Tim Impl.
    x = layers.Reshape((x.shape[1], 1, 1))(x)
    x = layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation="relu")(
        x
    )
    # x = layers.MaxPooling2D(pool_size=(2,1))(x)
    x = layers.Flatten()(x)

    x = layers.Dense(config.dense, activation="relu")(x)
    predictions = layers.Dense(config.output, activation="sigmoid")(x)

    jdc_extended = tf.keras.Model(inputs=jdc_model.input, outputs=predictions)

    return jdc_extended, spec_extraction, options
