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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "jdc_model": self.jdc_model,
            }
        )
        return config


def jdc(config):
    options = Options(config)

    pitch_range = np.arange(38, 83 + 1.0 / options.resolution, 1.0 / options.resolution)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    # JDC model
    jdc_model = melody_ResNet_joint_add(options)
    jdc_model.load_weights(os.path.join(os.getcwd(), config.model.jdc.model_path))
    jdc_model.trainable = config.model.jdc.trainable

    # jdc_model.summary()

    jdc_model = JDC(jdc_model)

    input = layers.Input(shape=(None, options.input_size, options.num_spec, 1))

    embeddings = layers.TimeDistributed(jdc_model)(input)
    x = layers.Dropout(config.model.dropout)(embeddings[0])  # embeddings

    print(f"JDC output shape: {x.shape}")
    # Delta features, how much the feature changes from one frame to the next
    delta = x[:, 1:, :] - x[:, :-1, :]
    delta = tf.concat([tf.zeros_like(delta[:, :1, :]), delta], axis=1)
    x = tf.stack([x, delta], axis=2)
    print(f"X after delta shape: {x.shape}")
    # At this point we have a tensor containing stacked features and delta features for each frame
    # x has the shape (batch_size, num_frames, 2, 31, 722)

    if config.model.bi_lstm.use:
        x = layers.Reshape((-1, x.shape[2] * x.shape[3] * x.shape[4]))(x)
        x = layers.Dropout(config.model.dropout)(x)
        print(f"Reshaped x: {x.shape}")
        x = layers.Bidirectional(
            layers.LSTM(
                config.model.bi_lstm.units,
            ),
        )(x)
        x = layers.Flatten()(x)
    elif config.model.gru.use:
        x = layers.Reshape((-1, x.shape[2] * x.shape[3] * x.shape[4]))(x)
        x = layers.Dropout(config.model.dropout)(x)
        print(f"Reshaped x: {x.shape}")
        x = layers.GRU(
            config.model.gru.units,
        )(x)
        x = layers.Flatten()(x)
    else:
        x = layers.Flatten()(x)
        x = layers.Dense(config.model.dense, activation="relu")(x)

    x = layers.Dense(config.model.dense, activation="relu")(x)
    predictions = layers.Dense(config.model.output, activation="sigmoid")(x)

    jdc_extended = tf.keras.Model(inputs=input, outputs=predictions)

    return jdc_extended, spec_extraction, options
