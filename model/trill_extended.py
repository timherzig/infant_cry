import tensorflow as tf
import tensorflow_hub as hub

from keras import layers, losses


def trill(config):
    input = layers.Input(shape=(None,))
    m = hub.KerasLayer(config.trill.hub_path, trainable=config.trill.trainable)

    # NOTE: Audio should be floats in [-1, 1], sampled at 16kHz. Model input is of
    # the shape [batch size, time].
    # audio_samples = tf.zeros([3, 64000])

    embeddings = m(input)["embedding"]
    # embeddings = tf.expand_dims(embeddings, axis=1)
    # x = layers.Conv1D(1, 1)(embeddings)
    x = layers.Dropout(config.d)(embeddings)  # embeddings

    if config.bi_lstm:
        x = tf.expand_dims(x, axis=1)
        x = layers.Bidirectional(layers.LSTM(config.bi_lstm.units))(x)
        x = layers.Flatten()(x)
    else:
        x = layers.Flatten()(x)
        x = layers.Dense(config.dense, activation="relu")(x)

    x = layers.Dense(config.dense, activation="relu")(x)
    predictions = layers.Dense(config.output, activation="sigmoid")(x)

    trill_pretrained = tf.keras.Model(inputs=m.input, outputs=predictions)

    return trill_pretrained
