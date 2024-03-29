import tensorflow as tf
import tensorflow_hub as hub

from keras import layers


def trill(config):
    config = config.model
    input = layers.Input(shape=(None,))
    m = hub.KerasLayer(config.trill.hub_path, trainable=config.trill.trainable)

    embeddings = m(input)["embedding"]
    x = layers.Dropout(config.dropout)(embeddings)  # embeddings

    if config.bi_lstm.use:
        x = tf.expand_dims(x, axis=1)
        x = layers.Bidirectional(layers.LSTM(config.bi_lstm.units))(x)
    else:
        x = layers.Flatten()(x)
        x = layers.Dense(config.dense, activation="relu")(x)

    # Convolutional layer used for visualizing/XAI Tim Impl.
    x = layers.Reshape((x.shape[1], 1, 1))(x)
    x = layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation="relu")(
        x
    )
    x = layers.Flatten()(x)

    x = layers.Dense(config.dense, activation="relu")(x)
    predictions = layers.Dense(config.output, activation="softmax")(x)

    trill_pretrained = tf.keras.Model(inputs=m.input, outputs=predictions)

    return trill_pretrained
