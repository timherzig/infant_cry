import tensorflow as tf
import tensorflow_hub as hub

from keras import layers, losses


def trill(config):
    input = layers.Input(shape=(None,))
    m = hub.KerasLayer(config.trill.hub_path, trainable=config.trill.trainable)

    embeddings = m(input)["embedding"]
    x = layers.Dropout(config.dropout)(embeddings)  # embeddings

    # # Add 2D Convolutional layer after the Dropout layer
    # x = layers.Reshape((-1, 128, 1))(x)  # Reshape the input to a 2D array
    # x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Add MaxPooling layer after the Conv2D

    if config.bi_lstm.use:
        x = tf.expand_dims(x, axis=1)
        x = layers.Bidirectional(layers.LSTM(config.bi_lstm.units))(x)
        x = layers.Flatten()(x)
    else:
        x = layers.Flatten()(x)
        x = layers.Dense(config.dense, activation="relu")(x)

    print(x.shape)
    x = layers.Reshape((1, 1, x.shape))(x)
    print(x.shape)
    x = layers.Conv2D(
        filters=1, kernel_size=(1, 1), padding="valid", activation="linear"
    )(x)
    print(x.shape)

    x = layers.Dense(config.dense, activation="relu")(x)
    predictions = layers.Dense(config.output, activation="sigmoid")(x)

    trill_pretrained = tf.keras.Model(inputs=m.input, outputs=predictions)

    return trill_pretrained
