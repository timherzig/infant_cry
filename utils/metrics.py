import tensorflow as tf
from keras import backend as K


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def weighted_ce_loss(class_weights):
    class_weights = tf.constant(class_weights, dtype=tf.float32)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, axis=-1)

    def loss(y_true, y_pred):
        weights = tf.reduce_sum(class_weights * y_true, axis=1)
        unweighted_losses = loss_fn(y_true, y_pred)
        weighted_losses = unweighted_losses * weights
        return tf.reduce_mean(weighted_losses)

    return loss
