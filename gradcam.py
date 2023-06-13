# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import random

from argparse import ArgumentParser
from utils.metrics import get_f1
from data.babycry import BabyCry

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        if self.layerName is None:
            self.layerName = self.find_last_conv()
            
    def find_last_conv(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError('Could not find a 4D layer. Aborting')

    def compute_heatmap(self, audio, eps=1e-8):
        gradModel = Model(inputs=[self.model.inputs], outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        with tf.GradientTape() as tape:
            conv_out, pred = gradModel(audio)
            loss = pred[:, self.classIdx]

        grads = tape.gradient(loss, conv_out)

        cast_conv_out = tf.cast(conv_out > 0, 'float32')
        cast_grad = tf.cast(grads > 0, 'float32')

        guided_grads = cast_conv_out * cast_grad * grad

        weights = tf.reduce_mean(guided_grad, axis=(0,1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_out), axis=-1)

        print(len(cam))

        return cam



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--audio_path",
        type=str,
        default="/netscratch/herzig/datasets/BabyCry_no_augment",
    )
    parser.add_argument("--model_path", type=str, default="checkpoints/trill1_14")
    parser.add_argument("--last_conv_layer_name", type=str, default="conv2d")
    args = parser.parse_args()

     # Load audio
    test_dataset = BabyCry(args.audio_path, "test", 1, False)
    audio, label = test_dataset.__getitem__(random.randint(0, test_dataset.__len__()))

    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(args.model_path, custom_objects={"get_f1": get_f1})

        cam = GradCAM(model, 0)
        heatmap = cam.compute_heatmap(audio)