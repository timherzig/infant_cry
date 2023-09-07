import os
import random
import librosa

import numpy as np
import tensorflow as tf
from tensorflow import keras

from argparse import ArgumentParser
from utils.metrics import get_f1
from data.babycry import BabyCry

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def save_and_display_gradcam(img, heatmap, cam_path="grad_cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


def make_gradcam_heatmap(input_array, model, last_conv_layer_name, pred_index=None):
    """
    From the TensorFlow Grad-CAM tutorial: https://keras.io/examples/vision/grad_cam/
    """

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(input_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    with tf.device("/CPU:0"):
        grads = tape.gradient(
            class_channel,
            last_conv_layer_output,
        )

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def grad_cam(dataset_root: str, model_path: str, last_conv_layer_name: str = "conv2d"):
    """
    Creates and saves a Grad-CAM image for the given audio file when classified using a pre-trained model.

    Parameters:
    ----------
        audio_path (str): path to audio file containing a baby-cry
        model_path (str): path to pre-trained model
        last_conv_layer_name (str): name of last convolutional layer in model, defaults to 'conv2d' (default for TRILL model in this repository)
    """

    # Load audio
    test_dataset = BabyCry(dataset_root, "test", 1, False)
    audio, label = test_dataset.__getitem__(random.randint(0, test_dataset.__len__()))

    model = tf.keras.models.load_model(model_path, custom_objects={"get_f1": get_f1})
    # model.layers[-1].activation = None

    # heatmap = make_gradcam_heatmap(audio, model, last_conv_layer_name)

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    # Save and display Grad CAM
    # save_and_display_gradcam(
    #     audio_spec_aug,
    #     heatmap,
    #     cam_path=model_path.split("/")[-1].split(".")[0]
    #     + "_"
    #     + audio_path.split("/")[-1].split(".")[0]
    #     + "_grad_cam.jpg",
    # )


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

    grad_cam(
        args.audio_path,
        args.model_path,
        last_conv_layer_name="conv2d",
    )
