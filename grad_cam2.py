import numpy as np
from argparse import ArgumentParser

# Model
import tensorflow as tf
from tensorflow import keras
from utils.metrics import get_f1

# Display
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from IPython.display import Image

# Data
import random
from data.babycry import BabyCry


def make_gradcam_heatmap(
    input, model, last_conv_layer_name, classification_layer_names
):
    """
    Makes a Grad-CAM heatmap for a given input and model.

    Parameters:
        input: The audio-input to analyse.
        model: The model to use for the Grad-CAM analysis.
        last_conv_layer_name: The name of the last convolutional layer in the model.
        classification_layer_names: The names of the classification layers in the model.
    """

    # Split model into two parts: until last conv layer and from last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classification_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Compute the gradient of the top predicted class for our audio-input
    with tf.GradientTape() as tape:
        # Get the output of the last conv layer
        last_conv_layer_output = last_conv_layer_model(input)
        print(f'Conv layer output shape: {last_conv_layer_output.shape}')
        tape.watch(last_conv_layer_output)
        # Get the outout of the 'original' model
        preds = classifier_model(last_conv_layer_output)
        # top_class_channel = preds[:, tf.argmax(preds[0])]
        print(f'Preds: {preds}')
        top_pred_index = tf.argmax(preds[0])
        print(f'Top pred index: {top_pred_index}')
        top_class_channel = preds[:, top_pred_index]
        print(f'Top class: {top_class_channel}')

    # Gradient of the top predicted class with regard to the output of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    print(f'Grads: {grads}')

    # Mean gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Average over all the channels to get the heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # Normalize the heatmap between 0 and 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap


def load_audio_sample(path: str):
    # Might not be used
    return


def main(
    model_path: str,
    dataset_root: str,
    last_conv_layer_name: str = "conv2d",
    classification_layer_names: list = [
        "flatten",
        "dense",
        "dense_1",
    ],
):
    model = keras.models.load_model(model_path, custom_objects={"get_f1": get_f1})
    print(f"Loaded model from {model_path}:")
    model.summary()
    print("-------------------------")

    # Load and prepare the audio sample
    test_dataset = BabyCry(dataset_root, "test", 1, False)
    audio, label = test_dataset.__getitem__(random.randint(0, test_dataset.__len__()))
    print(f"Loaded audio sample with label {label} from {dataset_root} test split.")
    print("-------------------------")
    print(f"Audio shape: {audio.shape}")
    print("-------------------------")

    print("Making Grad-CAM heatmap...")
    heatmap = make_gradcam_heatmap(
        audio, model, last_conv_layer_name, classification_layer_names
    )
    print("Done making Grad-CAM Heatmap.")
    print("-------------------------")

    print(f"Heatmap shape: {heatmap.shape}")
    print("Plotting heatmap...")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--audio_path",
        type=str,
        default="/netscratch/herzig/datasets/BabyCry_no_augment",
    )
    parser.add_argument("--model_path", type=str, default="checkpoints/trill1_22")
    parser.add_argument("--last_conv_layer_name", type=str, default="conv2d")
    args = parser.parse_args()

    main(args.model_path, args.audio_path, args.last_conv_layer_name)