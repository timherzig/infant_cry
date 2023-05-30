import os
import wandb
import tensorflow as tf

from keras import losses
from omegaconf import OmegaConf
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from utils.metrics import get_f1
from data.babycry import BabyCry
from model.trill_extended import trill
from utils.parser import parse_arguments


def main(args):
    # Load config & initialize wandb
    config = OmegaConf.load(args.config)
    os.makedirs("checkpoints", exist_ok=True)
    name = (
        config.model.name
        + f'_{len([x for x in os.listdir("checkpoints") if config.model.name in x]) + 1}'
    )
    wandb.init(
        project="trill_babycry",
        config=config.train,
        name=name,
    )
    save_dir = "checkpoints/" + name
    os.makedirs(save_dir)
    OmegaConf.save(config, os.path.join(save_dir, "config.yaml"))

    # Initialize datasets
    val_dataset = BabyCry(
        config.data.dir, "val", config.train.batch_size, config.data.spec
    )
    test_dataset = BabyCry(
        config.data.dir, "test", config.train.batch_size, config.data.spec
    )
    train_dataset = BabyCry(
        config.data.dir, "train", config.train.batch_size, config.data.spec
    )

    # Initialize model
    if args.checkpoint == None:
        model = trill(config.model)
    else:
        model = tf.keras.models.load_model(args.checkpoint)

    model.compile(
        optimizer=config.train.optimizer,
        loss=losses.BinaryCrossentropy(),
        metrics=[get_f1, "accuracy"],
    )

    model.summary()

    # Train model
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        callbacks=[
            WandbMetricsLogger(),
            WandbModelCheckpoint(save_dir, monitor="val_loss", mode="min"),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )

    model.evaluate(
        test_dataset,
        batch_size=config.train.batch_size,
    )

    print(f"Done! Model saved to {save_dir}")


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
