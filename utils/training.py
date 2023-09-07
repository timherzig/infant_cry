import tensorflow as tf

from keras import losses
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from model.trill_extended import trill
from model.jdc_extended import jdc
from utils.metrics import get_f1
from data.babycry import BabyCry


def train_single(
    train_speakers: list,
    val_speakers: list,
    test_dataset: BabyCry,
    args: dict,
    config: dict,
    save_dir: str,
):
    # Initialize model
    if args.checkpoint == None:
        if config.model == "trill":
            model = trill(config.model)
            spec_extraction = options = None
        elif config.model == "jdc":
            model, spec_extraction, options = jdc(config.model)
    else:
        model = tf.keras.models.load_model(args.checkpoint)

    model.compile(
        optimizer=config.train.optimizer,
        loss=losses.BinaryCrossentropy(),
        metrics=[get_f1, "accuracy"],
    )

    model.summary()

    # Initialize datasets
    val_dataset = BabyCry(
        config.data.dir,
        "val",
        config.train.batch_size,
        config.data.spec,
        val_speakers,
        spec_extraction,
        options,
    )

    train_dataset = BabyCry(
        config.data.dir,
        "train",
        config.train.batch_size,
        config.data.spec,
        train_speakers,
        spec_extraction,
        options,
    )

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

    return
