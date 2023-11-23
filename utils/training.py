import tensorflow as tf

from keras import losses

from data.babycry import BabyCry
from utils.metrics import get_f1
from model.jdc_extended import jdc
from model.trill_extended import trill


def train_single(
    train_speakers: list,
    val_speakers: list,
    args: dict,
    config: dict,
    save_dir: str,
):
    # Initialize model
    if config.model.name == "trill":
        model = trill(config)
        spec_extraction = options = None
        print(f"Got TRILL model: {model}")
    elif config.model.name == "jdc":
        model, spec_extraction, options = jdc(config)
        print(f"Got JDC model: {model}")

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
        spec_extraction=spec_extraction,
        options=options,
        augment=0,
        rir_dir=config.data.rir_dir,
    )

    train_dataset = BabyCry(
        config.data.dir,
        "train",
        config.train.batch_size,
        config.data.spec,
        train_speakers,
        spec_extraction=spec_extraction,
        options=options,
        augment=config.data.augment,
        rir_dir=config.data.rir_dir,
        save_audio=f"{save_dir}/example_train_audio",
    )

    test_dataset = BabyCry(
        config.data.dir,
        "test",
        config.train.batch_size,
        config.data.spec,
        spec_extraction=spec_extraction,
        options=options,
        augment=0,
        rir_dir=config.data.rir_dir,
    )

    # Train model
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
    )

    loss, f1, acc = model.evaluate(
        test_dataset,
        batch_size=config.train.batch_size,
    )

    model.save_weights(f"{save_dir}/model.h5")

    return loss, f1, acc
