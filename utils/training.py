from keras import losses
from keras.callbacks import EarlyStopping, TensorBoard

from data.babycry import BabyCry
from utils.metrics import get_f1, weighted_ce_loss
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

    callbacks = [TensorBoard(log_dir=f"{save_dir}/logs", histogram_freq=1)]

    if config.train.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        )

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
        mic_dir=config.data.mic_dir,
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
        mic_dir=config.data.mic_dir,
        save_audio=f"{save_dir}/example_train_audio",
    )

    weights = train_dataset.get_weights()
    print(f"Weights [J, G]: {weights}")

    test_dataset = BabyCry(
        config.data.dir,
        "test",
        config.train.batch_size,
        config.data.spec,
        spec_extraction=spec_extraction,
        options=options,
        augment=0,
        rir_dir=config.data.rir_dir,
        mic_dir=config.data.mic_dir,
    )

    model.compile(
        optimizer=config.train.optimizer,
        loss=weighted_ce_loss(weights),
        metrics=[get_f1, "accuracy"],
    )

    model.summary()

    # Train model
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        callbacks=callbacks,
    )

    loss, f1, acc = model.evaluate(
        test_dataset,
        batch_size=config.train.batch_size,
    )

    model.save_weights(f"{save_dir}/model.h5")

    return loss, f1, acc
