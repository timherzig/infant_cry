import os
import csv
import itertools
import pandas as pd
import tensorflow as tf

from keras import losses, optimizers
from keras.callbacks import (
    EarlyStopping,
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from data.babycry import BabyCry
from utils.metrics import get_f1, weighted_ce_loss
from utils.calculate_loso_results import calc_loso_results
from model.trill_extended import trill


def train_single(
    train_speakers: list, val_speakers: list, args: dict, config: dict, save_dir: str
):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Initialize model
    if config.model.name == "trill":
        model = trill(config)
        spec_extraction = options = None
        print(f"Got TRILL model: {model}")

    callbacks = [
        TensorBoard(log_dir=f"{save_dir}/logs", histogram_freq=1),
        ModelCheckpoint(
            f"{save_dir}/model/mdl_wts.hdf5",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    if config.train.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
        )

    # Initialize datasets
    val_dataset = BabyCry(
        config.data.dir,
        "val" if train_speakers is None else "train_loso",
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
        "train" if train_speakers is None else "train_loso",
        config.train.batch_size,
        config.data.spec,
        train_speakers,
        spec_extraction=spec_extraction,
        options=options,
        augment=config.data.augment,
        mix_up=config.data.mix_up,
        mix_up_alpha=config.data.mix_up_alpha,
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
        optimizer=optimizers.Adam(learning_rate=config.train.learning_rate),
        loss=(
            weighted_ce_loss(weights)
            if config.data.mix_up <= 0.0
            else losses.CategoricalCrossentropy()
        ),
        metrics=[get_f1, "accuracy"],
    )

    model.summary()

    # Train model
    if args.test is True:
        model.load_weights(f"{save_dir}/model/mdl_wts.hdf5")
    else:
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

    dev_loss, dev_f1, dev_acc = model.evaluate(
        val_dataset,
        batch_size=config.train.batch_size,
    )

    if train_speakers is None:
        model.save_weights(f"{save_dir}/model.h5")
    # model.save_weights(f"{save_dir}/model.h5")

    del model
    del train_dataset
    del val_dataset
    del test_dataset

    return loss, f1, acc, dev_loss, dev_f1, dev_acc


def train_loso(
    args: dict,
    config: dict,
    save_dir: str,
):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    completed_speakers = []
    if os.path.exists(f"{save_dir}/loso_results.csv"):
        with open(f"{save_dir}/loso_results.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            completed_speakers = [row[0] for row in reader]

    val_speakers = pd.read_csv(os.path.join(config.data.dir, "val.csv"))["id"].unique()
    train_speakers = pd.read_csv(os.path.join(config.data.dir, "train.csv"))[
        "id"
    ].unique()
    ger_val_speakers = [
        s for s in itertools.chain(val_speakers, train_speakers) if s.startswith("G")
    ]
    jpn_val_speakers = [
        s for s in itertools.chain(val_speakers, train_speakers) if s.startswith("J")
    ]

    if len(ger_val_speakers) > len(jpn_val_speakers):
        # ger_val_speakers = ger_val_speakers[: len(jpn_val_speakers)]
        jpn_val_speakers = list(
            itertools.islice(itertools.cycle(jpn_val_speakers), len(ger_val_speakers))
        )
    elif len(jpn_val_speakers) > len(ger_val_speakers):
        # jpn_val_speakers = jpn_val_speakers[: len(ger_val_speakers)]
        ger_val_speakers = list(
            itertools.islice(itertools.cycle(ger_val_speakers), len(jpn_val_speakers))
        )

    loso_results = {}

    for i in range(0, len(ger_val_speakers) - 1):
        print(f"loop {i} of {len(ger_val_speakers) - 1}")
        if f"val_{ger_val_speakers[i]}_{jpn_val_speakers[i]}" in completed_speakers:
            print(
                f"Skipping training for GER/JPN val speaker {ger_val_speakers[i]}/{jpn_val_speakers[i]}"
            )
            continue

        if jpn_val_speakers[i] == "J30":  # Skip this pair, unknown CUDA failure
            continue

        print(
            f"Training for GER/JPN val speaker {ger_val_speakers[i]}/{jpn_val_speakers[i]}"
        )

        cur_val_speakers = [ger_val_speakers[i], jpn_val_speakers[i]]
        cur_train_speakers = [
            s
            for s in itertools.chain(train_speakers, val_speakers)
            if (s != ger_val_speakers[i] and s != jpn_val_speakers[i])
        ]

        print(f"Train speakers: {cur_train_speakers}")
        print(f"Val speakers: {cur_val_speakers}")
        _, f1, acc, dev_loss, dev_f1, dev_acc = train_single(
            cur_train_speakers,
            cur_val_speakers,
            args,
            config,
            f"{save_dir}/val_{ger_val_speakers[i]}_{jpn_val_speakers[i]}",
        )

        loso_results[
            f"{ger_val_speakers[i]}_{ger_val_speakers[i+1]}_{jpn_val_speakers[i]}_{jpn_val_speakers[i+1]}"
        ] = [f1, acc, dev_loss, dev_f1, dev_acc]

        if not os.path.exists(f"{save_dir}/loso_results.csv"):
            with open(f"{save_dir}/loso_results.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "val_speakers",
                        "f1",
                        "acc",
                        "loss",
                        "dev_f1",
                        "dev_acc",
                    ]
                )

        with open(f"{save_dir}/loso_results.csv", "a") as f:
            str = [
                f"val_{ger_val_speakers[i]}_{jpn_val_speakers[i]}",
                f1,
                acc,
                dev_loss,
                dev_f1,
                dev_acc,
            ]
            writer = csv.writer(f)
            writer.writerow(str)

    avg_f1, avg_acc, avg_dev_f1, avg_dev_acc, max_f1, max_val_f1, min_f1, min_val_f1 = (
        calc_loso_results(f"{save_dir}/loso_results.csv")
    )

    return (
        "N/A",
        avg_f1,
        avg_acc,
        "N/A",
        avg_dev_f1,
        avg_dev_acc,
        max_f1,
        max_val_f1,
        min_f1,
        min_val_f1,
    )
