"""Train and evaluate a ResNet-50 based PSA grade classifier.

USAGE:
    card-grading-train --base_path /home/data/data_for_model_3

Training happens in two stages:
    1. warm-up: the ResNet-50 is frozen and only the classification head is trained
    2. fine-tuning: the ResNet-50 (or its last ``--unfreeze_after`` layers) is trained as well,
       with a low learning rate that decays over the epochs

Part of the training images is held out as a validation set. It is used to pick the best
epoch and for early stopping, so the test images are only used for the final evaluation.

Progress is saved after every epoch: pass the run's folder as ``--run_dir`` to continue a
run that was interrupted (or to train a finished run for more epochs).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from card_grading.data import CLASS_NAMES, GRADES, is_image, split_into_train_test

# Default (height, width): the cropped cards are portrait, 400 px wide
IMAGE_SIZE = (600, 400)

# The learning rate of the fine-tuning stage decays (cosine) to this share of its start value
FINAL_LR_FRACTION = 0.05

LOG_COLUMNS = ["stage", "epoch", "learning_rate", "loss", "val_exact", "val_within_1", "val_mean_abs_diff", "seconds"]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the PSA grade classifier.")
    parser.add_argument("-b", "--base_path", type=Path, required=True,
                        help="The base path for the data folder containing both /train and /test folders")
    parser.add_argument("-e", "--n_epochs", type=int, default=15,
                        help="Number of fine-tuning epochs (the second stage)")
    parser.add_argument("--head_epochs", type=int, default=3,
                        help="Number of warm-up epochs in which only the classification head is trained")
    parser.add_argument("-d", "--droprate", type=float, default=0.5,
                        help="The droprate for the dropout layers (float between 0 and 1)")
    parser.add_argument("--dense_units", type=int, nargs="*", default=[],
                        help="Sizes of extra dense layers between the ResNet50 and the prediction layer "
                             "(default: none)")
    parser.add_argument("--lr_dense", type=float, default=1e-3,
                        help="Learning rate for the warm-up stage")
    parser.add_argument("--lr_finetune", type=float, default=3e-5,
                        help="Starting learning rate for the fine-tuning stage")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--unfreeze_after", type=int, default=None,
                        help="How many ResNet50 layers, counted from the last, are fine-tuned (default: all)")
    parser.add_argument("--label_sigma", type=float, default=1.0,
                        help="Spread, in grades, of the training targets around the true grade, so that a near "
                             "miss counts as less wrong than a far miss (0 for plain one-hot targets)")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Share of the training images held out as the validation set")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Stop fine-tuning after this many epochs without improvement on the validation set "
                             "(0 to always train for n_epochs)")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply random flips, small shifts and contrast changes to the training images")
    parser.add_argument("--image_height", type=int, default=IMAGE_SIZE[0],
                        help="Height in pixels the images are resized to for the model")
    parser.add_argument("--image_width", type=int, default=IMAGE_SIZE[1],
                        help="Width in pixels the images are resized to for the model")
    parser.add_argument("--weights", type=str, default="imagenet",
                        help="ResNet50 weights: 'imagenet', a path to a weights file, or 'none' for random init")
    parser.add_argument("--log_root", type=Path, default=Path("logs"),
                        help="Folder in which each run gets its own folder")
    parser.add_argument("--run_dir", type=Path, default=None,
                        help="Folder of an earlier run to continue (default: start a new run in log_root)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the train/test split, the validation split and weight init")
    return parser.parse_args(argv)


# ---- data ---------------------------------------------------------------------------------


def list_images(path: Path) -> tuple[list[str], np.ndarray]:
    """The image files below ``path/<grade>/`` and their grades (1-10), in grade order."""
    paths: list[str] = []
    grades: list[int] = []
    for grade in GRADES:
        folder = path / str(grade)
        if not folder.is_dir():
            continue
        files = sorted(str(file) for file in folder.iterdir() if is_image(file))
        paths.extend(files)
        grades.extend([grade] * len(files))
    return paths, np.array(grades, dtype=int)


def split_validation(grades: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Boolean mask of the images held out for validation: the same share of every grade."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(grades), dtype=bool)
    for grade in np.unique(grades):
        indices = np.flatnonzero(grades == grade)
        mask[rng.choice(indices, int(round(len(indices) * fraction)), replace=False)] = True
    return mask


def ordinal_targets(grades: np.ndarray, sigma: float) -> np.ndarray:
    """Training targets for the grades: a distribution over the 10 grades centred on the true grade.

    With ``sigma`` 0 this is a plain one-hot target. Otherwise the neighbouring grades get part of
    the weight (a normal curve ``sigma`` grades wide), which tells the model that grades are ordered.
    """
    all_grades = np.array(GRADES, dtype="float32")
    if sigma <= 0:
        return (grades[:, None] == all_grades[None, :]).astype("float32")
    weights = np.exp(-0.5 * ((all_grades[None, :] - grades[:, None]) / sigma) ** 2)
    return (weights / weights.sum(axis=1, keepdims=True)).astype("float32")


def build_augmentation() -> keras.Sequential:
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.06, 0.06, fill_mode="nearest"),
        layers.RandomContrast(0.15),
    ])


def make_dataset(paths: list[str], grades: np.ndarray, image_size: tuple[int, int], batch_size: int, *,
                 label_sigma: float = 0.0, training: bool = False, augment: bool = False) -> tf.data.Dataset:
    """Batches of (images with values 0-255, targets) for the image files."""

    def load(path, target):
        # decode_image reads JPEG and PNG files alike
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return tf.image.resize(image, image_size), target

    dataset = tf.data.Dataset.from_tensor_slices((list(paths), ordinal_targets(np.asarray(grades), label_sigma)))
    if training:
        dataset = dataset.shuffle(len(paths), reshuffle_each_iteration=True)
    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training and augment:
        augmentation = build_augmentation()
        dataset = dataset.map(lambda x, y: (tf.clip_by_value(augmentation(x, training=True), 0.0, 255.0), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


# ---- model --------------------------------------------------------------------------------


def build_model(image_size: tuple[int, int] = IMAGE_SIZE, dense_units: tuple[int, ...] = (), droprate: float = 0.5,
                weights: str | None = "imagenet") -> tuple[keras.Model, keras.Model]:
    """Build the classifier. Returns ``(resnet50_base, full_model)``.

    The model takes images with values 0-255; the ResNet-50 preprocessing is part of it.
    """
    input_shape = tuple(image_size) + (3,)
    base = ResNet50(weights=weights, include_top=False, input_shape=input_shape)

    inputs = keras.Input(input_shape)
    # training=False keeps batch normalisation in inference mode, also while fine-tuning:
    # updating its statistics on a small dataset undoes what the pretrained weights learned
    x = base(preprocess_input(inputs), training=False)
    # Pool the feature map instead of flattening it: at the default image size, flattening
    # 19x13x2048 features into a dense layer would need ~500M weights
    x = layers.GlobalAveragePooling2D()(x)
    for units in dense_units:
        x = layers.Dropout(droprate)(x)
        x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(droprate)(x)
    predictions = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    return base, keras.Model(inputs=inputs, outputs=predictions)


def set_trainable_layers(base: keras.Model, n_trainable: int | None) -> None:
    """Make only the last ``n_trainable`` layers of the ResNet-50 trainable (``None`` for all, 0 for none)."""
    base.trainable = True
    first_trainable = 0 if n_trainable is None else len(base.layers) - n_trainable
    # every layer is set explicitly: assigning base.trainable does not reach the layers
    # when its value does not change
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= first_trainable


def load_weights(model: keras.Model, path: Path) -> None:
    """Load the layer weights saved by an earlier epoch.

    The optimizer state in the file is not used (every stage starts with a fresh optimizer),
    so the warning about it is silenced.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        model.load_weights(path)


def finetune_learning_rate(start: float, epoch: int, n_epochs: int) -> float:
    """Cosine decay from ``start`` to ``FINAL_LR_FRACTION`` of it over the fine-tuning epochs."""
    progress = min(epoch / max(n_epochs, 1), 1.0)
    return start * (FINAL_LR_FRACTION + (1 - FINAL_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * progress)))


# ---- evaluation ---------------------------------------------------------------------------


def predict_probabilities(model: keras.Model, dataset: tf.data.Dataset, flip: bool = False) -> np.ndarray:
    """Predicted grade probabilities; with ``flip`` averaged over the images and their mirror images."""
    probabilities = model.predict(dataset, verbose=0)
    if flip:
        mirrored = dataset.map(lambda x, y: (tf.image.flip_left_right(x), y))
        probabilities = (probabilities + model.predict(mirrored, verbose=0)) / 2
    return probabilities


def grade_metrics(probabilities: np.ndarray, true_grades: np.ndarray) -> dict:
    """How close the most likely grade is to the true grade."""
    predicted = np.argmax(probabilities, axis=1) + 1
    diffs = np.abs(predicted - true_grades)
    confusion = np.zeros((len(GRADES), len(GRADES)), dtype=int)
    np.add.at(confusion, (true_grades - 1, predicted - 1), 1)
    return {
        "n_images": int(len(true_grades)),
        "exact": float(np.mean(diffs == 0)),
        "within_1": float(np.mean(diffs <= 1)),
        "within_2": float(np.mean(diffs <= 2)),
        "mean_abs_diff": float(np.mean(diffs)),
        "median_abs_diff": float(np.median(diffs)),
        # rows: true grade 1-10, columns: predicted grade 1-10
        "confusion_matrix": confusion.tolist(),
    }


# ---- training run -------------------------------------------------------------------------


def create_run_dir(log_root: Path) -> Path:
    """Create a folder for this run.

    The pid keeps runs that start in the same second (e.g. a parallel grid search)
    from writing into the same folder.
    """
    run_dir = log_root / f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class TrainingRun:
    """One training run: its model, its data and the progress it keeps on disk."""

    def __init__(self, args: argparse.Namespace, run_dir: Path):
        self.args = args
        self.run_dir = run_dir
        self.state_file = run_dir / "state.json"
        self.last_weights = run_dir / "last.weights.h5"
        self.best_weights = run_dir / "best.weights.h5"
        self.log_file = run_dir / "log.csv"
        self.state = (json.loads(self.state_file.read_text()) if self.state_file.exists()
                      else {"head_epochs_done": 0, "finetune_epochs_done": 0, "best_mean_abs_diff": None,
                            "epochs_without_improvement": 0})

        image_size = (args.image_height, args.image_width)
        paths, grades = list_images(args.base_path / "train")
        if not paths:
            raise SystemExit(f"No training images found in {args.base_path / 'train'}")
        held_out = split_validation(grades, args.val_fraction, args.seed)
        train_paths = [path for path, is_val in zip(paths, held_out, strict=True) if not is_val]
        val_paths = [path for path, is_val in zip(paths, held_out, strict=True) if is_val]
        if not val_paths:
            raise SystemExit("The validation set is empty: use more training images or a larger --val_fraction")
        self.val_grades = grades[held_out]
        self.train_ds = make_dataset(train_paths, grades[~held_out], image_size, args.batch_size,
                                     label_sigma=args.label_sigma, training=True, augment=args.augment)
        self.val_ds = make_dataset(val_paths, self.val_grades, image_size, args.batch_size)
        test_paths, self.test_grades = list_images(args.base_path / "test")
        self.test_ds = make_dataset(test_paths, self.test_grades, image_size, args.batch_size) if test_paths else None
        print(f"{len(train_paths)} training, {len(val_paths)} validation and {len(test_paths)} test images")

        weights = None if args.weights.lower() == "none" else args.weights
        self.base, self.model = build_model(image_size, tuple(args.dense_units), args.droprate, weights)
        if self.last_weights.exists():
            load_weights(self.model, self.last_weights)

    def train(self) -> None:
        args = self.args
        if self.state["head_epochs_done"] < args.head_epochs:
            set_trainable_layers(self.base, 0)
            self.model.compile(keras.optimizers.Adam(args.lr_dense), loss="categorical_crossentropy")
            for epoch in range(self.state["head_epochs_done"], args.head_epochs):
                self._run_epoch("warm-up", epoch, args.lr_dense)

        if self.state["finetune_epochs_done"] < args.n_epochs:
            set_trainable_layers(self.base, args.unfreeze_after)
            self.model.compile(keras.optimizers.AdamW(args.lr_finetune, weight_decay=1e-4),
                               loss="categorical_crossentropy")
            for epoch in range(self.state["finetune_epochs_done"], args.n_epochs):
                if 0 < args.early_stopping_patience <= self.state["epochs_without_improvement"]:
                    print(f"No improvement for {args.early_stopping_patience} epochs, stopping")
                    break
                self._run_epoch("fine-tune", epoch, finetune_learning_rate(args.lr_finetune, epoch, args.n_epochs))

    def _run_epoch(self, stage: str, epoch: int, learning_rate: float) -> None:
        started = time.time()
        self.model.optimizer.learning_rate.assign(learning_rate)
        history = self.model.fit(self.train_ds, epochs=1, verbose=0, shuffle=False)
        metrics = grade_metrics(predict_probabilities(self.model, self.val_ds), self.val_grades)

        # everything a restart needs is on disk before the epoch counts as done
        self.model.save_weights(self.last_weights)
        best = self.state["best_mean_abs_diff"]
        fine_tuning = stage == "fine-tune"
        if best is None or metrics["mean_abs_diff"] < best:
            self.state["best_mean_abs_diff"] = metrics["mean_abs_diff"]
            self.state["epochs_without_improvement"] = 0
            self.model.save_weights(self.best_weights)
        elif fine_tuning:
            self.state["epochs_without_improvement"] += 1
        self.state["finetune_epochs_done" if fine_tuning else "head_epochs_done"] = epoch + 1

        row = {"stage": stage, "epoch": epoch + 1, "learning_rate": learning_rate,
               "loss": float(history.history["loss"][0]), "val_exact": metrics["exact"],
               "val_within_1": metrics["within_1"], "val_mean_abs_diff": metrics["mean_abs_diff"],
               "seconds": round(time.time() - started)}
        is_new = not self.log_file.exists()
        with open(self.log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        self.state_file.write_text(json.dumps(self.state))
        print(f"{stage} epoch {epoch + 1}: loss {row['loss']:.3f} | validation: exact {metrics['exact']:.3f}, "
              f"within 1 grade {metrics['within_1']:.3f}, mean difference {metrics['mean_abs_diff']:.2f} "
              f"| {row['seconds']}s", flush=True)

    def evaluate(self) -> dict:
        """Restore the best epoch, save it as a model and evaluate it on the validation and test images."""
        if self.best_weights.exists():
            load_weights(self.model, self.best_weights)
        self.model.save(self.run_dir / "best_model.keras")
        results = {"validation": grade_metrics(predict_probabilities(self.model, self.val_ds, flip=True),
                                               self.val_grades)}
        if self.test_ds is not None:
            results["test"] = grade_metrics(predict_probabilities(self.model, self.test_ds, flip=True),
                                            self.test_grades)
        (self.run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
        return results


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    if not args.base_path.is_dir():
        raise SystemExit(f"base_path does not exist: {args.base_path}")
    if args.run_dir is not None and not args.run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist: {args.run_dir}")
    keras.utils.set_random_seed(args.seed)

    run_dir = args.run_dir or create_run_dir(args.log_root)
    params = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (run_dir / "params.json").write_text(json.dumps(params, indent=2))

    split_into_train_test(args.base_path, args.seed)
    run = TrainingRun(args, run_dir)
    run.train()
    results = run.evaluate()

    for name, metrics in results.items():
        print(f"\n{name.capitalize()} set ({metrics['n_images']} images, best epoch):")
        print(f"  Exact grade:            {metrics['exact']:.3f}")
        print(f"  Within 1 grade:         {metrics['within_1']:.3f}")
        print(f"  Within 2 grades:        {metrics['within_2']:.3f}")
        print(f"  Grade Differential (mean):   {metrics['mean_abs_diff']:.2f}")
        print(f"  Grade Differential (median): {metrics['median_abs_diff']:.1f}")
    print(f"\nLogs, metrics and the saved model: {run_dir}\n")


if __name__ == "__main__":
    main()
