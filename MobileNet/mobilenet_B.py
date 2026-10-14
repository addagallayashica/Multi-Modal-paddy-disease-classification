"""MobileNetV2 image + metadata fusion training for paddy disease classification."""

import json
import os
import random
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

# Keep every input, checkpoint, and result path relative to this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "result_mobilenet_B")
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "paddy_disease_mobilenet_B_best.weights.h5")
FINAL_MODEL_PATH = os.path.join(RESULTS_DIR, "paddy_disease_mobilenet_B_model.keras")

# Reproducibility and training schedule.
SEED = 123
MAX_EPOCHS = 75                 # Total upper limit across both stages.
HEAD_EPOCHS = 15                # Stage 1: train the new fusion/classifier head.
INIT_LR = 1e-3                  # Stage 1 learning rate with MobileNetV2 frozen.
FINE_TUNE_LR = 1e-5             # Stage 2 learning rate to protect ImageNet weights.
BS = 8                          # Suited to the RTX 2050's available VRAM.

# Image and split parameters.
WIDTH = 224                     # Native MobileNetV2 ImageNet spatial size.
HEIGHT = 224
DEPTH = 3                       # RGB input.
VALIDATION_SPLIT = 0.2          # Fraction of train/ reserved for validation.
TEST_SPLIT = 0.15               # Fraction of source data reserved for test/.

# Regularization and stopping parameters.
PATIENCE = 10                   # Epochs without macro-F1 improvement.
MIN_DELTA = 1e-4
LABEL_SMOOTHING = 0.1
FUSION_DROPOUT = 0.35

# Multimodal fusion and metadata parameters.
PROJECTOR_DIM = 64              # Equal image/metadata width before concatenation.
FINE_TUNE_LAYERS = 24           # Top MobileNetV2 layers eligible for fine-tuning.
MIN_VARIETY_COUNT = 20          # Rare training varieties become the "other" token.
REQUIRE_GPU = True              # Prevent an accidental, very slow CPU training run.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

CLASS_FOLDERS = [
    "bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight",
    "black_stem_borer", "blast", "brown_spot", "downy_mildew", "hispa",
    "leaf_roller", "normal", "tungro", "white_stem_borer", "yellow_stem_borer",
]
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_reproducibility():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_ENABLE_XLA"] = "0"
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0"
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.optimizer.set_jit(False)
    gpus = tf.config.list_physical_devices("GPU")
    if REQUIRE_GPU and not gpus:
        raise RuntimeError("No TensorFlow GPU detected.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[INFO] Using GPU devices: {gpus}")


def image_files(directory):
    return {name.lower() for name in os.listdir(directory) if Path(name).suffix.lower() in IMAGE_EXTENSIONS}


def split_dataset():
    """Verify or create the deterministic class-stratified 85/15 split."""
    split_is_ready = True
    for class_name in CLASS_FOLDERS:
        source = os.path.join(BASE_DIR, class_name)
        train = os.path.join(TRAIN_DIR, class_name)
        test = os.path.join(TEST_DIR, class_name)
        if not all(os.path.isdir(path) for path in (source, train, test)):
            split_is_ready = False
            break
        if not image_files(source) or image_files(train) | image_files(test) != image_files(source):
            split_is_ready = False
            break
    if split_is_ready:
        print("[INFO] Complete train/test split verified. Skipping split.")
        return

    rng = np.random.default_rng(SEED)
    for class_name in CLASS_FOLDERS:
        source = os.path.join(BASE_DIR, class_name)
        if not os.path.isdir(source):
            raise FileNotFoundError(f"Missing source class directory: {source}")
        names = sorted(name for name in os.listdir(source) if Path(name).suffix.lower() in IMAGE_EXTENSIONS)
        if len(names) < 2:
            raise ValueError(f"Class {class_name!r} needs at least two images.")
        rng.shuffle(names)
        cut = int(len(names) * (1 - TEST_SPLIT))
        for destination_dir, selected in ((os.path.join(TRAIN_DIR, class_name), names[:cut]),
                                          (os.path.join(TEST_DIR, class_name), names[cut:])):
            os.makedirs(destination_dir, exist_ok=True)
            for name in selected:
                destination = os.path.join(destination_dir, name)
                if not os.path.exists(destination):
                    shutil.copy2(os.path.join(source, name), destination)
    print("[INFO] Dataset split created.")


def load_metadata():
    """Load metadata without using held-out values to fit preprocessing rules."""
    if not os.path.isfile(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
    metadata = pd.read_csv(METADATA_PATH, dtype={"image_id": str, "label": str, "variety": str})
    required = {"image_id", "label", "variety", "age"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")
    metadata["image_id"] = metadata["image_id"].str.strip().str.lower()
    metadata["label"] = metadata["label"].str.strip().str.lower()
    metadata["variety"] = metadata["variety"].fillna("").str.strip().str.lower().replace("", "unknown")
    metadata["age"] = pd.to_numeric(metadata["age"], errors="coerce")
    if metadata["image_id"].duplicated().any():
        raise ValueError("metadata.csv contains duplicate image_id values.")
    print(f"[INFO] Metadata rows={len(metadata)}, missing ages={int(metadata['age'].isna().sum())}")
    return metadata.set_index("image_id")


def collect_records(directory, metadata):
    records = []
    for class_index, class_name in enumerate(CLASS_FOLDERS):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing split class directory: {class_dir}")
        for name in sorted(os.listdir(class_dir)):
            if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = name.lower()
            if key not in metadata.index:
                raise ValueError(f"No metadata row found for image: {name}")
            row = metadata.loc[key]
            if row["label"] != class_name:
                raise ValueError(f"Metadata label mismatch for {name}: {row['label']} != {class_name}")
            records.append((os.path.join(class_dir, name), row["variety"], float(row["age"]) if pd.notna(row["age"]) else np.nan, class_index, name))
    if not records:
        raise ValueError(f"No images found under {directory}")
    return records


def validate_alignment(metadata, train_records, test_records):
    """Reject missing rows, duplicates, label errors, and train/test leakage."""
    all_records = train_records + test_records
    keys = [record[4].lower() for record in all_records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate image filename found in train/test records.")
    if set(keys) != set(metadata.index):
        raise ValueError("Metadata rows and split images are not a one-to-one match.")
    train_keys = {record[4].lower() for record in train_records}
    test_keys = {record[4].lower() for record in test_records}
    if train_keys & test_keys:
        raise ValueError("Train/test leakage detected.")
    print(f"[INFO] Alignment verified: train={len(train_records)}, test={len(test_records)}")


def split_training_records(records):
    labels = np.asarray([record[3] for record in records])
    rng = np.random.default_rng(SEED)
    train_records, validation_records = [], []
    for class_index in range(len(CLASS_FOLDERS)):
        indices = np.flatnonzero(labels == class_index)
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(f"Class {CLASS_FOLDERS[class_index]!r} needs at least two training images.")
        validation_count = max(1, int(round(len(indices) * VALIDATION_SPLIT)))
        validation_records.extend(records[index] for index in indices[:validation_count])
        train_records.extend(records[index] for index in indices[validation_count:])
    return train_records, validation_records


def group_rare_varieties(train_records, other_records):
    """Fit the rare-category map on training data, then apply it to other splits."""
    counts = pd.Series([record[1] for record in train_records]).value_counts()
    rare = set(counts[counts < MIN_VARIETY_COUNT].index)

    def transform(records):
        return [record[:1] + ("other" if record[1] in rare else record[1],) + record[2:] for record in records]

    print(f"[INFO] Grouped {len(rare)} rare varieties using training records only.")
    return transform(train_records), transform(other_records), rare


def impute_age(records, median):
    return [record[:2] + (float(record[2]) if not np.isnan(record[2]) else median,) + record[3:] for record in records]


def make_dataset(records, age_mean, age_std, training):
    """Build a lazy paired image/metadata dataset with normalized age values."""
    paths = np.asarray([record[0] for record in records])
    varieties = np.asarray([record[1] for record in records]).astype(str)
    ages = ((np.asarray([record[2] for record in records], dtype="float32") - age_mean) / age_std).astype("float32")
    labels = np.asarray([record[3] for record in records], dtype="int32")
    dataset = tf.data.Dataset.from_tensor_slices((paths, varieties, ages, labels))
    if training:
        dataset = dataset.shuffle(len(records), seed=SEED, reshuffle_each_iteration=True)

    def load_example(path, variety, age, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=DEPTH, expand_animations=False)
        image.set_shape((None, None, DEPTH))
        image = tf.image.resize_with_pad(image, HEIGHT, WIDTH)
        return {"image": tf.cast(image, tf.float32), "variety": variety, "age": tf.reshape(age, [1])}, label

    return dataset.map(load_example, num_parallel_calls=tf.data.AUTOTUNE).batch(BS).prefetch(tf.data.AUTOTUNE)


def create_model(variety_values, fine_tune=0):
    """Build MobileNetV2 plus equal-width late fusion of image and metadata."""
    image_input = tf.keras.Input((HEIGHT, WIDTH, DEPTH), name="image")
    variety_input = tf.keras.Input((), dtype=tf.string, name="variety")
    age_input = tf.keras.Input((1,), dtype=tf.float32, name="age")
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(HEIGHT, WIDTH, DEPTH)
    )
    backbone._name = "mobilenetv2_backbone"
    for layer in backbone.layers:
        layer.trainable = False
    if fine_tune:
        for layer in backbone.layers[-fine_tune:]:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

    image = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(5 / 360, fill_mode="nearest"),
        tf.keras.layers.RandomZoom((-0.2, 0.2), fill_mode="nearest"),
    ], name="image_augmentation")(image_input)
    # MobileNetV2 expects pixels in [-1, 1], unlike the custom CNN's [0, 1].
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = backbone(image)
    image = tf.keras.layers.GlobalAveragePooling2D()(image)
    image = tf.keras.layers.Dense(PROJECTOR_DIM, activation="relu", name="image_projector")(image)
    image = tf.keras.layers.BatchNormalization()(image)
    image = tf.keras.layers.Dropout(0.25)(image)

    lookup = tf.keras.layers.StringLookup(vocabulary=variety_values, mask_token=None, num_oov_indices=1)
    variety = tf.keras.layers.Embedding(lookup.vocabulary_size(), 32)(lookup(variety_input))
    variety = tf.keras.layers.Flatten()(variety)
    variety = tf.keras.layers.Dense(64, activation="relu")(variety)
    metadata = tf.keras.layers.Concatenate()([variety, age_input])
    metadata = tf.keras.layers.Dense(PROJECTOR_DIM, activation="relu", name="metadata_projector")(metadata)
    metadata = tf.keras.layers.BatchNormalization()(metadata)
    metadata = tf.keras.layers.Dropout(0.2)(metadata)

    # Equal projectors prevent the 1,280-dimensional backbone output from
    # overwhelming the smaller metadata branch during optimization.
    fused = tf.keras.layers.Concatenate(name="feature_fusion")([image, metadata])
    fused = tf.keras.layers.Dropout(FUSION_DROPOUT)(fused)
    fused = tf.keras.layers.Dense(128, activation="relu")(fused)
    fused = tf.keras.layers.BatchNormalization()(fused)
    fused = tf.keras.layers.Dropout(0.3)(fused)
    outputs = tf.keras.layers.Dense(len(CLASS_FOLDERS), activation="softmax", name="classifier")(fused)
    return tf.keras.Model([image_input, variety_input, age_input], outputs, name="mobilenet_B_multimodal")


def compile_model(model, learning_rate):
    """Compile with clipped Adam and sparse labels plus label smoothing."""
    def smoothed_loss(labels, probabilities):
        labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        one_hot = tf.one_hot(labels, depth=len(CLASS_FOLDERS))
        return tf.keras.losses.categorical_crossentropy(one_hot, probabilities, label_smoothing=LABEL_SMOOTHING)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0), loss=smoothed_loss, metrics=["accuracy"], jit_compile=False)


class MacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, validation_dataset):
        super().__init__()
        self.validation_dataset = validation_dataset

    def on_epoch_end(self, epoch, logs=None):
        labels, predictions = [], []
        for inputs, batch_labels in self.validation_dataset:
            labels.extend(batch_labels.numpy())
            predictions.extend(tf.argmax(self.model(inputs, training=False), axis=1).numpy())
        score = f1_score(labels, predictions, average="macro", zero_division=0)
        logs = logs or {}
        logs["val_macro_f1"] = float(score)
        print(f" - val_macro_f1: {score:.4f}")


def class_weights(records):
    labels = np.asarray([record[3] for record in records])
    counts = np.bincount(labels, minlength=len(CLASS_FOLDERS))
    return {index: len(labels) / (len(CLASS_FOLDERS) * max(count, 1)) for index, count in enumerate(counts)}


def collect_predictions(model, dataset):
    labels, predictions = [], []
    for inputs, batch_labels in dataset:
        labels.extend(batch_labels.numpy())
        predictions.extend(tf.argmax(model(inputs, training=False), axis=1).numpy())
    return np.asarray(labels), np.asarray(predictions)


def save_history_plot(history):
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history.history["accuracy"], label="Training Accuracy")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
    axes[1].plot(epochs, history.history["loss"], label="Training Loss")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation Loss")
    for axis in axes:
        axis.legend(); axis.grid(alpha=0.3)
    figure.tight_layout(); figure.savefig(os.path.join(RESULTS_DIR, "training_history.png"), dpi=150); plt.close(figure)


def main():
    """Run integrity checks, two-stage training, evaluation, and artifact export."""
    set_reproducibility()
    split_dataset()
    metadata = load_metadata()
    train_all = collect_records(TRAIN_DIR, metadata)
    test_records = collect_records(TEST_DIR, metadata)
    validate_alignment(metadata, train_all, test_records)
    train_records, validation_records = split_training_records(train_all)
    train_records, validation_records, rare_varieties = group_rare_varieties(
        train_records, validation_records
    )
    _, test_records, _ = group_rare_varieties(train_records, test_records)
    valid_ages = np.asarray([record[2] for record in train_records if not np.isnan(record[2])], dtype="float32")
    if not valid_ages.size:
        raise ValueError("No valid training ages are available for imputation.")
    age_median = float(np.median(valid_ages))
    train_records = impute_age(train_records, age_median)
    validation_records = impute_age(validation_records, age_median)
    test_records = impute_age(test_records, age_median)
    train_ages = np.asarray([record[2] for record in train_records], dtype="float32")
    age_mean, age_std = float(train_ages.mean()), float(train_ages.std()) or 1.0
    variety_values = sorted({record[1] for record in train_records})
    train_dataset = make_dataset(train_records, age_mean, age_std, True)
    validation_dataset = make_dataset(validation_records, age_mean, age_std, False)
    test_dataset = make_dataset(test_records, age_mean, age_std, False)
    weights = class_weights(train_records)

    # Stage 1: learn the randomly initialized metadata and fusion head while
    # preserving the pretrained MobileNetV2 representation.
    model = create_model(variety_values, fine_tune=0)
    compile_model(model, INIT_LR)
    callbacks = [
        MacroF1Callback(validation_dataset),
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]
    start = time.time()
    head_history = model.fit(train_dataset, validation_data=validation_dataset, epochs=HEAD_EPOCHS, class_weight=weights, callbacks=callbacks)

    # Stage 2: unfreeze only the top convolutional layers. BatchNorm remains
    # frozen because small batches can otherwise destabilize its statistics.
    for layer in model.get_layer("mobilenetv2_backbone").layers[-FINE_TUNE_LAYERS:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    compile_model(model, FINE_TUNE_LR)
    fine_history = model.fit(train_dataset, validation_data=validation_dataset, epochs=MAX_EPOCHS, initial_epoch=len(head_history.history["loss"]), class_weight=weights, callbacks=callbacks)
    history_data = {key: head_history.history.get(key, []) + fine_history.history.get(key, []) for key in set(head_history.history) | set(fine_history.history)}
    history = tf.keras.callbacks.History()
    history.history = history_data
    model.load_weights(MODEL_SAVE_PATH)
    save_history_plot(history)

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    true_labels, predicted_labels = collect_predictions(model, test_dataset)
    report = classification_report(true_labels, predicted_labels, labels=range(len(CLASS_FOLDERS)), target_names=CLASS_FOLDERS, output_dict=True, zero_division=0)
    print(classification_report(true_labels, predicted_labels, target_names=CLASS_FOLDERS, zero_division=0))
    pd.DataFrame(report).transpose().to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"))
    pd.DataFrame(confusion_matrix(true_labels, predicted_labels, labels=range(len(CLASS_FOLDERS))), index=CLASS_FOLDERS, columns=CLASS_FOLDERS).to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"))
    pd.DataFrame({"image_id": [record[4] for record in test_records], "label": [CLASS_FOLDERS[index] for index in predicted_labels]}).to_csv(os.path.join(RESULTS_DIR, "submission.csv"), index=False)
    pd.DataFrame(history.history).assign(epoch=np.arange(1, len(history.history["loss"]) + 1)).to_csv(os.path.join(RESULTS_DIR, "training_metrics.csv"), index=False)
    model.save(FINAL_MODEL_PATH)
    summary = {
        "model": "MobileNetV2 with variety and age metadata fusion",
        "framework": f"TensorFlow {tf.__version__}",
        "actual_epochs": len(history.history["loss"]),
        "test_loss": float(test_loss), "test_accuracy": float(test_accuracy),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "class_count": len(CLASS_FOLDERS), "training_samples": len(train_records),
        "validation_samples": len(validation_records), "test_samples": len(test_records),
        "metadata_columns": ["variety", "age"], "age_training_median": age_median,
        "age_training_mean": age_mean, "age_training_std": age_std,
        "projector_dimension": PROJECTOR_DIM, "fine_tune_layers": FINE_TUNE_LAYERS,
        "head_learning_rate": INIT_LR, "fine_tune_learning_rate": FINE_TUNE_LR,
        "label_smoothing": LABEL_SMOOTHING, "batch_size": BS, "image_size": f"{WIDTH}x{HEIGHT}x{DEPTH}",
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "head_epochs": HEAD_EPOCHS,
        "validation_split": VALIDATION_SPLIT,
        "test_split": TEST_SPLIT,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "fusion_dropout": FUSION_DROPOUT,
        "min_variety_count": MIN_VARIETY_COUNT,
        "batch_norm_frozen_during_fine_tuning": True,
        "optimizer": "Adam with clipnorm=1.0, XLA disabled",
        "clipnorm": 1.0,
        "fine_tune_backbone_layers": FINE_TUNE_LAYERS,
        "require_gpu": REQUIRE_GPU,
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(np.prod(variable.shape) for variable in model.trainable_weights)),
    }
    with open(os.path.join(RESULTS_DIR, "post_training_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
