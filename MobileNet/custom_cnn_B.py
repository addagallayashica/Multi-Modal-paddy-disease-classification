# =============================================================================
# Custom CNN + Crop Metadata (Variety + Plant Age) for Paddy Disease
# Configuration 2: Image + Crop Metadata
# =============================================================================

import os
import glob
import json
import time
import random
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Activation, Dropout,
    Dense, BatchNormalization, GlobalAveragePooling2D,
    Concatenate, Rescaling, RandomFlip, RandomRotation,
    RandomTranslation, RandomZoom, StringLookup
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras import backend as K

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "result_custom_cnn_config2")

os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 123
MAX_EPOCHS = 30
INIT_LR = 1e-3
# Lower defaults to avoid GPU OOM on typical workstation cards while training.
BS = 8
WIDTH = 128
HEIGHT = 128
DEPTH = 3
VALIDATION_SPLIT = 0.20
TEST_SPLIT = 0.15
PATIENCE = 15
MIN_DELTA = 1e-4

# Keep image branch identical in spirit to Config 1.
IMAGE_PROJECTOR_DIM = 256

# Keep metadata deliberately small so it supplements the image.
VARIETY_PROJECTOR_DIM = 16
AGE_PROJECTOR_DIM = 8
METADATA_PROJECTOR_DIM = 32

# Output files
MODEL_SAVE_PATH = os.path.join(
    RESULTS_DIR, "paddy_disease_custom_cnn_config2_best.keras"
)
FINAL_MODEL_PATH = os.path.join(
    RESULTS_DIR, "paddy_disease_custom_cnn_config2_model.keras"
)

CLASS_FOLDERS = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "black_stem_borer",
    "blast",
    "brown_spot",
    "downy_mildew",
    "hispa",
    "leaf_roller",
    "normal",
    "tungro",
    "white_stem_borer",
    "yellow_stem_borer",
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_reproducibility():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    print(f"[INFO] TensorFlow: {tf.__version__}")
    print(f"[INFO] GPU devices: {gpus}")


# =============================================================================
# STEP 1: VERIFY / CREATE TRAIN-TEST SPLIT
# =============================================================================

def split_dataset():
    split_is_ready = True

    for cls in CLASS_FOLDERS:
        source = os.path.join(BASE_DIR, cls)
        train = os.path.join(TRAIN_DIR, cls)
        test = os.path.join(TEST_DIR, cls)

        if not (
            os.path.isdir(source)
            and os.path.isdir(train)
            and os.path.isdir(test)
        ):
            split_is_ready = False
            break

        source_files = {
            f.lower() for f in os.listdir(source)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        }
        train_files = {
            f.lower() for f in os.listdir(train)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        }
        test_files = {
            f.lower() for f in os.listdir(test)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        }

        if not source_files or train_files | test_files != source_files:
            split_is_ready = False
            break

    if split_is_ready:
        print("[INFO] Complete train/test split verified. Skipping split.")
        return

    print("[INFO] Creating deterministic 85/15 class-stratified split...")

    rng = np.random.default_rng(SEED)

    for cls in CLASS_FOLDERS:
        source = os.path.join(BASE_DIR, cls)
        if not os.path.isdir(source):
            raise FileNotFoundError(f"Missing class folder: {source}")

        images = sorted(
            f for f in os.listdir(source)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        )

        rng.shuffle(images)
        split_idx = int(len(images) * (1 - TEST_SPLIT))

        train_images = images[:split_idx]
        test_images = images[split_idx:]

        train_cls = os.path.join(TRAIN_DIR, cls)
        test_cls = os.path.join(TEST_DIR, cls)

        os.makedirs(train_cls, exist_ok=True)
        os.makedirs(test_cls, exist_ok=True)

        for img in train_images:
            dst = os.path.join(train_cls, img)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(source, img), dst)

        for img in test_images:
            dst = os.path.join(test_cls, img)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(source, img), dst)

        print(f"{cls}: {len(train_images)} train, {len(test_images)} test")


# =============================================================================
# STEP 2: LOAD AND VALIDATE METADATA
# =============================================================================

def load_metadata():
    if not os.path.isfile(METADATA_PATH):
        raise FileNotFoundError(
            f"metadata.csv not found at: {METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)

    required = {"image_id", "label", "variety", "age"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {sorted(missing)}")

    metadata["image_id"] = (
        metadata["image_id"].astype(str).str.strip().str.lower()
    )
    metadata["label"] = metadata["label"].astype(str).str.strip()
    metadata["variety"] = (
        metadata["variety"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )
    metadata["age"] = pd.to_numeric(metadata["age"], errors="coerce")

    if metadata["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values found in metadata.csv.")

    print(f"[INFO] Metadata rows: {len(metadata)}")
    print(f"[INFO] Missing variety: {metadata['variety'].eq('unknown').sum()}")
    print(f"[INFO] Missing age: {metadata['age'].isna().sum()}")

    return metadata.set_index("image_id")


# =============================================================================
# STEP 3: COLLECT IMAGE RECORDS
# =============================================================================

def collect_records(directory, metadata):
    records = []

    for class_index, class_name in enumerate(CLASS_FOLDERS):
        class_dir = os.path.join(directory, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing directory: {class_dir}")

        for name in sorted(os.listdir(class_dir)):
            if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            key = name.lower()

            if key not in metadata.index:
                raise ValueError(f"No metadata row for image: {name}")

            row = metadata.loc[key]

            if row["label"] != class_name:
                raise ValueError(
                    f"Metadata label mismatch for {name}: "
                    f"{row['label']} != {class_name}"
                )

            records.append({
                "path": os.path.join(class_dir, name),
                "image_id": name,
                "label": class_index,
                "class_name": class_name,
                "variety": str(row["variety"]),
                "age": row["age"],
            })

    if not records:
        raise ValueError(f"No images found under {directory}")

    return pd.DataFrame(records)


# =============================================================================
# STEP 4: STRATIFIED TRAIN / VALIDATION SPLIT
# =============================================================================

def make_train_validation_split(train_records):
    train_df, valid_df = train_test_split(
        train_records,
        test_size=VALIDATION_SPLIT,
        stratify=train_records["label"],
        random_state=SEED,
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
    )


# =============================================================================
# STEP 5: PREPARE METADATA
# =============================================================================

def prepare_metadata(train_df, valid_df, test_df):
    # Median imputation learned ONLY from training data.
    age_median = float(train_df["age"].median())

    if np.isnan(age_median):
        raise ValueError("Training age values are completely missing.")

    for df in (train_df, valid_df, test_df):
        df["age"] = df["age"].fillna(age_median)

    # Normalization statistics learned ONLY from training data.
    age_mean = float(train_df["age"].mean())
    age_std = float(train_df["age"].std())

    if age_std == 0 or np.isnan(age_std):
        age_std = 1.0

    for df in (train_df, valid_df, test_df):
        df["age_normalized"] = (
            (df["age"] - age_mean) / age_std
        ).astype("float32")

    # Only varieties observed in training are used to build the lookup.
    variety_values = sorted(train_df["variety"].unique().tolist())

    print(f"[INFO] Variety vocabulary ({len(variety_values)}):")
    print(variety_values)
    print(f"[INFO] Training age mean: {age_mean:.4f}")
    print(f"[INFO] Training age std: {age_std:.4f}")
    print(f"[INFO] Training age median: {age_median:.4f}")

    return train_df, valid_df, test_df, variety_values, age_mean, age_std, age_median


# =============================================================================
# STEP 6: TF.DATA DATASET
# =============================================================================

def make_dataset(df, training):
    paths = df["path"].astype(str).values
    varieties = df["variety"].astype(str).values
    ages = df["age_normalized"].astype("float32").values
    labels = df["label"].astype("int32").values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, varieties, ages, labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def load_example(path, variety, age, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(
            image,
            channels=DEPTH,
            expand_animations=False,
        )
        image.set_shape((None, None, DEPTH))
        image = tf.image.resize_with_pad(
            image,
            HEIGHT,
            WIDTH,
        )
        image = tf.cast(image, tf.float32)

        inputs = {
            "image": image,
            "variety": variety,
            "age": tf.reshape(age, [1]),
        }

        return inputs, label

    dataset = dataset.map(
        load_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.batch(BS).prefetch(tf.data.AUTOTUNE)


# =============================================================================
# STEP 7: BUILD CUSTOM CNN + METADATA MODEL
# =============================================================================

def get_model(variety_values):
    # -------------------------
    # IMAGE INPUT
    # -------------------------
    image_input = Input(
        shape=(HEIGHT, WIDTH, DEPTH),
        name="image",
    )

    # Same basic visual augmentation/preprocessing as Config 1.
    image = RandomFlip(
        "horizontal",
        name="random_flip",
    )(image_input)

    image = RandomRotation(
        5 / 360,
        fill_mode="nearest",
        name="random_rotation",
    )(image)

    image = RandomTranslation(
        0.05,
        0.05,
        fill_mode="nearest",
        name="random_translation",
    )(image)

    image = RandomZoom(
        (-0.2, 0.2),
        fill_mode="nearest",
        name="random_zoom",
    )(image)

    image = Rescaling(
        1.0 / 255.0,
        name="rescaling",
    )(image)

    # -------------------------
    # IMAGE CNN
    # -------------------------
    # Matches the Config 1 CNN structure:
    # 32 -> 64 -> 128 -> GAP -> Dense(256)
    for filters in (32, 64, 128):

        image = Conv2D(
            filters,
            (3, 3),
            padding="same",
            name=f"conv_{filters}_a",
        )(image)

        image = Activation(
            "relu",
            name=f"relu_{filters}_a",
        )(image)

        image = BatchNormalization(
            name=f"bn_{filters}_a",
        )(image)

        if filters > 32:
            image = Conv2D(
                filters,
                (3, 3),
                padding="same",
                name=f"conv_{filters}_b",
            )(image)

            image = Activation(
                "relu",
                name=f"relu_{filters}_b",
            )(image)

            image = BatchNormalization(
                name=f"bn_{filters}_b",
            )(image)

        if filters == 32:
            image = MaxPooling2D(
                pool_size=(3, 3),
                name="pool_1",
            )(image)
        else:
            image = MaxPooling2D(
                pool_size=(2, 2),
                name=f"pool_{filters}",
            )(image)

        image = Dropout(
            0.25,
            name=f"dropout_{filters}",
        )(image)

    image = GlobalAveragePooling2D(
        name="global_average_pooling",
    )(image)

    image = Dense(
        IMAGE_PROJECTOR_DIM,
        name="image_projector",
    )(image)

    image = Activation(
        "relu",
        name="image_projector_relu",
    )(image)

    image = BatchNormalization(
        name="image_projector_bn",
    )(image)

    image = Dropout(
        0.50,
        name="image_projector_dropout",
    )(image)

    # -------------------------
    # VARIETY INPUT
    # -------------------------
    variety_input = Input(
        shape=(),
        dtype=tf.string,
        name="variety",
    )

    variety_lookup = StringLookup(
        vocabulary=variety_values,
        mask_token=None,
        num_oov_indices=1,
        output_mode="one_hot",
        name="variety_lookup",
    )

    variety = variety_lookup(variety_input)

    # Small branch: variety is auxiliary information.
    variety = Dense(
        VARIETY_PROJECTOR_DIM,
        activation="relu",
        name="variety_projector",
    )(variety)

    # -------------------------
    # AGE INPUT
    # -------------------------
    age_input = Input(
        shape=(1,),
        dtype=tf.float32,
        name="age",
    )

    # Age is already standardized using training-set statistics.
    age = Dense(
        AGE_PROJECTOR_DIM,
        activation="relu",
        name="age_projector",
    )(age_input)

    # -------------------------
    # METADATA FUSION
    # -------------------------
    metadata = Concatenate(
        name="metadata_fusion",
    )([variety, age])

    metadata = Dense(
        METADATA_PROJECTOR_DIM,
        activation="relu",
        name="metadata_projector",
    )(metadata)

    metadata = BatchNormalization(
        name="metadata_bn",
    )(metadata)

    metadata = Dropout(
        0.20,
        name="metadata_dropout",
    )(metadata)

    # -------------------------
    # IMAGE + METADATA FUSION
    # -------------------------
    fused = Concatenate(
        name="feature_fusion",
    )([image, metadata])

    # Compact fusion head to avoid metadata-driven overfitting.
    fused = Dense(
        128,
        activation="relu",
        name="fusion_dense",
    )(fused)

    fused = BatchNormalization(
        name="fusion_bn",
    )(fused)

    fused = Dropout(
        0.25,
        name="fusion_dropout",
    )(fused)

    outputs = Dense(
        len(CLASS_FOLDERS),
        activation="softmax",
        name="classifier",
    )(fused)

    model = Model(
        inputs=[image_input, variety_input, age_input],
        outputs=outputs,
        name="custom_cnn_config2_image_crop",
    )

    model.compile(
        optimizer=Adam(learning_rate=INIT_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =============================================================================
# STEP 8: METRICS CALLBACK
# =============================================================================

class MacroF1Callback(tf.keras.callbacks.Callback):

    def __init__(self, validation_dataset):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.val_macro_f1_history = []

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []

        for inputs, labels in self.validation_dataset:
            probabilities = self.model(
                inputs,
                training=False,
            )
            predictions = tf.argmax(
                probabilities,
                axis=1,
            )

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.numpy())

        score = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        self.val_macro_f1_history.append(float(score))

        if logs is not None:
            logs["val_macro_f1"] = float(score)

        print(f" - val_macro_f1: {score:.4f}")


# =============================================================================
# STEP 9: PREDICTIONS
# =============================================================================

def collect_predictions(model, dataset):
    labels = []
    probabilities = []

    for inputs, batch_labels in dataset:
        batch_prob = model(
            inputs,
            training=False,
        )

        labels.extend(batch_labels.numpy())
        probabilities.extend(batch_prob.numpy())

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return labels, predictions, probabilities


# =============================================================================
# STEP 10: PLOT TRAINING HISTORY
# =============================================================================

def save_training_plot(history):
    epochs = range(
        1,
        len(history.history["loss"]) + 1,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    axes[0].plot(
        epochs,
        history.history["accuracy"],
        label="Training Accuracy",
    )
    axes[0].plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )
    axes[0].set_title(
        "Custom CNN Config 2 - Accuracy"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs,
        history.history["loss"],
        label="Training Loss",
    )
    axes[1].plot(
        epochs,
        history.history["val_loss"],
        label="Validation Loss",
    )
    axes[1].set_title(
        "Custom CNN Config 2 - Loss"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "training_history.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )
    plt.close()

    print(f"[INFO] Training plot saved: {path}")


# =============================================================================
# STEP 11: MAIN
# =============================================================================

def main():

    set_reproducibility()

    print("\n" + "=" * 70)
    print("CUSTOM CNN - CONFIGURATION 2: IMAGE + CROP")
    print("=" * 70)

    # Dataset
    split_dataset()

    metadata = load_metadata()

    train_records_all = collect_records(
        TRAIN_DIR,
        metadata,
    )

    test_records = collect_records(
        TEST_DIR,
        metadata,
    )

    # Check train/test filename leakage.
    train_ids = set(
        train_records_all["image_id"].str.lower()
    )
    test_ids = set(
        test_records["image_id"].str.lower()
    )

    overlap = train_ids.intersection(test_ids)

    if overlap:
        raise ValueError(
            f"Train/test leakage detected: {list(overlap)[:5]}"
        )

    # Stratified validation split.
    train_df, valid_df = make_train_validation_split(
        train_records_all
    )

    # Prepare crop metadata.
    (
        train_df,
        valid_df,
        test_records,
        variety_values,
        age_mean,
        age_std,
        age_median,
    ) = prepare_metadata(
        train_df,
        valid_df,
        test_records,
    )

    print("\n[INFO] Dataset:")
    print(f"  Training:   {len(train_df)}")
    print(f"  Validation: {len(valid_df)}")
    print(f"  Test:       {len(test_records)}")

    # Datasets.
    train_dataset = make_dataset(
        train_df,
        training=True,
    )

    validation_dataset = make_dataset(
        valid_df,
        training=False,
    )

    test_dataset = make_dataset(
        test_records,
        training=False,
    )

    # Model.
    print("\n[INFO] Building Config 2 model...")

    model = get_model(
        variety_values
    )

    model.summary()

    print(
        f"\n[INFO] Total parameters: "
        f"{model.count_params():,}"
    )

    # Class weights from training data ONLY.
    class_counts = np.bincount(
        train_df["label"].values,
        minlength=len(CLASS_FOLDERS),
    )

    class_weight = {
        index: len(train_df) /
        (
            len(CLASS_FOLDERS) *
            max(int(count), 1)
        )
        for index, count in enumerate(class_counts)
    }

    print("\n[INFO] Class weights:")
    for index, weight in class_weight.items():
        print(
            f"  {index}: "
            f"{CLASS_FOLDERS[index]} = {weight:.4f}"
        )

    # Callbacks.
    macro_f1_callback = MacroF1Callback(
        validation_dataset
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    callbacks = [
        macro_f1_callback,
        early_stop,
        checkpoint,
        reduce_lr,
    ]

    # Train.
    print("\n" + "=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    start_time = time.time()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    elapsed = time.time() - start_time
    actual_epochs = len(
        history.history["loss"]
    )

    print(
        f"\n[INFO] Training completed in "
        f"{elapsed / 60:.1f} minutes"
    )

    # Best epoch based on validation loss,
    # matching ModelCheckpoint and EarlyStopping.
    best_epoch = int(
        np.argmin(
            history.history["val_loss"]
        ) + 1
    )

    best_val_loss = float(
        history.history["val_loss"][best_epoch - 1]
    )

    best_val_acc = float(
        history.history["val_accuracy"][best_epoch - 1]
    )

    best_val_macro_f1 = float(
        macro_f1_callback.val_macro_f1_history[
            best_epoch - 1
        ]
    )

    print(
        f"[INFO] Best epoch: {best_epoch}"
    )
    print(
        f"[INFO] Best val accuracy: "
        f"{best_val_acc:.4f}"
    )
    print(
        f"[INFO] Best val macro-F1: "
        f"{best_val_macro_f1:.4f}"
    )
    print(
        f"[INFO] Best val loss: "
        f"{best_val_loss:.4f}"
    )

    # Plot.
    save_training_plot(history)

    # Evaluate test.
    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    test_loss = float(
        model.evaluate(
            test_dataset,
            verbose=1,
        )[0]
    )

    true_labels, predicted_labels, probabilities = (
        collect_predictions(
            model,
            test_dataset,
        )
    )

    test_accuracy = float(
        np.mean(
            true_labels == predicted_labels
        )
    )

    print(
        f"\n[RESULT] Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"[RESULT] Test Loss: "
        f"{test_loss:.4f}"
    )

    # Classification report.
    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=range(len(CLASS_FOLDERS)),
        target_names=CLASS_FOLDERS,
        zero_division=0,
    )

    print("\n--- Classification Report ---")
    print(report_text)

    report_data = classification_report(
        true_labels,
        predicted_labels,
        labels=range(len(CLASS_FOLDERS)),
        target_names=CLASS_FOLDERS,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix.
    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(len(CLASS_FOLDERS)),
    )

    # Normalized confusion matrix.
    normalized_cm = cm.astype(np.float64)

    row_totals = normalized_cm.sum(
        axis=1,
        keepdims=True,
    )

    np.divide(
        normalized_cm,
        row_totals,
        out=normalized_cm,
        where=row_totals != 0,
    )

    # Additional metrics.
    balanced_acc = float(
        balanced_accuracy_score(
            true_labels,
            predicted_labels,
        )
    )

    macro_precision = float(
        precision_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )

    macro_recall = float(
        recall_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )

    macro_f1 = float(
        f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision = float(
        precision_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )

    weighted_recall = float(
        recall_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )

    weighted_f1 = float(
        f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )

    mcc = float(
        matthews_corrcoef(
            true_labels,
            predicted_labels,
        )
    )

    kappa = float(
        cohen_kappa_score(
            true_labels,
            predicted_labels,
        )
    )

    # Save classification report.
    classification_report_path = os.path.join(
        RESULTS_DIR,
        "classification_report.csv",
    )

    pd.DataFrame(
        report_data
    ).transpose().to_csv(
        classification_report_path
    )

    # Save confusion matrix.
    confusion_matrix_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.csv",
    )

    pd.DataFrame(
        cm,
        index=CLASS_FOLDERS,
        columns=CLASS_FOLDERS,
    ).to_csv(
        confusion_matrix_path
    )

    # Save normalized confusion matrix.
    normalized_cm_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix_normalized.csv",
    )

    pd.DataFrame(
        normalized_cm,
        index=CLASS_FOLDERS,
        columns=CLASS_FOLDERS,
    ).to_csv(
        normalized_cm_path
    )

    # Save overall metrics.
    overall_metrics = {
        "model": "Custom CNN - Config 2 Image + Crop",
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_val_acc,
        "best_validation_macro_f1": best_val_macro_f1,
        "actual_epochs": actual_epochs,
        "training_samples": len(train_df),
        "validation_samples": len(valid_df),
        "test_samples": len(test_records),
        "batch_size": BS,
        "image_size": f"{WIDTH}x{HEIGHT}",
        "learning_rate": INIT_LR,
        "optimizer": "Adam",
        "metadata": "variety + age",
        "variety_count": len(variety_values),
        "variety_projector_dim": VARIETY_PROJECTOR_DIM,
        "age_projector_dim": AGE_PROJECTOR_DIM,
        "metadata_projector_dim": METADATA_PROJECTOR_DIM,
        "total_parameters": model.count_params(),
        "training_time_seconds": round(elapsed, 1),
    }

    overall_metrics_path = os.path.join(
        RESULTS_DIR,
        "overall_metrics.csv",
    )

    pd.DataFrame(
        [overall_metrics]
    ).to_csv(
        overall_metrics_path,
        index=False,
    )

    # Save training history.
    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_df) + 1,
        ),
    )

    history_df["val_macro_f1"] = (
        macro_f1_callback.val_macro_f1_history
    )

    history_path = os.path.join(
        RESULTS_DIR,
        "training_metrics.csv",
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # Save submission.
    submission = pd.DataFrame({
        "image_id": test_records["image_id"].values,
        "true_label": [
            CLASS_FOLDERS[i]
            for i in true_labels
        ],
        "predicted_label": [
            CLASS_FOLDERS[i]
            for i in predicted_labels
        ],
    })

    submission_path = os.path.join(
        RESULTS_DIR,
        "submission.csv",
    )

    submission.to_csv(
        submission_path,
        index=False,
    )

    # Save final model.
    model.save(
        FINAL_MODEL_PATH
    )

    # Save complete JSON summary.
    summary = {
        "status": "completed",
        "configuration": "Config 2 - Image + Crop",
        "random_seed": SEED,
        "dataset": {
            "class_count": len(CLASS_FOLDERS),
            "class_names": CLASS_FOLDERS,
            "training_samples": len(train_df),
            "validation_samples": len(valid_df),
            "test_samples": len(test_records),
        },
        "preprocessing": {
            "image_size": [WIDTH, HEIGHT],
            "channels": DEPTH,
            "rescale": "1/255",
            "validation_split": VALIDATION_SPLIT,
            "test_split": TEST_SPLIT,
        },
        "crop_metadata": {
            "columns": ["variety", "age"],
            "variety_vocabulary": variety_values,
            "age_imputation": "training_median",
            "age_training_mean": age_mean,
            "age_training_std": age_std,
            "age_training_median": age_median,
        },
        "training": {
            "batch_size": BS,
            "max_epochs": MAX_EPOCHS,
            "actual_epochs": actual_epochs,
            "best_epoch": best_epoch,
            "learning_rate": INIT_LR,
            "optimizer": "Adam",
            "loss": "sparse_categorical_crossentropy",
            "early_stopping_monitor": "val_loss",
            "checkpoint_monitor": "val_loss",
        },
        "model": {
            "name": "Custom CNN Config 2 Image + Crop",
            "total_parameters": int(
                model.count_params()
            ),
            "image_projector_dimension": IMAGE_PROJECTOR_DIM,
            "variety_projector_dimension": VARIETY_PROJECTOR_DIM,
            "age_projector_dimension": AGE_PROJECTOR_DIM,
            "metadata_projector_dimension": METADATA_PROJECTOR_DIM,
            "fusion_dimension": 128,
        },
        "metrics": overall_metrics,
        "artifacts": {
            "best_model": MODEL_SAVE_PATH,
            "final_model": FINAL_MODEL_PATH,
            "training_history": history_path,
            "classification_report": classification_report_path,
            "confusion_matrix": confusion_matrix_path,
            "normalized_confusion_matrix": normalized_cm_path,
            "overall_metrics": overall_metrics_path,
            "submission": submission_path,
        },
    }

    summary_path = os.path.join(
        RESULTS_DIR,
        "post_training_metrics.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Test Accuracy       : {test_accuracy * 100:.2f}%")
    print(f"Macro Precision     : {macro_precision * 100:.2f}%")
    print(f"Macro Recall        : {macro_recall * 100:.2f}%")
    print(f"Macro F1            : {macro_f1 * 100:.2f}%")
    print(f"Weighted F1         : {weighted_f1 * 100:.2f}%")
    print(f"Balanced Accuracy   : {balanced_acc * 100:.2f}%")
    print(f"MCC                 : {mcc:.4f}")
    print(f"Cohen Kappa         : {kappa:.4f}")
    print(f"Best Epoch          : {best_epoch}")
    print(f"Parameters          : {model.count_params():,}")
    print("=" * 70)
    print(f"[INFO] Results directory: {RESULTS_DIR}")
    print("[INFO] ALL DONE!")


if __name__ == "__main__":
    main()