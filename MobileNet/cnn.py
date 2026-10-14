# =============================================================================
# CNN for Paddy Disease Classification
# Cleaned Python script (converted from Jupyter notebook)
# Dataset: 12 disease classes + 1 normal class (13 total)
# =============================================================================

import os
import glob
import json
import time
import numpy as np
import pandas as pd


























import cv2
from pathlib import Path

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Activation,
    Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')

SEED = 123
MAX_EPOCHS = 75         # Early stopping can halt training before this ceiling
INIT_LR = 1e-3
BS = 32
WIDTH = 256
HEIGHT = 256
DEPTH = 3
VALIDATION_SPLIT = 0.2  # 20% of training data used for validation
TEST_SPLIT = 0.15       # 15% of total data reserved for testing

# --- Early Stopping & Checkpoint Configuration ---
# PATIENCE: How many epochs to wait after val_loss stops improving.
#   Rule of thumb: ~10% of expected epochs. For ~16K images / 13 classes,
#   convergence typically happens around epoch 30-60.
PATIENCE = 15
MIN_DELTA = 1e-4        # Minimum improvement to count as "progress"
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'paddy_disease_cnn_best.keras')

# =============================================================================
# STEP 1: SPLIT DATASET INTO TRAIN AND TEST DIRECTORIES
# =============================================================================
# The original dataset has all images in class folders at BASE_DIR.
# We split them into train/ and test/ subdirectories.

CLASS_FOLDERS = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak',
    'bacterial_panicle_blight', 'black_stem_borer',
    'blast', 'brown_spot', 'downy_mildew', 'hispa',
    'leaf_roller', 'normal', 'tungro',
    'white_stem_borer', 'yellow_stem_borer'
]

def split_dataset():
    """Split original class folders into train/ and test/ directories."""
    import shutil
    np.random.seed(SEED)

    split_is_ready = all(
        os.path.isdir(os.path.join(TRAIN_DIR, cls))
        and os.path.isdir(os.path.join(TEST_DIR, cls))
        and any(
            name.lower().endswith(('.jpg', '.jpeg', '.png'))
            for name in os.listdir(os.path.join(TRAIN_DIR, cls))
        )
        and any(
            name.lower().endswith(('.jpg', '.jpeg', '.png'))
            for name in os.listdir(os.path.join(TEST_DIR, cls))
        )
        for cls in CLASS_FOLDERS
    )
    if split_is_ready:
        print("[INFO] Complete train/test split already exists. Skipping split.")
        return

    print("[INFO] Splitting dataset into Train ({:.0%}) and Test ({:.0%})...".format(
        1 - TEST_SPLIT, TEST_SPLIT))

    for cls in CLASS_FOLDERS:
        src_dir = os.path.join(BASE_DIR, cls)
        if not os.path.isdir(src_dir):
            print(f"  [WARN] Class folder not found: {cls}")
            continue

        # Get all image files
        images = [f for f in os.listdir(src_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Shuffle and split
        np.random.shuffle(images)
        split_idx = int(len(images) * (1 - TEST_SPLIT))
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        # Create directories
        train_cls_dir = os.path.join(TRAIN_DIR, cls)
        test_cls_dir = os.path.join(TEST_DIR, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(test_cls_dir, exist_ok=True)

        # Copy files (using copy to preserve originals)
        for img in train_imgs:
            src = os.path.join(src_dir, img)
            dst = os.path.join(train_cls_dir, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        for img in test_imgs:
            src = os.path.join(src_dir, img)
            dst = os.path.join(test_cls_dir, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        print(f"  {cls}: {len(train_imgs)} train, {len(test_imgs)} test")

    print("[INFO] Dataset split complete!\n")


# Run the split
split_dataset()

# =============================================================================
# STEP 2: VERIFY DATASET
# =============================================================================
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("\n--- Training Set ---")
total_train = 0
for filepath in sorted(glob.glob(TRAIN_DIR + '/*/')):
    files = glob.glob(filepath + '*')
    count = len(files)
    total_train += count
    print(f"  {count:>5} \t {Path(filepath).name}")
print(f"  {'─'*30}")
print(f"  Total: {total_train} training images")

print("\n--- Test Set ---")
total_test = 0
for filepath in sorted(glob.glob(TEST_DIR + '/*/')):
    files = glob.glob(filepath + '*')
    count = len(files)
    total_test += count
    print(f"  {count:>5} \t {Path(filepath).name}")
print(f"  {'─'*30}")
print(f"  Total: {total_test} test images")

# =============================================================================
# STEP 3: DETERMINE NUMBER OF CLASSES
# =============================================================================
n_classes = len(glob.glob(TRAIN_DIR + '/*/'))
print(f"\n[INFO] Number of classes detected: {n_classes}")
print(f"[INFO] Validation split: {VALIDATION_SPLIT:.0%} of training data")
print(f"[INFO] Image size: {WIDTH}x{HEIGHT}x{DEPTH}")
print(f"[INFO] Batch size: {BS}")
print(f"[INFO] Max epochs: {MAX_EPOCHS} (early stopping with patience={PATIENCE})")
print(f"[INFO] Learning rate: {INIT_LR}")

# =============================================================================
# STEP 4: BUILD CNN MODEL
# =============================================================================
def get_model():
    model = Sequential()
    inputShape = (HEIGHT, WIDTH, DEPTH)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (DEPTH, HEIGHT, WIDTH)
        chanDim = 1

    # Block 1: Conv -> BN -> Pool -> Dropout
    model.add(Input(shape=inputShape))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    # Block 2: Conv -> BN -> Conv -> BN -> Pool -> Dropout
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3: Conv -> BN -> Conv -> BN -> Pool -> Dropout
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Global pooling keeps the classifier compact and reduces overfitting.
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    # Compile with categorical_crossentropy (matches softmax + categorical labels)
    opt = Adam(learning_rate=INIT_LR)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )
    return model


print("\n[INFO] Building CNN model...")
model = get_model()
model.summary()

# Save model architecture diagram
try:
    plot_model(model, os.path.join(BASE_DIR, 'model.png'), show_shapes=True)
    print("[INFO] Model architecture saved to model.png")
except Exception as e:
    print(f"[WARN] Could not save model plot (graphviz may not be installed): {e}")

# =============================================================================
# STEP 5: DATA GENERATORS (Train + Validation from train/, Test from test/)
# =============================================================================
print("\n[INFO] Setting up data generators...")

# Training data generator WITH augmentation + validation split
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=5,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)

# Validation must use deterministic preprocessing without augmentation.
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT
)

# Test data generator - ONLY rescaling, no augmentation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Training generator (80% of train/)
train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    subset='training',
    target_size=(WIDTH, HEIGHT),
    color_mode="rgb",
    batch_size=BS,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

# Validation generator (20% of train/)
valid_generator = validation_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    subset='validation',
    target_size=(WIDTH, HEIGHT),
    color_mode="rgb",
    batch_size=BS,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

# Test generator (from test/)
test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=(WIDTH, HEIGHT),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

print(f"\n[INFO] Class indices: {train_generator.class_indices}")
print(f"[INFO] Training samples:   {train_generator.samples}")
print(f"[INFO] Validation samples: {valid_generator.samples}")
print(f"[INFO] Test samples:       {test_generator.samples}")

# =============================================================================
# STEP 6: TRAIN THE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)

STEP_SIZE_TRAIN = len(train_generator)
STEP_SIZE_VALID = len(valid_generator)
STEP_SIZE_TEST = len(test_generator)

# Balance the loss contribution from the uneven class distribution.
class_counts = np.bincount(train_generator.classes)
class_weight = {
    class_index: train_generator.samples / (n_classes * count)
    for class_index, count in enumerate(class_counts)
    if count > 0
}

# -------------------------------------------------------------------------
# CALLBACKS — the professional way to determine epochs
# -------------------------------------------------------------------------
# 1. EarlyStopping: Monitors val_loss. If it doesn't improve by MIN_DELTA
#    for PATIENCE consecutive epochs, training stops automatically.
#    'restore_best_weights=True' rolls back to the best epoch's weights.
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    restore_best_weights=True,
    verbose=1
)

# 2. ModelCheckpoint: Saves the model at the epoch with the lowest val_loss.
#    Even if early stopping triggers, you'll always have the best weights on disk.
checkpoint = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# 3. ReduceLROnPlateau: If val_loss stalls for 5 epochs, reduce the learning
#    rate by 50%. This helps the model escape plateaus before early stopping
#    decides to give up entirely.
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stop, checkpoint, reduce_lr]

print(f"[INFO] Callbacks configured:")
print(f"  - EarlyStopping: patience={PATIENCE}, min_delta={MIN_DELTA}")
print(f"  - ModelCheckpoint: saving best to {MODEL_SAVE_PATH}")
print(f"  - ReduceLROnPlateau: factor=0.5, patience=5, min_lr=1e-6")

start_time = time.time()

history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=MAX_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight
)

elapsed = time.time() - start_time
actual_epochs = len(history.history['loss'])
print(f"\n[INFO] Training completed in {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
print(f"[INFO] Stopped at epoch {actual_epochs}/{MAX_EPOCHS}")

# Identify the best epoch
best_epoch = np.argmin(history.history['val_loss']) + 1
best_val_loss = min(history.history['val_loss'])
best_val_acc = history.history['val_accuracy'][best_epoch - 1]
print(f"[INFO] Best epoch: {best_epoch} (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")

# =============================================================================
# STEP 7: PLOT TRAINING HISTORY
# =============================================================================
print("\n[INFO] Generating training plots...")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy')
ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(epochs_range, loss, 'b-', label='Training Loss')
ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'training_history.png'), dpi=150)
print(f"[INFO] Training plots saved to training_history.png")

# =============================================================================
# STEP 8: EVALUATE ON TEST SET
# =============================================================================
print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)

test_generator.reset()

# Evaluate
test_loss, test_acc = model.evaluate(
    test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1
)
print(f"\n[RESULT] Test Accuracy: {test_acc:.4f}")
print(f"[RESULT] Test Loss:     {test_loss:.4f}")

# Predict
test_generator.reset()
pred = model.predict(
    test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1
)

predicted_class_indices = np.argmax(pred, axis=1)
labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# Classification Report
true_classes = test_generator.classes[:len(predicted_class_indices)]
target_names = list(train_generator.class_indices.keys())

print("\n--- Classification Report ---")
classification_report_text = classification_report(
    true_classes,
    predicted_class_indices,
    target_names=target_names,
    zero_division=0
)
print(classification_report_text)
classification_report_data = classification_report(
    true_classes,
    predicted_class_indices,
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

# =============================================================================
# STEP 9: SAVE RESULTS
# =============================================================================
print("\n[INFO] Saving prediction results...")

# Class distribution
print("\n--- Training Class Distribution ---")
print(pd.Series(train_generator.classes).value_counts().rename(
    index=dict((v, k) for k, v in train_generator.class_indices.items())
))

print("\n--- Prediction Distribution ---")
print(pd.Series(predictions).value_counts())

# Save submission CSV
filenames = test_generator.filenames[:len(predictions)]
results = pd.DataFrame({"image_id": filenames, "label": predictions})
results.image_id = results.image_id.str.replace('\\', '/', regex=False)
results_path = os.path.join(BASE_DIR, "submission.csv")
results.to_csv(results_path, index=False)
print(f"\n[INFO] Results saved to {results_path}")
print(results.head(10))

# Save numeric training and post-training metrics for later analysis.
history_metrics = pd.DataFrame(history.history)
history_metrics.insert(0, "epoch", range(1, len(history_metrics) + 1))
history_metrics_path = os.path.join(BASE_DIR, "training_metrics.csv")
history_metrics.to_csv(history_metrics_path, index=False)

report_metrics_path = os.path.join(BASE_DIR, "classification_report.csv")
pd.DataFrame(classification_report_data).transpose().to_csv(report_metrics_path)

confusion_matrix_path = os.path.join(BASE_DIR, "confusion_matrix.csv")
confusion_matrix_values = confusion_matrix(
    true_classes, predicted_class_indices, labels=range(n_classes)
)
pd.DataFrame(
    confusion_matrix_values,
    index=target_names,
    columns=target_names
).to_csv(confusion_matrix_path)

normalized_confusion_matrix = confusion_matrix_values.astype(np.float64)
row_totals = normalized_confusion_matrix.sum(axis=1, keepdims=True)
np.divide(
    normalized_confusion_matrix,
    row_totals,
    out=normalized_confusion_matrix,
    where=row_totals != 0
)
normalized_confusion_matrix_path = os.path.join(
    BASE_DIR, "confusion_matrix_normalized.csv"
)
pd.DataFrame(
    normalized_confusion_matrix,
    index=target_names,
    columns=target_names
).to_csv(normalized_confusion_matrix_path)

# Save final model (best weights were already restored by EarlyStopping)
model_path = os.path.join(BASE_DIR, "paddy_disease_cnn_model.keras")
model.save(model_path)
print(f"[INFO] Final model saved to {model_path}")
print(f"[INFO] Best checkpoint saved at {MODEL_SAVE_PATH}")
print(f"[INFO] Best epoch was {best_epoch}/{actual_epochs} (val_loss={best_val_loss:.4f})")

summary_metrics = {
    "status": "completed",
    "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "random_seed": SEED,
    "dataset": {
        "class_count": n_classes,
        "class_names": target_names,
        "class_indices": train_generator.class_indices,
        "training_samples": train_generator.samples,
        "validation_samples": valid_generator.samples,
        "test_samples": test_generator.samples,
        "training_class_counts": {
            target_names[index]: int(count)
            for index, count in enumerate(class_counts)
        }
    },
    "preprocessing": {
        "image_size": [WIDTH, HEIGHT],
        "channels": DEPTH,
        "color_mode": "rgb",
        "rescale": "1/255",
        "validation_split": VALIDATION_SPLIT,
        "test_split": TEST_SPLIT
    },
    "augmentation": {
        "rotation_range": 5,
        "shear_range": 0.2,
        "zoom_range": 0.2,
        "width_shift_range": 0.05,
        "height_shift_range": 0.05,
        "horizontal_flip": True,
        "fill_mode": "nearest"
    },
    "training": {
        "batch_size": BS,
        "max_epochs": MAX_EPOCHS,
        "actual_epochs": actual_epochs,
        "best_epoch": int(best_epoch),
        "initial_learning_rate": INIT_LR,
        "final_learning_rate": float(tf.keras.backend.get_value(
            model.optimizer.learning_rate
        )),
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "class_weight": class_weight,
        "early_stopping": {
            "monitor": "val_loss",
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "restore_best_weights": True
        },
        "reduce_lr_on_plateau": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 5,
            "min_learning_rate": 1e-6
        }
    },
    "model": {
        "input_shape": list(model.input_shape[1:]),
        "output_classes": int(model.output_shape[-1]),
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(
            np.prod(variable.shape) for variable in model.trainable_weights
        )),
        "non_trainable_parameters": int(sum(
            np.prod(variable.shape) for variable in model.non_trainable_weights
        ))
    },
    "metrics": {
        "best_validation_loss": float(best_val_loss),
        "best_validation_accuracy": float(best_val_acc),
        "final_training_loss": float(history.history["loss"][-1]),
        "final_training_accuracy": float(history.history["accuracy"][-1]),
        "final_validation_loss": float(history.history["val_loss"][-1]),
        "final_validation_accuracy": float(history.history["val_accuracy"][-1]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_precision": float(classification_report_data["macro avg"]["precision"]),
        "test_macro_recall": float(classification_report_data["macro avg"]["recall"]),
        "test_macro_f1": float(classification_report_data["macro avg"]["f1-score"]),
        "test_weighted_precision": float(classification_report_data["weighted avg"]["precision"]),
        "test_weighted_recall": float(classification_report_data["weighted avg"]["recall"]),
        "test_weighted_f1": float(classification_report_data["weighted avg"]["f1-score"]),
        "per_class": {
            name: {
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
                "support": int(values["support"])
            }
            for name, values in classification_report_data.items()
            if name in target_names
        }
    },
    "artifacts": {
        "best_model": os.path.basename(MODEL_SAVE_PATH),
        "final_model": os.path.basename(model_path),
        "training_history": os.path.basename(history_metrics_path),
        "classification_report": os.path.basename(report_metrics_path),
        "confusion_matrix": os.path.basename(confusion_matrix_path),
        "normalized_confusion_matrix": os.path.basename(normalized_confusion_matrix_path)
    }
}
summary_metrics_path = os.path.join(BASE_DIR, "post_training_metrics.json")
with open(summary_metrics_path, "w", encoding="utf-8") as metrics_file:
    json.dump(summary_metrics, metrics_file, indent=2)

print(f"[INFO] Training metrics saved to {history_metrics_path}")
print(f"[INFO] Classification report saved to {report_metrics_path}")
print(f"[INFO] Confusion matrix saved to {confusion_matrix_path}")
print(f"[INFO] Normalized confusion matrix saved to {normalized_confusion_matrix_path}")
print(f"[INFO] Summary metrics saved to {summary_metrics_path}")

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)