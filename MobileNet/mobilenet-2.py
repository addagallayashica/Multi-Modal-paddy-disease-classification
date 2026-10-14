# =============================================================================
# MobileNet Transfer Learning for Paddy Disease Classification
# Cleaned Python script — parameters aligned with cnn.py for fair comparison
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# =============================================================================
# CONFIGURATION  (matched with cnn.py for fair comparison)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
RESULTS_DIR = os.path.join(BASE_DIR, 'result_mobilenet_a')

SEED = 123
MAX_EPOCHS = 75             # Total epoch ceiling across both training phases
HEAD_EPOCHS = 15             # Initial head-only training phase
HEAD_LR = 1e-3               # Classifier head learning rate
FINE_TUNE_LR = 1e-5          # Lower rate for pretrained-layer fine-tuning
BS = 32                     
WIDTH = 256                 
HEIGHT = 256                
DEPTH = 3                   
VALIDATION_SPLIT = 0.2      # 20% of training data for validation
TEST_SPLIT = 0.15           # 15% of total data reserved for testing

# --- Early Stopping & Checkpoint Configuration (matched with cnn.py) ---
PATIENCE = 10               # Well-proportioned for 75 epochs with fine-tuning
MIN_DELTA = 1e-4            # Same as CNN
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'paddy_disease_mobilenet_best.keras')

# --- MobileNet Fine-Tuning Configuration ---
FINE_TUNE_LAYERS = 24       # Number of top layers to unfreeze for fine-tuning

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# STEP 1: SPLIT DATASET INTO TRAIN AND TEST DIRECTORIES
# =============================================================================
# Reuses the same split logic as cnn.py to ensure identical data partitions.

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
print(f"[INFO] Head learning rate: {HEAD_LR}")
print(f"[INFO] Fine-tuning learning rate: {FINE_TUNE_LR}")

# =============================================================================
# STEP 4: BUILD MOBILENET MODEL
# =============================================================================
def create_model_mobilenet(input_shape, n_classes, optimizer):
    """
    Builds a MobileNet transfer-learning model.

    input_shape: tuple - the shape of input images (height, width, channels)
    n_classes:   int   - number of classes for the output layer
    optimizer:   keras optimizer instance
    """
    # Load MobileNet pre-trained on ImageNet, excluding the top FC layers
    conv_base = MobileNet(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # Train the new head first. BatchNormalization layers remain frozen during
    # the later fine-tuning phase because the dataset is relatively small.
    for layer in conv_base.layers:
        layer.trainable = False

    # New classification head
    top_model = conv_base.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(256, activation='relu')(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dropout(0.5)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)

    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compile — same loss & metrics as CNN
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, conv_base


print("\n[INFO] Building MobileNet model...")
input_shape = (HEIGHT, WIDTH, DEPTH)
opt = Adam(learning_rate=HEAD_LR)
model, conv_base = create_model_mobilenet(input_shape, n_classes, opt)
model.summary()

# Save model architecture diagram
try:
    plot_model(model, os.path.join(RESULTS_DIR, 'model_mobilenet.png'), show_shapes=True)
    print("[INFO] Model architecture saved to model_mobilenet.png")
except Exception as e:
    print(f"[WARN] Could not save model plot (graphviz may not be installed): {e}")

# =============================================================================
# STEP 5: DATA GENERATORS
# =============================================================================
print("\n[INFO] Setting up data generators...")

# Training data generator WITH augmentation + validation split
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=5,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)

# Validation — ONLY rescaling, no augmentation (same as CNN)
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SPLIT
)

# Test data generator — ONLY rescaling, no augmentation (same as CNN)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

# Test generator (from test/) — same as CNN: class_mode="categorical", batch_size=1
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

# Balance the loss contribution from the uneven class distribution (same as CNN)
class_counts = np.bincount(train_generator.classes)
class_weight = {
    class_index: train_generator.samples / (n_classes * count)
    for class_index, count in enumerate(class_counts)
    if count > 0
}

start_time = time.time()

def make_callbacks(min_lr):
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=1
        )
    ]

print(f"[INFO] Phase 1: training classifier head for up to {HEAD_EPOCHS} epochs")
head_history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=min(HEAD_EPOCHS, MAX_EPOCHS),
    callbacks=make_callbacks(1e-6),
    class_weight=class_weight
)

head_epochs_completed = len(head_history.history['loss'])
fine_tune_start = head_epochs_completed

if head_epochs_completed < MAX_EPOCHS:
    print(f"[INFO] Phase 2: fine-tuning top {FINE_TUNE_LAYERS} non-BN layers")
    for layer in conv_base.layers:
        layer.trainable = False
    for layer in conv_base.layers[-FINE_TUNE_LAYERS:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        initial_epoch=fine_tune_start,
        epochs=MAX_EPOCHS,
        callbacks=make_callbacks(1e-7),
        class_weight=class_weight
    )
else:
    fine_tune_history = None

history_data = {}
for key in head_history.history:
    history_data[key] = list(head_history.history[key])
if fine_tune_history is not None:
    for key in fine_tune_history.history:
        history_data.setdefault(key, [])
        history_data[key].extend(fine_tune_history.history[key])

class TrainingHistory:
    def __init__(self, data):
        self.history = data

history = TrainingHistory(history_data)

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
plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'), dpi=150)
print(f"[INFO] Training plots saved to {RESULTS_DIR}/training_history.png")

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
# STEP 9: CONFUSION MATRIX HEATMAP
# =============================================================================
print("\n[INFO] Generating confusion matrix heatmap...")

try:
    import seaborn as sns

    cm = confusion_matrix(true_classes, predicted_class_indices, labels=range(n_classes))
    fig_cm, ax_cm = plt.subplots(1, 1, figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        square=True,
        xticklabels=target_names,
        yticklabels=target_names,
        fmt='d',
        cmap=plt.cm.Blues,
        cbar=False,
        ax=ax_cm
    )
    ax_cm.set_xticklabels(ax_cm.get_xticklabels(), fontsize=10, rotation=45, ha="right")
    ax_cm.set_yticklabels(ax_cm.get_yticklabels(), fontsize=10)
    ax_cm.set_ylabel('True Label', fontsize=12)
    ax_cm.set_xlabel('Predicted Label', fontsize=12)
    ax_cm.set_title('MobileNet — Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
    print(f"[INFO] Confusion matrix heatmap saved to {RESULTS_DIR}/confusion_matrix.png")
except ImportError:
    print("[WARN] seaborn not installed — skipping confusion matrix heatmap.")

# =============================================================================
# STEP 10: SAVE ALL RESULTS TO result_mobilenet_a/
# =============================================================================
print("\n[INFO] Saving all results to result_mobilenet_a/ ...")

# Class distribution
print("\n--- Training Class Distribution ---")
print(pd.Series(train_generator.classes).value_counts().rename(
    index=dict((v, k) for k, v in train_generator.class_indices.items())
))

print("\n--- Prediction Distribution ---")
print(pd.Series(predictions).value_counts())

# 1. Submission CSV
filenames = test_generator.filenames[:len(predictions)]
results = pd.DataFrame({"image_id": filenames, "label": predictions})
results.image_id = results.image_id.str.replace('\\', '/', regex=False)
results_path = os.path.join(RESULTS_DIR, "submission.csv")
results.to_csv(results_path, index=False)
print(f"\n[INFO] Results saved to {results_path}")
print(results.head(10))

# 2. Training metrics CSV (epoch-by-epoch history)
history_metrics = pd.DataFrame(history.history)
history_metrics.insert(0, "epoch", range(1, len(history_metrics) + 1))
history_metrics_path = os.path.join(RESULTS_DIR, "training_metrics.csv")
history_metrics.to_csv(history_metrics_path, index=False)

# 3. Classification report CSV
report_metrics_path = os.path.join(RESULTS_DIR, "classification_report.csv")
pd.DataFrame(classification_report_data).transpose().to_csv(report_metrics_path)

# 4. Confusion matrix CSV
confusion_matrix_path = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
pd.DataFrame(
    confusion_matrix(true_classes, predicted_class_indices, labels=range(n_classes)),
    index=target_names,
    columns=target_names
).to_csv(confusion_matrix_path)

# 5. Summary metrics JSON
summary_metrics = {
    "model": "MobileNet (Transfer Learning)",
    "actual_epochs": actual_epochs,
    "best_epoch": int(best_epoch),
    "best_validation_loss": float(best_val_loss),
    "best_validation_accuracy": float(best_val_acc),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "class_count": n_classes,
    "training_samples": train_generator.samples,
    "validation_samples": valid_generator.samples,
    "test_samples": test_generator.samples,
    "training_time_seconds": round(elapsed, 1),
    "fine_tune_layers": FINE_TUNE_LAYERS,
    "max_epochs": MAX_EPOCHS,
    "head_learning_rate": HEAD_LR,
    "fine_tune_learning_rate": FINE_TUNE_LR,
    "head_epochs": HEAD_EPOCHS,
    "batch_size": BS,
    "image_size": f"{WIDTH}x{HEIGHT}x{DEPTH}",
    "patience": PATIENCE
}
summary_metrics_path = os.path.join(RESULTS_DIR, "post_training_metrics.json")
with open(summary_metrics_path, "w", encoding="utf-8") as metrics_file:
    json.dump(summary_metrics, metrics_file, indent=2)

print(f"[INFO] Training metrics saved to {history_metrics_path}")
print(f"[INFO] Classification report saved to {report_metrics_path}")
print(f"[INFO] Confusion matrix saved to {confusion_matrix_path}")
print(f"[INFO] Summary metrics saved to {summary_metrics_path}")

# Save final model (best weights were already restored by EarlyStopping)
model_path = os.path.join(BASE_DIR, "paddy_disease_mobilenet_model.keras")
model.save(model_path)
print(f"[INFO] Final model saved to {model_path}")
print(f"[INFO] Best checkpoint saved at {MODEL_SAVE_PATH}")
print(f"[INFO] Best epoch was {best_epoch}/{actual_epochs} (val_loss={best_val_loss:.4f})")

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)