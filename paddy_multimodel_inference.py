"""Run the four trained Paddy Doctor classifiers over an unseen image folder.

The saved training pipelines use model-specific image sizes and preprocessing.
This runner follows those checkpoint contracts and supplies training-derived,
neutral metadata values to both multimodal models.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import io
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "valid" / "test_images"
DEFAULT_OUTPUT_CSV = BASE_DIR / "paddy_multimodel_inference_results.csv"
TRAIN_DIR = BASE_DIR / "train"
METADATA_CSV = BASE_DIR / "metadata.csv"

CLASS_NAMES = (
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
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
QUALITY_COLUMNS = (
    "image_id",
    "aspect_ratio",
    "brightness",
    "is_low_res",
    "is_poorly_exposed",
)
CSV_COLUMNS = (
    *QUALITY_COLUMNS,
    "pred_cnn",
    "pred_cnn_metadata",
    "pred_mobilenet",
    "pred_mobilenet_metadata",
)

# Set from the training metadata once, before the first inference batch.
DEFAULT_AGE_SCALED: dict[str, np.float32] = {}
DEFAULT_VARIETY_ENCODED = np.int32(0)

MODEL_CONFIGS = (
    {
        "key": "cnn",
        "display_name": "CNN",
        "path": BASE_DIR / "paddy_disease_cnn_best.keras",
        "kind": "cnn",
        "results_dir": BASE_DIR / "result_cnn_a",
    },
    {
        "key": "cnn_metadata",
        "display_name": "CNN + Metadata",
        "path": BASE_DIR / "result_custom_cnn_B" / "paddy_disease_custom_cnn_B_best.keras",
        "metrics": BASE_DIR / "result_custom_cnn_B" / "post_training_metrics.json",
        "kind": "cnn_metadata",
        "results_dir": BASE_DIR / "result_custom_cnn_B",
    },
    {
        "key": "mobilenet",
        "display_name": "MobileNet",
        "path": BASE_DIR / "paddy_disease_mobilenet_best.keras",
        "kind": "mobilenet",
        "results_dir": BASE_DIR / "result_mobilenet_a",
    },
    {
        "key": "mobilenet_metadata",
        "display_name": "MobileNet + Metadata",
        "path": BASE_DIR / "result_mobilenet_B" / "paddy_disease_mobilenet_B_model.keras",
        "metrics": BASE_DIR / "result_mobilenet_B" / "post_training_metrics.json",
        "kind": "mobilenet_metadata",
        "results_dir": BASE_DIR / "result_mobilenet_B",
    },
)


@dataclass
class ImageRecord:
    image_id: str
    path: Path
    rgb: np.ndarray
    aspect_ratio: float
    brightness: float
    is_low_res: int
    is_poorly_exposed: int


@dataclass
class ModelRunner:
    key: str
    display_name: str
    kind: str
    model: tf.keras.Model
    height: int
    width: int
    results_dir: Path
    age_scaled: np.float32 | None = None


def configure_tensorflow() -> None:
    """Enable memory growth where possible without hiding configuration errors."""
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            logging.warning("Could not enable GPU memory growth for %s: %s", gpu, error)
    logging.info("TensorFlow %s; GPU devices: %s", tf.__version__, tf.config.list_physical_devices("GPU"))


def discover_images(input_dir: Path, expected_images: int | None) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}. "
            "Pass the image folder with --input-dir."
        )

    paths = sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )
    if not paths:
        raise ValueError(f"No supported image files found in {input_dir}")
    if expected_images is not None and len(paths) != expected_images:
        raise ValueError(
            f"Found {len(paths)} supported image files in {input_dir}; "
            f"expected {expected_images}. Use --expected-images 0 to disable this check."
        )
    return paths


def inspect_image(path: Path, input_dir: Path) -> ImageRecord | None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        logging.warning("Skipping unreadable or invalid image: %s", path)
        return None

    height, width = bgr.shape[:2]
    if height == 0 or width == 0:
        logging.warning("Skipping image with invalid dimensions: %s", path)
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    brightness = float(np.sqrt(np.mean(np.square(gray))))
    relative_path = path.relative_to(input_dir).as_posix()
    record = ImageRecord(
        image_id=relative_path,
        path=path,
        rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        aspect_ratio=float(width / height),
        brightness=brightness,
        is_low_res=int(width < 480 or height < 480),
        is_poorly_exposed=int(brightness < 40 or brightness > 220),
    )
    if record.is_low_res or record.is_poorly_exposed:
        logging.warning(
            "Image quality flag for %s: size=%dx%d aspect_ratio=%.4f "
            "brightness=%.2f low_res=%d poorly_exposed=%d",
            record.image_id,
            width,
            height,
            record.aspect_ratio,
            record.brightness,
            record.is_low_res,
            record.is_poorly_exposed,
        )
    return record


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training metadata metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def training_variety_counts() -> Counter[str]:
    if not METADATA_CSV.is_file():
        raise FileNotFoundError(
            f"Training metadata is required to select the most frequent variety: {METADATA_CSV}"
        )
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Training split is required to select the most frequent training variety: {TRAIN_DIR}"
        )

    metadata = pd.read_csv(
        METADATA_CSV,
        dtype={"image_id": str, "variety": str},
        usecols=["image_id", "variety"],
    )
    metadata["image_id"] = metadata["image_id"].str.strip().str.lower()
    metadata["variety"] = (
        metadata["variety"].fillna("unknown").str.strip().str.lower().replace("", "unknown")
    )
    if metadata["image_id"].duplicated().any():
        raise ValueError(f"{METADATA_CSV} contains duplicate image_id values.")

    train_ids = {
        path.name.lower()
        for path in TRAIN_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    counts = Counter(
        metadata.loc[metadata["image_id"].isin(train_ids), "variety"].tolist()
    )
    if not counts:
        raise ValueError(
            "Could not match any training images to metadata.csv; "
            "cannot derive the required synthetic variety value."
        )
    return counts


def initialize_metadata_defaults(runners: list[ModelRunner]) -> str:
    """Derive neutral metadata values from each model's recorded training set."""
    global DEFAULT_VARIETY_ENCODED

    counts = training_variety_counts()
    most_frequent_variety = min(counts, key=lambda name: (-counts[name], name))
    encoded_indices: list[int] = []
    routed_varieties: list[str] = []

    for runner in runners:
        if runner.kind not in {"cnn_metadata", "mobilenet_metadata"}:
            continue
        metrics = load_metrics(MODEL_CONFIGS_BY_KEY[runner.key]["metrics"])
        age_median = float(metrics["age_training_median"])
        age_mean = float(metrics["age_training_mean"])
        age_std = float(metrics["age_training_std"])
        if not np.isfinite([age_median, age_mean, age_std]).all() or age_std <= 0:
            raise ValueError(f"Invalid training age statistics for {runner.display_name}.")
        runner.age_scaled = np.float32((age_median - age_mean) / age_std)
        DEFAULT_AGE_SCALED[runner.key] = runner.age_scaled

        lookups = [
            layer
            for layer in runner.model.layers
            if isinstance(layer, tf.keras.layers.StringLookup)
        ]
        if len(lookups) != 1:
            raise ValueError(
                f"Expected exactly one variety StringLookup in {runner.display_name}; "
                f"found {len(lookups)}."
            )
        vocabulary = lookups[0].get_vocabulary()
        selected_variety = most_frequent_variety
        if selected_variety not in vocabulary:
            # MobileNet+B maps training categories below its configured count
            # threshold to "other" before fitting StringLookup.
            rare_threshold = int(metrics.get("min_variety_count", 20))
            if (
                runner.kind == "mobilenet_metadata"
                and counts[selected_variety] < rare_threshold
                and "other" in vocabulary
            ):
                selected_variety = "other"
            else:
                raise ValueError(
                    f"Most frequent training variety {selected_variety!r} is not "
                    f"in {runner.display_name}'s saved vocabulary."
                )
        routed_varieties.append(selected_variety)
        encoded_indices.append(vocabulary.index(selected_variety))

    if not encoded_indices or len(set(routed_varieties)) != 1:
        raise ValueError(
            "The multimodal models do not share a compatible variety vocabulary."
        )
    DEFAULT_VARIETY_ENCODED = np.int32(encoded_indices[0])
    if len(set(encoded_indices)) != 1:
        raise ValueError(
            "The multimodal models encode the selected training variety differently."
        )
    most_frequent_variety = routed_varieties[0]
    logging.info(
        "Synthetic metadata: variety=%r, saved-vocabulary index=%d, normalized ages=%s",
        most_frequent_variety,
        int(DEFAULT_VARIETY_ENCODED),
        {key: float(value) for key, value in DEFAULT_AGE_SCALED.items()},
    )
    return most_frequent_variety


MODEL_CONFIGS_BY_KEY = {config["key"]: config for config in MODEL_CONFIGS}


def input_name(tensor: Any) -> str:
    name = str(tensor.name).split(":", maxsplit=1)[0].split("/", maxsplit=1)[0]
    return name.lower()


def build_native_cnn_from_v3_weights(model_path: Path) -> tf.keras.Model:
    """Load the native CNN's Keras 3 weights under a TensorFlow 2.10 runtime."""
    with zipfile.ZipFile(model_path) as archive:
        if "model.weights.h5" not in archive.namelist():
            raise ValueError(f"No Keras 3 weights found in {model_path}.")
        weights_data = archive.read("model.weights.h5")

    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(256, 256, 3), name="input_layer"),
            layers.Conv2D(32, (3, 3), padding="same", name="conv2d"),
            layers.Activation("relu", name="activation"),
            layers.BatchNormalization(axis=-1, name="batch_normalization"),
            layers.MaxPooling2D(pool_size=(3, 3), name="max_pooling2d"),
            layers.Dropout(0.25, name="dropout"),
            layers.Conv2D(64, (3, 3), padding="same", name="conv2d_1"),
            layers.Activation("relu", name="activation_1"),
            layers.BatchNormalization(axis=-1, name="batch_normalization_1"),
            layers.Conv2D(64, (3, 3), padding="same", name="conv2d_2"),
            layers.Activation("relu", name="activation_2"),
            layers.BatchNormalization(axis=-1, name="batch_normalization_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"),
            layers.Dropout(0.25, name="dropout_1"),
            layers.Conv2D(128, (3, 3), padding="same", name="conv2d_3"),
            layers.Activation("relu", name="activation_3"),
            layers.BatchNormalization(axis=-1, name="batch_normalization_3"),
            layers.Conv2D(128, (3, 3), padding="same", name="conv2d_4"),
            layers.Activation("relu", name="activation_4"),
            layers.BatchNormalization(axis=-1, name="batch_normalization_4"),
            layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2"),
            layers.Dropout(0.25, name="dropout_2"),
            layers.GlobalAveragePooling2D(name="global_average_pooling2d"),
            layers.Dense(256, name="dense"),
            layers.Activation("relu", name="activation_5"),
            layers.BatchNormalization(axis=-1, name="batch_normalization_5"),
            layers.Dropout(0.5, name="dropout_3"),
            layers.Dense(len(CLASS_NAMES), name="dense_1"),
            layers.Activation("softmax", name="activation_6"),
        ],
        name="sequential",
    )

    loaded_layers: set[str] = set()
    with h5py.File(io.BytesIO(weights_data), "r") as weights_file:
        for layer_group in weights_file["layers"].values():
            variables = layer_group.get("vars")
            if variables is None or len(variables) == 0:
                continue
            saved_name = variables.attrs.get("name")
            if isinstance(saved_name, bytes):
                saved_name = saved_name.decode("utf-8")
            if not saved_name:
                raise ValueError(f"Unnamed layer weights in {model_path}.")

            layer = model.get_layer(str(saved_name))
            saved_weights = [
                np.asarray(variables[key])
                for key in sorted(variables, key=lambda value: int(value))
            ]
            expected_weights = layer.get_weights()
            if [weight.shape for weight in saved_weights] != [
                weight.shape for weight in expected_weights
            ]:
                raise ValueError(
                    f"Weight shapes do not match for CNN layer {saved_name!r}."
                )
            layer.set_weights(saved_weights)
            loaded_layers.add(str(saved_name))

    expected_layers = {
        layer.name for layer in model.layers if layer.weights
    }
    if loaded_layers != expected_layers:
        raise ValueError(
            "CNN checkpoint weights were incomplete: "
            f"missing={sorted(expected_layers - loaded_layers)}, "
            f"unexpected={sorted(loaded_layers - expected_layers)}."
        )
    return model


def load_runners() -> list[ModelRunner]:
    runners: list[ModelRunner] = []
    for config in MODEL_CONFIGS:
        model_path = config["path"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained model file not found: {model_path}")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except (OSError, TypeError, ValueError) as error:
            if config["kind"] != "cnn" or not zipfile.is_zipfile(model_path):
                raise RuntimeError(
                    f"Could not load {config['display_name']} checkpoint "
                    f"{model_path}: {error}"
                ) from error
            logging.warning(
                "Native CNN checkpoint uses a newer Keras archive format; "
                "reconstructing its cnn.py architecture and loading stored weights."
            )
            model = build_native_cnn_from_v3_weights(model_path)

        image_inputs = [tensor for tensor in model.inputs if len(tensor.shape) == 4]
        if len(image_inputs) != 1:
            raise ValueError(
                f"{config['display_name']} must have exactly one 4D image input; "
                f"found {len(image_inputs)}."
            )
        image_shape = image_inputs[0].shape
        if image_shape[-1] != 3 or image_shape[1] is None or image_shape[2] is None:
            raise ValueError(
                f"Unsupported image input shape for {config['display_name']}: {image_shape}"
            )
        if len(model.inputs) != (1 if config["kind"] in {"cnn", "mobilenet"} else 3):
            raise ValueError(
                f"Unexpected input count for {config['display_name']}: "
                f"expected {1 if config['kind'] in {'cnn', 'mobilenet'} else 3}, "
                f"found {len(model.inputs)}."
            )
        if int(model.output_shape[-1]) != len(CLASS_NAMES):
            raise ValueError(
                f"{config['display_name']} returns {model.output_shape[-1]} classes, "
                f"but {len(CLASS_NAMES)} class names are configured."
            )

        runners.append(
            ModelRunner(
                key=config["key"],
                display_name=config["display_name"],
                kind=config["kind"],
                model=model,
                height=int(image_shape[1]),
                width=int(image_shape[2]),
                results_dir=config["results_dir"],
            )
        )
        logging.info(
            "Loaded %s from %s with input size %dx%d",
            config["display_name"],
            model_path,
            image_shape[1],
            image_shape[2],
        )
    return runners


def resize_image(rgb: np.ndarray, height: int, width: int, pad: bool) -> np.ndarray:
    source_height, source_width = rgb.shape[:2]
    if not pad:
        return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def prepare_images(records: list[ImageRecord], runner: ModelRunner) -> np.ndarray:
    pad = runner.kind in {"cnn_metadata", "mobilenet_metadata"}
    batch = np.stack(
        [
            resize_image(record.rgb, runner.height, runner.width, pad)
            for record in records
        ]
    ).astype(np.float32)

    if runner.kind == "cnn":
        return batch / np.float32(255.0)
    if runner.kind == "mobilenet":
        # The original MobileNet generator used mobilenet.preprocess_input.
        return tf.keras.applications.mobilenet.preprocess_input(batch)
    if runner.kind == "cnn_metadata":
        # custom_cnn_B embeds Rescaling(1/255) in the saved model.
        return batch
    if runner.kind == "mobilenet_metadata":
        # mobilenet_B embeds MobileNetV2 preprocess_input in the saved model.
        return batch
    raise ValueError(f"Unknown model kind: {runner.kind}")


def predict_batch(
    records: list[ImageRecord],
    runner: ModelRunner,
    default_variety: str,
) -> tuple[list[str], list[float]]:
    images = prepare_images(records, runner)
    if runner.kind in {"cnn", "mobilenet"}:
        # Image-only checkpoints receive only the 4D image tensor.
        outputs = runner.model(images)
    else:
        if runner.age_scaled is None:
            raise RuntimeError(f"Synthetic age value was not initialized for {runner.display_name}.")

        variety_values = np.full((len(records),), default_variety, dtype=object)
        age_values = np.full(
            (len(records), 1), runner.age_scaled, dtype=np.float32
        )
        model_inputs: list[np.ndarray] = []
        for tensor in runner.model.inputs:
            name = input_name(tensor)
            if name == "image":
                model_inputs.append(images)
            elif name == "variety":
                model_inputs.append(variety_values)
            elif name == "age":
                model_inputs.append(age_values)
            else:
                raise ValueError(
                    f"Unrecognized input {tensor.name!r} in {runner.display_name}; "
                    "expected image, variety, and age."
                )
        # Both metadata checkpoints expose variety and age as separate model
        # inputs, so the paired input list preserves their trained signatures.
        outputs = runner.model(model_inputs)

    probabilities = np.asarray(outputs)
    if probabilities.ndim != 2 or probabilities.shape != (len(records), len(CLASS_NAMES)):
        raise ValueError(
            f"Unexpected prediction shape from {runner.display_name}: {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{runner.display_name} returned non-finite prediction scores.")
    predicted_indices = np.argmax(probabilities, axis=1)
    confidence_percent = np.clip(
        probabilities[np.arange(len(records)), predicted_indices] * 100.0,
        0.0,
        100.0,
    )
    return (
        [CLASS_NAMES[index] for index in predicted_indices],
        confidence_percent.astype(float).tolist(),
    )


def write_csv_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)


def write_prediction_counts(
    frame: pd.DataFrame,
    runners: list[ModelRunner],
    output_csv: Path,
) -> pd.DataFrame:
    count_columns: dict[str, Any] = {"disease_class": list(CLASS_NAMES)}
    for runner in runners:
        prediction_column = f"pred_{runner.key}"
        unexpected = set(frame[prediction_column].dropna()) - set(CLASS_NAMES)
        if unexpected:
            raise ValueError(
                f"Unexpected labels for {runner.display_name}: {sorted(unexpected)}"
            )
        counts = frame[prediction_column].value_counts().reindex(
            CLASS_NAMES, fill_value=0
        )
        count_columns[runner.display_name] = counts.to_numpy(dtype=np.int64)
        model_counts = pd.DataFrame(
            {
                "disease_class": CLASS_NAMES,
                "prediction_count": counts.to_numpy(dtype=np.int64),
            }
        )
        count_path = runner.results_dir / "class_prediction_counts.csv"
        write_csv_atomically(model_counts, count_path)
        logging.info("Saved per-class counts for %s to %s", runner.display_name, count_path)

    combined_counts = pd.DataFrame(count_columns)
    combined_counts_path = output_csv.with_name("paddy_multimodel_class_counts.csv")
    write_csv_atomically(combined_counts, combined_counts_path)
    logging.info("Saved combined per-class counts to %s", combined_counts_path)
    return combined_counts


def run_inference(
    input_dir: Path,
    output_csv: Path,
    batch_size: int,
    expected_images: int | None,
) -> pd.DataFrame:
    paths = discover_images(input_dir, expected_images)
    runners = load_runners()
    default_variety = initialize_metadata_defaults(runners)

    results: dict[str, list[str]] = {runner.key: [] for runner in runners}
    confidence_results: dict[str, list[float]] = {
        runner.key: [] for runner in runners
    }
    quality_rows: list[dict[str, Any]] = []
    processed_count = 0
    for offset in range(0, len(paths), batch_size):
        batch_records = [
            record
            for path in paths[offset : offset + batch_size]
            if (record := inspect_image(path, input_dir)) is not None
        ]
        if not batch_records:
            continue
        for runner in runners:
            batch_labels, batch_confidence = predict_batch(
                batch_records, runner, default_variety
            )
            results[runner.key].extend(batch_labels)
            confidence_results[runner.key].extend(batch_confidence)
        quality_rows.extend(
            {
                "image_id": record.image_id,
                "aspect_ratio": record.aspect_ratio,
                "brightness": record.brightness,
                "is_low_res": record.is_low_res,
                "is_poorly_exposed": record.is_poorly_exposed,
            }
            for record in batch_records
        )
        processed_count += len(batch_records)
        logging.info(
            "Completed images %d-%d of %d across all four models",
            processed_count - len(batch_records) + 1,
            processed_count,
            len(paths),
        )

    if not quality_rows:
        raise ValueError("No readable images were found; no result CSV was written.")
    if expected_images is not None and processed_count != expected_images:
        raise ValueError(
            f"Processed {processed_count} readable images; expected {expected_images}."
        )

    frame = pd.DataFrame(quality_rows)
    for runner in runners:
        frame[f"pred_{runner.key}"] = results[runner.key]
    frame = frame.loc[:, CSV_COLUMNS]
    write_csv_atomically(frame, output_csv)

    for runner in runners:
        prediction_frame = pd.DataFrame(
            {
                "image_id": frame["image_id"],
                "label": frame[f"pred_{runner.key}"],
                "confidence_percent": confidence_results[runner.key],
            }
        )
        result_path = runner.results_dir / "inference_predictions.csv"
        write_csv_atomically(prediction_frame, result_path)
        logging.info("Saved %s predictions to %s", runner.display_name, result_path)

    write_prediction_counts(frame, runners, output_csv)
    logging.info(
        "Saved %d image predictions to %s (low-resolution=%d, poor-exposure=%d)",
        len(frame),
        output_csv,
        int(frame["is_low_res"].sum()),
        int(frame["is_poorly_exposed"].sum()),
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify unseen Paddy Doctor images with all four trained models."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing unseen images (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Combined result CSV (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Images processed per model call; lower this if GPU memory is limited.",
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=0,
        help="Optional expected image count; set to 0 to disable the count check (default).",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    if args.expected_images < 0:
        parser.error("--expected-images must be 0 or greater.")
    if args.expected_images == 0:
        args.expected_images = None
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    configure_tensorflow()
    run_inference(
        input_dir=args.input_dir.resolve(),
        output_csv=args.output_csv.resolve(),
        batch_size=args.batch_size,
        expected_images=args.expected_images,
    )


if __name__ == "__main__":
    main()
