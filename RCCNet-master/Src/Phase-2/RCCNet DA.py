# ============================================================
# PHASE 2
# RCCNet - BANANA RIPENESS CLASSIFICATION
# WITH DATA AUGMENTATION
#
# FRAMEWORK:
#   TensorFlow / Keras only
#
# DATA:
#   C:\Users\E028.6\Downloads\Fruit\Banana
#
# EXPECTED STRUCTURE:
#
# Banana/
# ├── Ripe/
# │   ├── Train/
# │   └── Test/
# ├── Unripe/
# │   ├── Train/
# │   └── Test/
# └── Rotten/
#     ├── Train/
#     └── Test/
#
# ALSO SUPPORTED:
#
# Banana/
# ├── Train/
# │   ├── Ripe/
# │   ├── Unripe/
# │   └── Rotten/
# └── Test/
#     ├── Ripe/
#     ├── Unripe/
#     └── Rotten/
#
# ALSO SUPPORTED:
#
# Banana/
# ├── Ripe/
# ├── Unripe/
# └── Rotten/
#
# In the last case, the script creates:
#   80% Train
#   20% Test
#
# From Train:
#   90% Training
#   10% Validation
#
# Final effective ratio:
#   72% Training
#    8% Validation
#   20% Testing
#
# OUTPUT:
#   Accuracy graph
#   Loss graph
#   Confusion matrix
#   Complete epoch CSV
#   Classification report
#   Final metrics
#   Run information
#   Best model
#   Final model
# ============================================================

import os

# ------------------------------------------------------------
# Must be set before importing TensorFlow
# ------------------------------------------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
    CSVLogger
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\E028.6\Downloads\Fruit"
)

FRUIT_NAME = "Banana"

FRUIT_DIR = DATASET_ROOT / FRUIT_NAME

# Results are saved next to the Dataset folder
RESULTS_ROOT = (
    DATASET_ROOT.parent
    / "Fruit_Research_Results"
)

RESULTS_DIR = (
    RESULTS_ROOT
    / FRUIT_NAME
    / "RCCNet"
    / "DA"
)

# Three classes requested for Phase 2
CLASS_NAMES = [
    "Ripe",
    "Unripe",
    "Rotten"
]

NUM_CLASSES = 3

# RCCNet image size
IMG_WIDTH = 32
IMG_HEIGHT = 32

# Standard training settings
BATCH_SIZE = 64
EPOCHS = 30

# 10% of the training set becomes validation
VALIDATION_FROM_TRAIN = 0.10

RANDOM_STATE = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# 2. CREATE RESULT DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 4. GPU CHECK
# ============================================================

print()
print("=" * 80)
print("GPU / TENSORFLOW CHECK")
print("=" * 80)

print(
    "TensorFlow version:",
    tf.__version__
)

GPUS = tf.config.list_physical_devices("GPU")

if GPUS:

    print()
    print("GPU detected:")

    for gpu in GPUS:
        print("   ", gpu)

    try:

        for gpu in GPUS:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

        print(
            "GPU memory growth: ENABLED"
        )

    except RuntimeError as error:

        print(
            "GPU configuration message:"
        )

        print(error)

else:

    print()
    print(
        "WARNING: No GPU detected."
    )

    print(
        "Training will run on CPU."
    )

print("=" * 80)


# ============================================================
# 5. HEADER
# ============================================================

print()
print("=" * 80)
print("BANANA RIPENESS - RCCNet")
print("WITH DATA AUGMENTATION")
print("=" * 80)

print(
    "Dataset root:",
    DATASET_ROOT
)

print(
    "Fruit:",
    FRUIT_NAME
)

print(
    "Fruit folder:",
    FRUIT_DIR
)

print(
    "Results folder:",
    RESULTS_DIR
)

print(
    "Image size:",
    IMG_WIDTH,
    "x",
    IMG_HEIGHT
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Maximum epochs:",
    EPOCHS
)

print(
    "Classes:",
    CLASS_NAMES
)

print(
    "Data augmentation: ENABLED"
)

print("=" * 80)


# ============================================================
# 6. CHECK BASIC PATHS
# ============================================================

if not DATASET_ROOT.exists():

    raise FileNotFoundError(
        f"\nDataset root does not exist:\n{DATASET_ROOT}"
    )

if not FRUIT_DIR.exists():

    raise FileNotFoundError(
        f"\nBanana folder does not exist:\n{FRUIT_DIR}"
    )


# ============================================================
# 7. HELPER: FIND IMAGE FILES
# ============================================================

def image_files(folder):

    if not folder.exists():

        return []

    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )


# ============================================================
# 8. FIND DATASET STRUCTURE
# ============================================================

def detect_dataset_layout():

    # --------------------------------------------------------
    # Layout A:
    #
    # Banana/
    #   Ripe/Train
    #   Ripe/Test
    #   Unripe/Train
    #   Unripe/Test
    #   Rotten/Train
    #   Rotten/Test
    # --------------------------------------------------------

    layout_a = True

    for class_name in CLASS_NAMES:

        train_folder = (
            FRUIT_DIR
            / class_name
            / "Train"
        )

        test_folder = (
            FRUIT_DIR
            / class_name
            / "Test"
        )

        if (
            not train_folder.exists()
            or not test_folder.exists()
        ):

            layout_a = False
            break

    if layout_a:

        return "CLASS_THEN_SPLIT"


    # --------------------------------------------------------
    # Layout B:
    #
    # Banana/
    #   Train/Ripe
    #   Train/Unripe
    #   Train/Rotten
    #   Test/Ripe
    #   Test/Unripe
    #   Test/Rotten
    # --------------------------------------------------------

    layout_b = True

    train_root = FRUIT_DIR / "Train"
    test_root = FRUIT_DIR / "Test"

    for class_name in CLASS_NAMES:

        train_folder = (
            train_root
            / class_name
        )

        test_folder = (
            test_root
            / class_name
        )

        if (
            not train_folder.exists()
            or not test_folder.exists()
        ):

            layout_b = False
            break

    if layout_b:

        return "SPLIT_THEN_CLASS"


    # --------------------------------------------------------
    # Layout C:
    #
    # Banana/
    #   Ripe/
    #   Unripe/
    #   Rotten/
    #
    # Images directly inside classes.
    # We will create 80:20.
    # --------------------------------------------------------

    layout_c = True

    for class_name in CLASS_NAMES:

        class_folder = (
            FRUIT_DIR
            / class_name
        )

        if not class_folder.exists():

            layout_c = False
            break

        files = image_files(
            class_folder
        )

        if len(files) == 0:

            layout_c = False
            break

    if layout_c:

        return "CLASS_ONLY"


    return "UNKNOWN"


# ============================================================
# 9. DISPLAY DETECTED STRUCTURE
# ============================================================

layout = detect_dataset_layout()

print()
print("=" * 80)
print("DATASET STRUCTURE CHECK")
print("=" * 80)

print(
    "Detected layout:",
    layout
)

if layout == "UNKNOWN":

    print()
    print(
        "Could not automatically detect the expected structure."
    )

    print()
    print(
        "Banana folder contents:"
    )

    for item in sorted(
        FRUIT_DIR.iterdir()
    ):

        print(
            "   ",
            item
        )

    raise RuntimeError(
        "\nPlease check the Banana folder structure."
    )


# ============================================================
# 10. COLLECT PRE-SPLIT DATA
# ============================================================

def collect_class_then_split():

    train_paths = []
    train_labels = []

    test_paths = []
    test_labels = []

    print()
    print("=" * 80)
    print("READING CLASS / TRAIN / TEST STRUCTURE")
    print("=" * 80)

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        train_folder = (
            FRUIT_DIR
            / class_name
            / "Train"
        )

        test_folder = (
            FRUIT_DIR
            / class_name
            / "Test"
        )

        train_files = image_files(
            train_folder
        )

        test_files = image_files(
            test_folder
        )

        print()
        print(
            f"{class_name}:"
        )

        print(
            "   Train:",
            len(train_files)
        )

        print(
            "   Test :",
            len(test_files)
        )

        for path in train_files:

            train_paths.append(
                str(path)
            )

            train_labels.append(
                class_index
            )

        for path in test_files:

            test_paths.append(
                str(path)
            )

            test_labels.append(
                class_index
            )

    return (
        np.array(train_paths),
        np.array(
            train_labels,
            dtype=np.int64
        ),
        np.array(test_paths),
        np.array(
            test_labels,
            dtype=np.int64
        )
    )


# ============================================================
# 11. COLLECT SPLIT / CLASS STRUCTURE
# ============================================================

def collect_split_then_class():

    train_paths = []
    train_labels = []

    test_paths = []
    test_labels = []

    train_root = (
        FRUIT_DIR
        / "Train"
    )

    test_root = (
        FRUIT_DIR
        / "Test"
    )

    print()
    print("=" * 80)
    print("READING TRAIN / TEST / CLASS STRUCTURE")
    print("=" * 80)

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        train_folder = (
            train_root
            / class_name
        )

        test_folder = (
            test_root
            / class_name
        )

        train_files = image_files(
            train_folder
        )

        test_files = image_files(
            test_folder
        )

        print()
        print(
            f"{class_name}:"
        )

        print(
            "   Train:",
            len(train_files)
        )

        print(
            "   Test :",
            len(test_files)
        )

        for path in train_files:

            train_paths.append(
                str(path)
            )

            train_labels.append(
                class_index
            )

        for path in test_files:

            test_paths.append(
                str(path)
            )

            test_labels.append(
                class_index
            )

    return (
        np.array(train_paths),
        np.array(
            train_labels,
            dtype=np.int64
        ),
        np.array(test_paths),
        np.array(
            test_labels,
            dtype=np.int64
        )
    )


# ============================================================
# 12. COLLECT CLASS-ONLY DATA AND CREATE 80:20
# ============================================================

def collect_class_only_and_split():

    all_paths = []
    all_labels = []

    print()
    print("=" * 80)
    print("READING CLASS-ONLY STRUCTURE")
    print("=" * 80)

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_folder = (
            FRUIT_DIR
            / class_name
        )

        files = image_files(
            class_folder
        )

        print(
            f"{class_name:<10}:",
            len(files)
        )

        for path in files:

            all_paths.append(
                str(path)
            )

            all_labels.append(
                class_index
            )

    all_paths = np.array(
        all_paths
    )

    all_labels = np.array(
        all_labels,
        dtype=np.int64
    )

    train_paths, test_paths, train_labels, test_labels = (
        train_test_split(

            all_paths,

            all_labels,

            test_size=0.20,

            random_state=RANDOM_STATE,

            stratify=all_labels
        )
    )

    print()
    print(
        "Created 80:20 split automatically."
    )

    return (
        train_paths,
        train_labels,
        test_paths,
        test_labels
    )


# ============================================================
# 13. LOAD DATA ACCORDING TO STRUCTURE
# ============================================================

if layout == "CLASS_THEN_SPLIT":

    original_train_paths, original_train_labels, \
        test_paths, test_labels = (
            collect_class_then_split()
        )

elif layout == "SPLIT_THEN_CLASS":

    original_train_paths, original_train_labels, \
        test_paths, test_labels = (
            collect_split_then_class()
        )

elif layout == "CLASS_ONLY":

    original_train_paths, original_train_labels, \
        test_paths, test_labels = (
            collect_class_only_and_split()
        )


# ============================================================
# 14. CHECK TRAIN / TEST COUNTS
# ============================================================

print()
print("=" * 80)
print("ORIGINAL TRAIN / TEST")
print("=" * 80)

print(
    "Original training images:",
    len(original_train_paths)
)

print(
    "Testing images:",
    len(test_paths)
)

print(
    "Original total:",
    len(original_train_paths)
    +
    len(test_paths)
)


# ============================================================
# 15. CREATE VALIDATION SET
# ============================================================

(
    train_paths,
    val_paths,
    train_labels,
    val_labels
) = train_test_split(

    original_train_paths,

    original_train_labels,

    test_size=VALIDATION_FROM_TRAIN,

    random_state=RANDOM_STATE,

    stratify=original_train_labels
)


# ============================================================
# 16. FINAL SPLIT
# ============================================================

print()
print("=" * 80)
print("FINAL EXPERIMENT SPLIT")
print("=" * 80)

print(
    "Training   :",
    len(train_paths)
)

print(
    "Validation :",
    len(val_paths)
)

print(
    "Testing    :",
    len(test_paths)
)

print(
    "Total      :",
    len(train_paths)
    +
    len(val_paths)
    +
    len(test_paths)
)


# ============================================================
# 17. CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("FINAL CLASS DISTRIBUTION")
print("=" * 80)

for class_index, class_name in enumerate(
    CLASS_NAMES
):

    train_count = np.sum(
        train_labels == class_index
    )

    val_count = np.sum(
        val_labels == class_index
    )

    test_count = np.sum(
        test_labels == class_index
    )

    print(
        f"{class_name:<10}"
        f"Train={train_count:<6}"
        f"Val={val_count:<6}"
        f"Test={test_count:<6}"
    )


# ============================================================
# 18. LOAD IMAGE DATA
# ============================================================

def load_image_dataset(
    paths,
    labels
):

    images = []
    valid_labels = []

    print()
    print(
        "Loading",
        len(paths),
        "images..."
    )

    for index, path in enumerate(paths):

        image = cv2.imread(
            path
        )

        if image is None:

            print(
                "WARNING: Could not load:",
                path
            )

            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (
                IMG_WIDTH,
                IMG_HEIGHT
            )
        )

        image = image.astype(
            np.float32
        )

        images.append(
            image
        )

        valid_labels.append(
            labels[index]
        )

    X = np.array(
        images,
        dtype=np.float32
    )

    y = np.array(
        valid_labels,
        dtype=np.int64
    )

    return X, y


# ============================================================
# 19. LOAD TRAIN / VAL / TEST
# ============================================================

X_train, y_train = load_image_dataset(
    train_paths,
    train_labels
)

X_val, y_val = load_image_dataset(
    val_paths,
    val_labels
)

X_test, y_test = load_image_dataset(
    test_paths,
    test_labels
)


# ============================================================
# 20. NORMALIZATION
# ============================================================

X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0


# ============================================================
# 21. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(

    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        layers.RandomContrast(
            0.10
        )

    ],

    name="RCCNet_Data_Augmentation"
)


print()
print("=" * 80)
print("DATA AUGMENTATION SETTINGS")
print("=" * 80)

print(
    "Horizontal flip : ENABLED"
)

print(
    "Rotation        : ENABLED"
)

print(
    "Zoom            : ENABLED"
)

print(
    "Translation     : ENABLED"
)

print(
    "Contrast        : ENABLED"
)

print()
print(
    "Training only:"
)

print(
    "Validation: NOT augmented"
)

print(
    "Test: NOT augmented"
)


# ============================================================
# 22. BUILD RCCNET
# ============================================================

def build_rccnet():

    inputs = Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        ),
        name="RCCNet_Input"
    )

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    x = data_augmentation(
        inputs
    )

    # --------------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------------

    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        name="block1_conv1"
    )(x)

    x = BatchNormalization(
        name="block1_bn1"
    )(x)

    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        name="block1_conv2"
    )(x)

    x = BatchNormalization(
        name="block1_bn2"
    )(x)

    x = MaxPooling2D(
        (2, 2),
        name="block1_pool"
    )(x)

    # --------------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------------

    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        name="block2_conv1"
    )(x)

    x = BatchNormalization(
        name="block2_bn1"
    )(x)

    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        name="block2_conv2"
    )(x)

    x = BatchNormalization(
        name="block2_bn2"
    )(x)

    x = MaxPooling2D(
        (2, 2),
        name="block2_pool"
    )(x)

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    x = Flatten(
        name="flatten"
    )(x)

    x = Dense(
        256,
        activation="relu",
        name="dense1"
    )(x)

    x = BatchNormalization(
        name="dense_bn"
    )(x)

    x = Dropout(
        0.5,
        name="dropout"
    )(x)

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="RCCNet_Banana_DA"
    )

    return model


# ============================================================
# 23. BUILD MODEL
# ============================================================

print()
print("=" * 80)
print("BUILDING RCCNET")
print("=" * 80)

model = build_rccnet()

model.summary()


# ============================================================
# 24. COMPILE
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=1e-3
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# 25. CALLBACK PATHS
# ============================================================

best_model_path = (
    RESULTS_DIR /
    "RCCNet_DA_Best.keras"
)

final_model_path = (
    RESULTS_DIR /
    "RCCNet_DA_Final.keras"
)

history_csv_path = (
    RESULTS_DIR /
    "RCCNet_DA_History.csv"
)


# ============================================================
# 26. CALLBACKS
# ============================================================

callbacks = [

    ModelCheckpoint(

        filepath=best_model_path,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=3,

        min_lr=1e-6,

        verbose=1
    ),

    EarlyStopping(

        monitor="val_loss",

        patience=7,

        restore_best_weights=True,

        verbose=1
    ),

    CSVLogger(

        filename=history_csv_path,

        append=False
    )
]


# ============================================================
# 27. TRAIN
# ============================================================

print()
print("=" * 80)
print("STARTING RCCNet DA TRAINING")
print("=" * 80)

print(
    "Maximum epochs:",
    EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

start_time = time.time()


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=callbacks,

    verbose=1
)


training_time = (
    time.time()
    -
    start_time
)


print()
print(
    "Training time:",
    f"{training_time / 60:.2f} minutes"
)


# ============================================================
# 28. LOAD BEST MODEL
# ============================================================

model = tf.keras.models.load_model(
    best_model_path
)

print()
print(
    "Best model loaded."
)


# ============================================================
# 29. VALIDATION RESULT
# ============================================================

print()
print("=" * 80)
print("VALIDATION RESULT")
print("=" * 80)

val_loss, val_accuracy = model.evaluate(

    X_val,

    y_val,

    verbose=1
)

print()

print(
    f"Validation Loss: "
    f"{val_loss:.6f}"
)

print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.4f}%"
)


# ============================================================
# 30. TEST RESULT
# ============================================================

print()
print("=" * 80)
print("FINAL TEST RESULT")
print("=" * 80)

test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=1
)

print()

print(
    f"Test Loss: "
    f"{test_loss:.6f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.4f}%"
)


# ============================================================
# 31. TEST PREDICTIONS
# ============================================================

print()
print("=" * 80)
print("GENERATING TEST PREDICTIONS")
print("=" * 80)

probabilities = model.predict(

    X_test,

    batch_size=BATCH_SIZE,

    verbose=1
)


y_pred = np.argmax(
    probabilities,
    axis=1
)

y_true = y_test


# ============================================================
# 32. FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print()
print("=" * 80)
print("FINAL TEST METRICS")
print("=" * 80)

print(
    f"Accuracy  : "
    f"{accuracy * 100:.4f}%"
)

print(
    f"Precision : "
    f"{precision * 100:.4f}%"
)

print(
    f"Recall    : "
    f"{recall * 100:.4f}%"
)

print(
    f"F1 Score  : "
    f"{f1 * 100:.4f}%"
)


# ============================================================
# 33. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=np.arange(
        NUM_CLASSES
    )
)

cm_accuracy = (
    np.trace(cm)
    /
    np.sum(cm)
)


print()
print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm)

print()
print(
    f"Confusion Matrix Accuracy: "
    f"{cm_accuracy * 100:.4f}%"
)


# ============================================================
# 34. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    labels=np.arange(
        NUM_CLASSES
    ),

    target_names=CLASS_NAMES,

    digits=4,

    zero_division=0
)


print()
print("=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(report)


with open(

    RESULTS_DIR /
    "RCCNet_DA_Classification_Report.txt",

    "w"

) as file:

    file.write(report)


# ============================================================
# 35. CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(9, 8)
)

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

plt.title(
    "Banana Ripeness - RCCNet\n"
    "With Data Augmentation",
    fontsize=16
)

plt.colorbar()

tick_marks = np.arange(
    NUM_CLASSES
)

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

threshold = cm.max() / 2.0

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(

            j,

            i,

            str(
                cm[i, j]
            ),

            ha="center",

            va="center",

            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),

            fontsize=12
        )


plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()

plt.savefig(

    RESULTS_DIR /
    "RCCNet_DA_Confusion_Matrix.png",

    dpi=300,

    bbox_inches="tight"
)

plt.close()


# ============================================================
# 36. ACCURACY GRAPH
# ============================================================

epoch_numbers = np.arange(

    1,

    len(
        history.history[
            "accuracy"
        ]
    ) + 1

)


plt.figure(
    figsize=(12, 7)
)

plt.plot(

    epoch_numbers,

    history.history[
        "accuracy"
    ],

    marker="o",

    label="Training Accuracy"
)

plt.plot(

    epoch_numbers,

    history.history[
        "val_accuracy"
    ],

    marker="o",

    label="Validation Accuracy"
)

plt.title(
    "Banana Ripeness - RCCNet Accuracy\n"
    "With Data Augmentation",
    fontsize=16
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(

    RESULTS_DIR /
    "RCCNet_DA_Accuracy.png",

    dpi=300,

    bbox_inches="tight"
)

plt.close()


# ============================================================
# 37. LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(

    epoch_numbers,

    history.history[
        "loss"
    ],

    marker="o",

    label="Training Loss"
)

plt.plot(

    epoch_numbers,

    history.history[
        "val_loss"
    ],

    marker="o",

    label="Validation Loss"
)

plt.title(
    "Banana Ripeness - RCCNet Loss\n"
    "With Data Augmentation",
    fontsize=16
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(

    RESULTS_DIR /
    "RCCNet_DA_Loss.png",

    dpi=300,

    bbox_inches="tight"
)

plt.close()


# ============================================================
# 38. BEST EPOCH RESULTS
# ============================================================

best_train_accuracy = max(
    history.history[
        "accuracy"
    ]
)

best_val_accuracy = max(
    history.history[
        "val_accuracy"
    ]
)

best_train_loss = min(
    history.history[
        "loss"
    ]
)

best_val_loss = min(
    history.history[
        "val_loss"
    ]
)

best_val_epoch = (
    int(
        np.argmax(
            history.history[
                "val_accuracy"
            ]
        )
    )
    + 1
)


# ============================================================
# 39. COMPLETE CSV
# ============================================================

history_df = pd.DataFrame(

    {

        "epoch":
            epoch_numbers,

        "accuracy":
            history.history[
                "accuracy"
            ],

        "val_accuracy":
            history.history[
                "val_accuracy"
            ],

        "loss":
            history.history[
                "loss"
            ],

        "val_loss":
            history.history[
                "val_loss"
            ]
    }
)


complete_csv_path = (
    RESULTS_DIR /
    "RCCNet_DA_Complete_History.csv"
)


history_df.to_csv(

    complete_csv_path,

    index=False
)


# ============================================================
# 40. FINAL METRICS FILE
# ============================================================

metrics_path = (
    RESULTS_DIR /
    "RCCNet_DA_Final_Metrics.txt"
)


with open(
    metrics_path,
    "w"
) as file:

    file.write(
        "BANANA RIPENESS - RCCNet\n"
    )

    file.write(
        "WITH DATA AUGMENTATION\n"
    )

    file.write(
        "========================================\n\n"
    )

    file.write(
        f"TensorFlow Version: "
        f"{tf.__version__}\n"
    )

    file.write(
        f"GPU Detected: "
        f"{bool(GPUS)}\n\n"
    )

    file.write(
        f"Dataset Root: "
        f"{DATASET_ROOT}\n"
    )

    file.write(
        f"Fruit: "
        f"{FRUIT_NAME}\n"
    )

    file.write(
        f"Technology: RCCNet\n"
    )

    file.write(
        f"Experiment: DA\n\n"
    )

    file.write(
        f"Image Size: "
        f"{IMG_WIDTH}x{IMG_HEIGHT}\n"
    )

    file.write(
        f"Batch Size: "
        f"{BATCH_SIZE}\n"
    )

    file.write(
        f"Configured Epochs: "
        f"{EPOCHS}\n\n"
    )

    file.write(
        f"Original Train Images: "
        f"{len(original_train_paths)}\n"
    )

    file.write(
        f"Training Images: "
        f"{len(train_paths)}\n"
    )

    file.write(
        f"Validation Images: "
        f"{len(val_paths)}\n"
    )

    file.write(
        f"Test Images: "
        f"{len(test_paths)}\n"
    )

    file.write(
        f"Total Images Used: "
        f"{len(train_paths) + len(val_paths) + len(test_paths)}\n\n"
    )

    file.write(
        f"Validation Loss: "
        f"{val_loss:.6f}\n"
    )

    file.write(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.4f}%\n\n"
    )

    file.write(
        f"Test Loss: "
        f"{test_loss:.6f}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.4f}%\n"
    )

    file.write(
        f"Precision: "
        f"{precision * 100:.4f}%\n"
    )

    file.write(
        f"Recall: "
        f"{recall * 100:.4f}%\n"
    )

    file.write(
        f"F1 Score: "
        f"{f1 * 100:.4f}%\n"
    )

    file.write(
        f"Confusion Matrix Accuracy: "
        f"{cm_accuracy * 100:.4f}%\n\n"
    )

    file.write(
        f"Best Training Accuracy: "
        f"{best_train_accuracy * 100:.4f}%\n"
    )

    file.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.4f}%\n"
    )

    file.write(
        f"Best Training Loss: "
        f"{best_train_loss:.6f}\n"
    )

    file.write(
        f"Best Validation Loss: "
        f"{best_val_loss:.6f}\n"
    )

    file.write(
        f"Best Validation Epoch: "
        f"{best_val_epoch}\n\n"
    )

    file.write(
        "Data Augmentation:\n"
    )

    file.write(
        "Horizontal Flip\n"
    )

    file.write(
        "Random Rotation = 0.08\n"
    )

    file.write(
        "Random Zoom = 0.10\n"
    )

    file.write(
        "Random Translation = 0.05\n"
    )

    file.write(
        "Random Contrast = 0.10\n"
    )


# ============================================================
# 41. RUN INFORMATION
# ============================================================

run_info_path = (
    RESULTS_DIR /
    "RCCNet_DA_Run_Information.txt"
)


with open(
    run_info_path,
    "w"
) as file:

    file.write(
        "RCCNet Phase 2 Experiment\n\n"
    )

    file.write(
        f"Fruit: {FRUIT_NAME}\n"
    )

    file.write(
        "Technology: RCCNet\n"
    )

    file.write(
        "Mode: DA\n"
    )

    file.write(
        "Data Augmentation: TRUE\n\n"
    )

    file.write(
        f"Dataset Root:\n"
        f"{DATASET_ROOT}\n\n"
    )

    file.write(
        f"Fruit Folder:\n"
        f"{FRUIT_DIR}\n\n"
    )

    file.write(
        f"Results Folder:\n"
        f"{RESULTS_DIR}\n\n"
    )

    file.write(
        f"Original Train Images: "
        f"{len(original_train_paths)}\n"
    )

    file.write(
        f"Training Images: "
        f"{len(train_paths)}\n"
    )

    file.write(
        f"Validation Images: "
        f"{len(val_paths)}\n"
    )

    file.write(
        f"Test Images: "
        f"{len(test_paths)}\n"
    )

    file.write(
        f"Total Images Used: "
        f"{len(train_paths) + len(val_paths) + len(test_paths)}\n"
    )


# ============================================================
# 42. SAVE FINAL MODEL
# ============================================================

model.save(
    final_model_path
)


# ============================================================
# 43. FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("RCCNet DA EXPERIMENT COMPLETE")
print("=" * 80)

print()
print(
    "Fruit:",
    FRUIT_NAME
)

print(
    "Technology:",
    "RCCNet"
)

print(
    "Mode:",
    "DA"
)

print()

print(
    "Training images:",
    len(train_paths)
)

print(
    "Validation images:",
    len(val_paths)
)

print(
    "Testing images:",
    len(test_paths)
)

print(
    "Total images:",
    len(train_paths)
    +
    len(val_paths)
    +
    len(test_paths)
)

print()

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: "
    f"{test_loss:.6f}"
)

print(
    f"Precision: "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall: "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score: "
    f"{f1 * 100:.2f}%"
)

print()

print(
    "Results saved to:"
)

print(
    RESULTS_DIR.resolve()
)

print()

print("Output files:")

print(
    "1. RCCNet_DA_Best.keras"
)

print(
    "2. RCCNet_DA_Final.keras"
)

print(
    "3. RCCNet_DA_History.csv"
)

print(
    "4. RCCNet_DA_Complete_History.csv"
)

print(
    "5. RCCNet_DA_Accuracy.png"
)

print(
    "6. RCCNet_DA_Loss.png"
)

print(
    "7. RCCNet_DA_Confusion_Matrix.png"
)

print(
    "8. RCCNet_DA_Classification_Report.txt"
)

print(
    "9. RCCNet_DA_Final_Metrics.txt"
)

print(
    "10. RCCNet_DA_Run_Information.txt"
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)