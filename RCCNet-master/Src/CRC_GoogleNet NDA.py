# ================================================================
# GOOGLENET - BANANA RIPENESS CLASSIFICATION
# WITHOUT DATA AUGMENTATION
# ================================================================

import os

# ---------------------------------------------------------------
# IMPORTANT:
# Set this BEFORE importing TensorFlow
# ---------------------------------------------------------------

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ================================================================
# IMPORTS
# ================================================================

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path

from tensorflow.keras import layers, Model

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger
)

from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ================================================================
# SETTINGS
# ================================================================

DATASET_PATH = r"C:\Users\E028.6\Downloads\Banana_Dataset"

IMG_HEIGHT = 128
IMG_WIDTH = 128

BATCH_SIZE = 32

EPOCHS = 30

NUM_CLASSES = 4

RANDOM_SEED = 42

CLASS_NAMES = [
    "Overripe",
    "Rippen",
    "Rotten",
    "Unripe"
]

# ================================================================
# DATA AUGMENTATION
# ================================================================

# IMPORTANT:
# This experiment DOES NOT use data augmentation.

DATA_AUGMENTATION = False

# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)

# ================================================================
# PRINT INFORMATION
# ================================================================

print("=" * 70)
print("BANANA RIPENESS - GOOGLENET")
print("WITHOUT DATA AUGMENTATION")
print("=" * 70)

print("TensorFlow:", tf.__version__)

print("Dataset:", DATASET_PATH)

print(
    "Image size:",
    IMG_WIDTH,
    "x",
    IMG_HEIGHT
)

print("Batch size:", BATCH_SIZE)

print("Epochs:", EPOCHS)

print("Classes:", CLASS_NAMES)

print(
    "Data augmentation:",
    DATA_AUGMENTATION
)

print()

# ================================================================
# CHECK DATASET
# ================================================================

dataset_path = Path(DATASET_PATH)

if not dataset_path.exists():

    raise FileNotFoundError(
        "\nDataset not found:\n"
        + DATASET_PATH
        + "\n\n"
        "Please check the dataset path."
    )

# ================================================================
# CHECK CLASS FOLDERS
# ================================================================

for class_name in CLASS_NAMES:

    class_folder = dataset_path / class_name

    if not class_folder.exists():

        raise FileNotFoundError(
            f"\nMissing class folder:\n"
            f"{class_folder}\n\n"
            f"Expected folders:\n"
            f"  Overripe\n"
            f"  Rippen\n"
            f"  Rotten\n"
            f"  Unripe"
        )

print(
    "Dataset folders found successfully."
)

# ================================================================
# COLLECT IMAGE PATHS
# ================================================================

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_paths = []

labels = []

print()
print("=" * 70)
print("COUNTING DATASET IMAGES")
print("=" * 70)

# ---------------------------------------------------------------
# Read each class folder
# ---------------------------------------------------------------

for class_index, class_name in enumerate(CLASS_NAMES):

    class_folder = dataset_path / class_name

    files = sorted(
        [
            p
            for p in class_folder.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower()
                in valid_extensions
            )
        ],
        key=lambda p: p.name.lower()
    )

    print(
        f"{class_name:10s}: "
        f"{len(files):5d} images"
    )

    for file_path in files:

        image_paths.append(
            str(file_path)
        )

        labels.append(
            class_index
        )

# ================================================================
# CONVERT TO NUMPY
# ================================================================

image_paths = np.array(
    image_paths
)

labels = np.array(
    labels,
    dtype=np.int64
)

print()

print(
    "Total images:",
    len(image_paths)
)

if len(image_paths) == 0:

    raise RuntimeError(
        "No images were found."
    )

# ================================================================
# TOTAL CLASS DISTRIBUTION
# ================================================================

print()
print("=" * 70)
print("TOTAL CLASS DISTRIBUTION")
print("=" * 70)

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(
        labels == i
    )

    print(
        f"{class_name:10s}: "
        f"{count:5d}"
    )

# ================================================================
# TRAIN / VALIDATION / TEST SPLIT
#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
#
# STRATIFIED SPLIT
# ================================================================

print()
print("=" * 70)
print("CREATING DATASET SPLIT")
print("=" * 70)

# ---------------------------------------------------------------
# First split:
#
# 70% training
# 30% temporary
# ---------------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(

    image_paths,

    labels,

    test_size=0.30,

    random_state=RANDOM_SEED,

    stratify=labels
)

# ---------------------------------------------------------------
# Second split:
#
# 15% validation
# 15% testing
# ---------------------------------------------------------------

X_val, X_test, y_val, y_test = train_test_split(

    X_temp,

    y_temp,

    test_size=0.50,

    random_state=RANDOM_SEED,

    stratify=y_temp
)

# ================================================================
# PRINT SPLIT
# ================================================================

print()

print("Dataset split:")

print(
    "Training   :",
    len(X_train)
)

print(
    "Validation :",
    len(X_val)
)

print(
    "Testing    :",
    len(X_test)
)

# ================================================================
# CLASS DISTRIBUTION AFTER SPLIT
# ================================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION AFTER SPLIT")
print("=" * 70)

for i, class_name in enumerate(CLASS_NAMES):

    train_count = np.sum(
        y_train == i
    )

    val_count = np.sum(
        y_val == i
    )

    test_count = np.sum(
        y_test == i
    )

    print(
        f"{class_name:10s} "
        f"Train={train_count:5d} "
        f"Val={val_count:5d} "
        f"Test={test_count:5d}"
    )

# ================================================================
# IMAGE LOADING FUNCTION
# ================================================================

def load_image(path, label):

    # ------------------------------------------------------------
    # Read image file
    # ------------------------------------------------------------

    image = tf.io.read_file(
        path
    )

    # ------------------------------------------------------------
    # Decode image
    # ------------------------------------------------------------

    image = tf.image.decode_image(

        image,

        channels=3,

        expand_animations=False
    )

    # ------------------------------------------------------------
    # Set image shape
    # ------------------------------------------------------------

    image.set_shape(
        [None, None, 3]
    )

    # ------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------

    image = tf.image.resize(

        image,

        [
            IMG_HEIGHT,
            IMG_WIDTH
        ]
    )

    # ------------------------------------------------------------
    # Convert to float
    # ------------------------------------------------------------

    image = tf.cast(
        image,
        tf.float32
    )

    # ------------------------------------------------------------
    # Normalize pixel values
    #
    # 0-255 -> 0-1
    # ------------------------------------------------------------

    image = image / 255.0

    return image, label

# ================================================================
# CREATE TF.DATA DATASET
# ================================================================

def create_dataset(
    paths,
    labels,
    training=False
):

    # ------------------------------------------------------------
    # Create TensorFlow dataset
    # ------------------------------------------------------------

    dataset = tf.data.Dataset.from_tensor_slices(

        (
            paths,
            labels
        )
    )

    # ------------------------------------------------------------
    # Shuffle ONLY training data
    # ------------------------------------------------------------

    if training:

        dataset = dataset.shuffle(

            buffer_size=len(paths),

            seed=RANDOM_SEED,

            reshuffle_each_iteration=True
        )

    # ------------------------------------------------------------
    # Load images
    # ------------------------------------------------------------

    dataset = dataset.map(

        load_image,

        num_parallel_calls=tf.data.AUTOTUNE
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # NO DATA AUGMENTATION HERE
    #
    # There is deliberately NO:
    # RandomFlip
    # RandomRotation
    # RandomZoom
    # RandomContrast
    # etc.
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------

    dataset = dataset.batch(
        BATCH_SIZE
    )

    # ------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset

# ================================================================
# CREATE DATASETS
# ================================================================

train_dataset = create_dataset(

    X_train,

    y_train,

    training=True
)

val_dataset = create_dataset(

    X_val,

    y_val,

    training=False
)

test_dataset = create_dataset(

    X_test,

    y_test,

    training=False
)

# ================================================================
# DATASET STATUS
# ================================================================

print()
print("=" * 70)
print("DATASETS CREATED")
print("=" * 70)

print(
    "Training dataset   : Ready"
)

print(
    "Validation dataset : Ready"
)

print(
    "Testing dataset    : Ready"
)

print(
    "Data augmentation  : DISABLED"
)

# ================================================================
# GOOGLENET / INCEPTION MODULE
# ================================================================

def inception_module(

    x,

    filters_1x1,

    filters_3x3_reduce,

    filters_3x3,

    filters_5x5_reduce,

    filters_5x5,

    filters_pool_proj

):

    # ============================================================
    # BRANCH 1
    # 1x1 convolution
    # ============================================================

    branch1 = layers.Conv2D(

        filters_1x1,

        (1, 1),

        padding="same",

        activation="relu"

    )(x)

    branch1 = layers.BatchNormalization()(

        branch1

    )

    # ============================================================
    # BRANCH 2
    # 1x1 -> 3x3
    # ============================================================

    branch2 = layers.Conv2D(

        filters_3x3_reduce,

        (1, 1),

        padding="same",

        activation="relu"

    )(x)

    branch2 = layers.BatchNormalization()(

        branch2

    )

    branch2 = layers.Conv2D(

        filters_3x3,

        (3, 3),

        padding="same",

        activation="relu"

    )(branch2)

    branch2 = layers.BatchNormalization()(

        branch2

    )

    # ============================================================
    # BRANCH 3
    # 1x1 -> 5x5
    # ============================================================

    branch3 = layers.Conv2D(

        filters_5x5_reduce,

        (1, 1),

        padding="same",

        activation="relu"

    )(x)

    branch3 = layers.BatchNormalization()(

        branch3

    )

    branch3 = layers.Conv2D(

        filters_5x5,

        (5, 5),

        padding="same",

        activation="relu"

    )(branch3)

    branch3 = layers.BatchNormalization()(

        branch3

    )

    # ============================================================
    # BRANCH 4
    # MAX POOL -> 1x1
    # ============================================================

    branch4 = layers.MaxPooling2D(

        (3, 3),

        strides=(1, 1),

        padding="same"

    )(x)

    branch4 = layers.Conv2D(

        filters_pool_proj,

        (1, 1),

        padding="same",

        activation="relu"

    )(branch4)

    branch4 = layers.BatchNormalization()(

        branch4

    )

    # ============================================================
    # CONCATENATE BRANCHES
    # ============================================================

    output = layers.Concatenate(

        axis=-1

    )(
        [
            branch1,
            branch2,
            branch3,
            branch4
        ]
    )

    return output

# ================================================================
# BUILD GOOGLENET
# ================================================================

def build_googlenet():

    # ============================================================
    # INPUT
    # ============================================================

    inputs = layers.Input(

        shape=(

            IMG_HEIGHT,

            IMG_WIDTH,

            3

        )

    )

    # ============================================================
    # INITIAL CONVOLUTION
    # ============================================================

    x = layers.Conv2D(

        64,

        (7, 7),

        strides=(2, 2),

        padding="same",

        activation="relu"

    )(inputs)

    x = layers.BatchNormalization()(

        x

    )

    x = layers.MaxPooling2D(

        (3, 3),

        strides=(2, 2),

        padding="same"

    )(x)

    # ============================================================
    # CONVOLUTIONAL SECTION
    # ============================================================

    x = layers.Conv2D(

        64,

        (1, 1),

        padding="same",

        activation="relu"

    )(x)

    x = layers.BatchNormalization()(

        x

    )

    x = layers.Conv2D(

        192,

        (3, 3),

        padding="same",

        activation="relu"

    )(x)

    x = layers.BatchNormalization()(

        x

    )

    x = layers.MaxPooling2D(

        (3, 3),

        strides=(2, 2),

        padding="same"

    )(x)

    # ============================================================
    # INCEPTION 3a
    # ============================================================

    x = inception_module(

        x,

        64,

        96,

        128,

        16,

        32,

        32

    )

    # ============================================================
    # INCEPTION 3b
    # ============================================================

    x = inception_module(

        x,

        128,

        128,

        192,

        32,

        96,

        64

    )

    x = layers.MaxPooling2D(

        (3, 3),

        strides=(2, 2),

        padding="same"

    )(x)

    # ============================================================
    # INCEPTION 4a
    # ============================================================

    x = inception_module(

        x,

        192,

        96,

        208,

        16,

        48,

        64

    )

    # ============================================================
    # INCEPTION 4b
    # ============================================================

    x = inception_module(

        x,

        160,

        112,

        224,

        24,

        64,

        64

    )

    # ============================================================
    # INCEPTION 4c
    # ============================================================

    x = inception_module(

        x,

        128,

        128,

        256,

        24,

        64,

        64

    )

    # ============================================================
    # INCEPTION 4d
    # ============================================================

    x = inception_module(

        x,

        112,

        144,

        288,

        32,

        64,

        64

    )

    # ============================================================
    # INCEPTION 4e
    # ============================================================

    x = inception_module(

        x,

        256,

        160,

        320,

        32,

        128,

        128

    )

    x = layers.MaxPooling2D(

        (3, 3),

        strides=(2, 2),

        padding="same"

    )(x)

    # ============================================================
    # INCEPTION 5a
    # ============================================================

    x = inception_module(

        x,

        256,

        160,

        320,

        32,

        128,

        128

    )

    # ============================================================
    # INCEPTION 5b
    # ============================================================

    x = inception_module(

        x,

        384,

        192,

        384,

        48,

        128,

        128

    )

    # ============================================================
    # CLASSIFICATION HEAD
    # ============================================================

    x = layers.GlobalAveragePooling2D()(

        x

    )

    x = layers.Dropout(

        0.40

    )(x)

    outputs = layers.Dense(

        NUM_CLASSES,

        activation="softmax",

        name="banana_classification"

    )(x)

    # ============================================================
    # CREATE MODEL
    # ============================================================

    model = Model(

        inputs=inputs,

        outputs=outputs,

        name="GoogleNet_Banana_NDA"

    )

    return model

# ================================================================
# CREATE MODEL
# ================================================================

model = build_googlenet()

# ================================================================
# MODEL SUMMARY
#
# line_length=120 FIXES:
# ValueError: Insufficient console width to print summary.
# ================================================================

print()
print("=" * 70)
print("GOOGLENET MODEL")
print("=" * 70)

model.summary(
    line_length=120
)

# ================================================================
# COMPILE MODEL
# ================================================================

model.compile(

    optimizer=Adam(

        learning_rate=0.001

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)

# ================================================================
# CALLBACKS
# ================================================================

callbacks = [

    # ------------------------------------------------------------
    # Reduce learning rate when validation loss stops improving
    # ------------------------------------------------------------

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=3,

        min_lr=1e-6,

        verbose=1

    ),

    # ------------------------------------------------------------
    # Save best model based on validation accuracy
    # ------------------------------------------------------------

    ModelCheckpoint(

        "GoogLeNet_NDA_best.keras",

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1

    ),

    # ------------------------------------------------------------
    # Stop training if validation loss does not improve
    # ------------------------------------------------------------

    EarlyStopping(

        monitor="val_loss",

        patience=7,

        restore_best_weights=True,

        verbose=1

    ),

    # ------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------

    CSVLogger(

        "GoogLeNet_NDA_history.csv",

        append=False

    )

]

# ================================================================
# START TRAINING
# ================================================================

print()
print("=" * 70)
print("STARTING GOOGLENET TRAINING")
print("WITHOUT DATA AUGMENTATION")
print("=" * 70)

print(
    "Data augmentation: DISABLED"
)

print(
    "Training images:",
    len(X_train)
)

print(
    "Validation images:",
    len(X_val)
)

print(
    "Testing images:",
    len(X_test)
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Maximum epochs:",
    EPOCHS
)

print()

start_time = time.time()

# ================================================================
# TRAIN
# ================================================================

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1

)

# ================================================================
# TRAINING TIME
# ================================================================

training_time = (
    time.time()
    -
    start_time
)

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Training time: "
    f"{training_time / 60:.2f} minutes"
)

# ================================================================
# LOAD BEST MODEL
# ================================================================

print()
print("=" * 70)
print("LOADING BEST GOOGLENET MODEL")
print("=" * 70)

model = tf.keras.models.load_model(

    "GoogLeNet_NDA_best.keras"

)

print(
    "Best model loaded successfully."
)

# ================================================================
# FINAL TEST EVALUATION
# ================================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(

    test_dataset,

    verbose=1

)

print()
print("FINAL TEST RESULTS")
print("-" * 50)

print(
    f"Test Loss     : "
    f"{test_loss:.6f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy:.6f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

# ================================================================
# GENERATE PREDICTIONS
# ================================================================

print()
print("=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

y_probability = model.predict(

    test_dataset,

    verbose=1

)

# ================================================================
# CONVERT PROBABILITIES TO CLASS PREDICTIONS
# ================================================================

y_pred = np.argmax(

    y_probability,

    axis=1

)

# ================================================================
# VERIFY PREDICTION COUNT
# ================================================================

print()

print(
    "Actual test labels:",
    len(y_test)
)

print(
    "Predicted labels:",
    len(y_pred)
)

if len(y_test) != len(y_pred):

    raise ValueError(

        "Number of predictions does not "
        "match number of test labels."

    )

# ================================================================
# ACCURACY USING SKLEARN
# ================================================================

prediction_accuracy = accuracy_score(

    y_test,

    y_pred

)

print()
print(
    "Accuracy from predictions:"
)

print(
    f"{prediction_accuracy * 100:.2f}%"
)

# ================================================================
# CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(

    y_test,

    y_pred,

    labels=np.arange(NUM_CLASSES)

)

# ================================================================
# PRINT CONFUSION MATRIX
# ================================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(cm)

# ================================================================
# CONFUSION MATRIX ACCURACY
# ================================================================

cm_accuracy = (

    np.trace(cm)
    /
    np.sum(cm)

)

print()

print(
    "Accuracy calculated from "
    "confusion matrix:"
)

print(
    f"{cm_accuracy * 100:.2f}%"
)

# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(

    y_test,

    y_pred,

    labels=np.arange(NUM_CLASSES),

    target_names=CLASS_NAMES,

    digits=4,

    zero_division=0

)

print()

print(report)

# ================================================================
# CONFUSION MATRIX PLOT
# ================================================================

plt.figure(

    figsize=(9, 8)

)

plt.imshow(

    cm,

    interpolation="nearest",

    cmap="Blues"

)

plt.title(

    "Banana Ripeness - GoogleNet Confusion Matrix",

    fontsize=15

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

plt.xlabel(

    "Predicted Class"

)

plt.ylabel(

    "Actual Class"

)

# ================================================================
# WRITE NUMBERS INSIDE CONFUSION MATRIX
# ================================================================

threshold = cm.max() / 2.0

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(

            j,

            i,

            format(
                cm[i, j],
                "d"
            ),

            ha="center",

            va="center",

            color=(

                "white"

                if cm[i, j] > threshold

                else "black"

            ),

            fontsize=13

        )

plt.tight_layout()

plt.savefig(

    "GoogLeNet_NDA_Confusion_Matrix.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ================================================================
# ACCURACY GRAPH
# ================================================================

plt.figure(

    figsize=(10, 6)

)

plt.plot(

    history.history["accuracy"],

    marker="o",

    label="Training Accuracy"

)

plt.plot(

    history.history["val_accuracy"],

    marker="o",

    label="Validation Accuracy"

)

plt.title(

    "Banana Ripeness - GoogleNet Accuracy"

)

plt.xlabel(

    "Epoch"

)

plt.ylabel(

    "Accuracy"

)

plt.grid(

    True,

    alpha=0.3

)

plt.legend()

plt.tight_layout()

plt.savefig(

    "GoogLeNet_NDA_Accuracy.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ================================================================
# LOSS GRAPH
# ================================================================

plt.figure(

    figsize=(10, 6)

)

plt.plot(

    history.history["loss"],

    marker="o",

    label="Training Loss"

)

plt.plot(

    history.history["val_loss"],

    marker="o",

    label="Validation Loss"

)

plt.title(

    "Banana Ripeness - GoogleNet Loss"

)

plt.xlabel(

    "Epoch"

)

plt.ylabel(

    "Loss"

)

plt.grid(

    True,

    alpha=0.3

)

plt.legend()

plt.tight_layout()

plt.savefig(

    "GoogLeNet_NDA_Loss.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ================================================================
# FIND BEST VALIDATION ACCURACY
# ================================================================

best_val_accuracy = max(

    history.history["val_accuracy"]

)

best_val_epoch = (

    np.argmax(
        history.history["val_accuracy"]
    )
    + 1

)

# ================================================================
# FIND BEST VALIDATION LOSS
# ================================================================

best_val_loss = min(

    history.history["val_loss"]

)

best_val_loss_epoch = (

    np.argmin(
        history.history["val_loss"]
    )
    + 1

)

# ================================================================
# LAST EPOCH RESULTS
# ================================================================

last_train_accuracy = (

    history.history["accuracy"][-1]

)

last_val_accuracy = (

    history.history["val_accuracy"][-1]

)

last_train_loss = (

    history.history["loss"][-1]

)

last_val_loss = (

    history.history["val_loss"][-1]

)

# ================================================================
# SAVE FINAL RESULTS
# ================================================================

with open(

    "GoogLeNet_NDA_final_results.txt",

    "w"

) as f:

    f.write(
        "Banana Ripeness - GoogleNet\n"
    )

    f.write(
        "WITHOUT DATA AUGMENTATION\n"
    )

    f.write(
        "========================================\n\n"
    )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    f.write(
        f"Dataset: {DATASET_PATH}\n"
    )

    f.write(
        f"Image Size: "
        f"{IMG_WIDTH} x {IMG_HEIGHT}\n"
    )

    f.write(
        f"Number of Classes: "
        f"{NUM_CLASSES}\n"
    )

    f.write(
        f"Batch Size: "
        f"{BATCH_SIZE}\n"
    )

    f.write(
        f"Maximum Epochs: "
        f"{EPOCHS}\n"
    )

    f.write(
        "Data Augmentation: DISABLED\n\n"
    )

    # ------------------------------------------------------------
    # Dataset split
    # ------------------------------------------------------------

    f.write(
        "Dataset Split\n"
    )

    f.write(
        "----------------------------------------\n"
    )

    f.write(
        f"Training: "
        f"{len(X_train)}\n"
    )

    f.write(
        f"Validation: "
        f"{len(X_val)}\n"
    )

    f.write(
        f"Testing: "
        f"{len(X_test)}\n\n"
    )

    # ------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------

    f.write(
        "Final Test Results\n"
    )

    f.write(
        "----------------------------------------\n"
    )

    f.write(
        f"Test Loss: "
        f"{test_loss:.6f}\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Prediction Accuracy: "
        f"{prediction_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Confusion Matrix Accuracy: "
        f"{cm_accuracy * 100:.2f}%\n\n"
    )

    # ------------------------------------------------------------
    # Best validation results
    # ------------------------------------------------------------

    f.write(
        "Best Validation Results\n"
    )

    f.write(
        "----------------------------------------\n"
    )

    f.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Best Validation Accuracy Epoch: "
        f"{best_val_epoch}\n"
    )

    f.write(
        f"Best Validation Loss: "
        f"{best_val_loss:.6f}\n"
    )

    f.write(
        f"Best Validation Loss Epoch: "
        f"{best_val_loss_epoch}\n\n"
    )

    # ------------------------------------------------------------
    # Last epoch
    # ------------------------------------------------------------

    f.write(
        "Last Epoch Results\n"
    )

    f.write(
        "----------------------------------------\n"
    )

    f.write(
        f"Training Accuracy: "
        f"{last_train_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Validation Accuracy: "
        f"{last_val_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Training Loss: "
        f"{last_train_loss:.6f}\n"
    )

    f.write(
        f"Validation Loss: "
        f"{last_val_loss:.6f}\n\n"
    )

    # ------------------------------------------------------------
    # Training time
    # ------------------------------------------------------------

    f.write(
        f"Training Time: "
        f"{training_time / 60:.2f} minutes\n\n"
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    f.write(
        "Confusion Matrix\n"
    )

    f.write(
        "----------------------------------------\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\n"
    )

    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------

    f.write(
        "Classification Report\n"
    )

    f.write(
        "----------------------------------------\n"
    )

    f.write(
        report
    )

# ================================================================
# FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("GOOGLENET WITHOUT DATA AUGMENTATION - FINAL RESULT")
print("=" * 70)

print()

print(
    "Data augmentation: DISABLED"
)

print()

print(
    f"Training Accuracy "
    f"(last epoch): "
    f"{last_train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy "
    f"(last epoch): "
    f"{last_val_accuracy * 100:.2f}%"
)

print()

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Accuracy Epoch: "
    f"{best_val_epoch}"
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

print()

print(
    f"Confusion Matrix Accuracy: "
    f"{cm_accuracy * 100:.2f}%"
)

print()

print(
    f"Training Time: "
    f"{training_time / 60:.2f} minutes"
)

# ================================================================
# CLASS ORDER
# ================================================================

print()
print("Class order:")

for i, name in enumerate(CLASS_NAMES):

    print(
        f"  {i} = {name}"
    )

# ================================================================
# FILES CREATED
# ================================================================

print()
print("Files created:")

print(
    "1. GoogLeNet_NDA_best.keras"
)

print(
    "2. GoogLeNet_NDA_history.csv"
)

print(
    "3. GoogLeNet_NDA_Confusion_Matrix.png"
)

print(
    "4. GoogLeNet_NDA_Accuracy.png"
)

print(
    "5. GoogLeNet_NDA_Loss.png"
)

print(
    "6. GoogLeNet_NDA_final_results.txt"
)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)