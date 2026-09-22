import os

# Must be BEFORE TensorFlow import
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

PATH = r"C:\Users\E028.6\Downloads\Banana_Dataset"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 40
RANDOM_STATE = 42

print("\n==============================================")
print("BANANA RIPENESS - ENHANCED ALEXNET")
print("==============================================")

print("TensorFlow:", tf.__version__)
print("Dataset:", PATH)
print("Image size:", IMG_SIZE, "x", IMG_SIZE)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(PATH):

    raise FileNotFoundError(
        "\nDataset not found:\n" + PATH
    )


# ============================================================
# FIND 4 CLASSES
# ============================================================

class_names = sorted([
    folder
    for folder in os.listdir(PATH)
    if os.path.isdir(
        os.path.join(PATH, folder)
    )
])

print("\nClasses found:")

for i, name in enumerate(class_names):
    print(i, "->", name)


NUM_CLASSES = len(class_names)

if NUM_CLASSES != 4:

    raise ValueError(
        f"\nExpected 4 classes but found "
        f"{NUM_CLASSES}: {class_names}"
    )


# ============================================================
# COLLECT IMAGES
# ============================================================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

image_paths = []
labels = []


for class_index, class_name in enumerate(class_names):

    class_path = os.path.join(
        PATH,
        class_name
    )

    files = [
        f
        for f in os.listdir(class_path)
        if f.lower().endswith(valid_extensions)
    ]

    print(
        f"{class_name}: {len(files)} images"
    )

    for filename in files:

        image_paths.append(
            os.path.join(
                class_path,
                filename
            )
        )

        labels.append(class_index)


image_paths = np.array(image_paths)
labels = np.array(labels)


print("\nTotal images:", len(image_paths))


if len(image_paths) == 0:

    raise ValueError(
        "No images were found."
    )


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

train_paths, temp_paths, train_labels, temp_labels = train_test_split(

    image_paths,
    labels,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=labels
)


val_paths, test_paths, val_labels, test_labels = train_test_split(

    temp_paths,
    temp_labels,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=temp_labels
)


print("\n==============================================")
print("DATA SPLIT")
print("==============================================")

print("Training   :", len(train_paths))
print("Validation :", len(val_paths))
print("Testing    :", len(test_paths))


# ============================================================
# SHOW CLASS DISTRIBUTION
# ============================================================

print("\nTraining distribution:")

for i, name in enumerate(class_names):

    count = np.sum(train_labels == i)

    print(
        f"{name}: {count}"
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path, label):

    image_data = tf.io.read_file(path)

    image_data = tf.image.decode_image(
        image_data,
        channels=3,
        expand_animations=False
    )

    image_data = tf.image.resize(
        image_data,
        [IMG_SIZE, IMG_SIZE]
    )

    image_data = tf.cast(
        image_data,
        tf.float32
    )

    # Normalize to [0,1]
    image_data = image_data / 255.0

    label = tf.one_hot(
        label,
        NUM_CLASSES
    )

    return image_data, label


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

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
        height_factor=0.08,
        width_factor=0.08
    ),

    layers.RandomContrast(
        0.10
    )

], name="banana_augmentation")


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.shuffle(
    1000,
    seed=RANDOM_STATE
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(train_labels),

    y=train_labels
)


class_weights = {
    i: float(weight)
    for i, weight in enumerate(
        class_weights_array
    )
}


print("\nClass weights:")

for i, name in enumerate(class_names):

    print(
        name,
        "->",
        round(class_weights[i], 4)
    )


# ============================================================
# ENHANCED ALEXNET
# ============================================================

def build_alexnet():

    inputs = layers.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )

    # ----------------------------
    # Augmentation
    # ----------------------------

    x = data_augmentation(inputs)

    # ----------------------------
    # Block 1
    # ----------------------------

    x = layers.Conv2D(
        64,
        (5, 5),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    # ----------------------------
    # Block 2
    # ----------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    # ----------------------------
    # Block 3
    # ----------------------------

    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)


    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    # ----------------------------
    # Block 4
    # ----------------------------

    x = layers.Conv2D(
        384,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)


    x = layers.Conv2D(
        384,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    # ----------------------------
    # Classifier
    # ----------------------------

    x = layers.GlobalAveragePooling2D()(x)


    x = layers.Dense(
        512,
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        0.45
    )(x)


    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.30
    )(x)


    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)


    return Model(
        inputs,
        outputs,
        name="Banana_Enhanced_AlexNet"
    )


# ============================================================
# BUILD MODEL
# ============================================================

model = build_alexnet()

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.0003
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    ModelCheckpoint(

        "Banana_AlexNet_best.keras",

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=3,

        min_lr=1e-6,

        verbose=1
    ),

    EarlyStopping(

        monitor="val_loss",

        patience=8,

        restore_best_weights=True,

        verbose=1
    ),

    CSVLogger(
        "Banana_AlexNet_history.csv"
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n==============================================")
print("STARTING ALEXNET TRAINING")
print("==============================================")


start_time = time.time()


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


training_time = (
    time.time() - start_time
) / 60


print(
    f"\nTraining time: "
    f"{training_time:.2f} minutes"
)


# ============================================================
# TEST
# ============================================================

print("\n==============================================")
print("FINAL TEST")
print("==============================================")


test_loss, test_accuracy = model.evaluate(

    test_dataset,

    verbose=1
)


print(
    f"\nTest Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    "Banana_AlexNet_FINAL.keras"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n==============================================")
print("CLASSIFICATION REPORT")
print("==============================================")


predictions = model.predict(
    test_dataset,
    verbose=1
)


predicted_classes = np.argmax(
    predictions,
    axis=1
)


print(
    classification_report(

        test_labels,

        predicted_classes,

        target_names=class_names,

        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# BLUE THEME
# ============================================================

cm = confusion_matrix(

    test_labels,

    predicted_classes
)


print("\nConfusion Matrix:")

print(cm)


plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    cm,
    cmap="Blues"
)

plt.title(
    "Banana Ripeness - AlexNet Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(NUM_CLASSES),
    class_names,
    rotation=45
)

plt.yticks(
    range(NUM_CLASSES),
    class_names
)


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(

            j,
            i,

            str(cm[i, j]),

            ha="center",

            va="center",

            color="black"
        )


plt.colorbar()

plt.tight_layout()

plt.savefig(

    "Banana_AlexNet_confusion_matrix.png",

    dpi=300,

    bbox_inches="tight"
)

plt.show()


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(

    history.history["accuracy"],

    label="Training Accuracy",

    linewidth=2
)

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy",

    linewidth=2
)

plt.title(
    "Banana Ripeness - AlexNet Accuracy"
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

    "Banana_AlexNet_accuracy.png",

    dpi=300
)

plt.show()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(

    history.history["loss"],

    label="Training Loss",

    linewidth=2
)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss",

    linewidth=2
)

plt.title(
    "Banana Ripeness - AlexNet Loss"
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

    "Banana_AlexNet_loss.png",

    dpi=300
)

plt.show()


# ============================================================
# FINAL
# ============================================================

print("\n==============================================")
print("ALEXNET TRAINING COMPLETE")
print("==============================================")

print(
    "\nGenerated files:"
)

print(
    "Banana_AlexNet_best.keras"
)

print(
    "Banana_AlexNet_FINAL.keras"
)

print(
    "Banana_AlexNet_history.csv"
)

print(
    "Banana_AlexNet_confusion_matrix.png"
)

print(
    "Banana_AlexNet_accuracy.png"
)

print(
    "Banana_AlexNet_loss.png"
)