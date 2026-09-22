# ============================================================
# ALEXNET - BANANA RIPENESS CLASSIFICATION
# WITHOUT DATA AUGMENTATION
# ============================================================

import os

# Disable oneDNN messages BEFORE importing TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    ModelCheckpoint,
    EarlyStopping
)
from tensorflow.keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ============================================================
# SETTINGS
# ============================================================

PATH = r"C:\Users\E028.6\Downloads\Banana_Dataset"

IMG_HEIGHT = 128
IMG_WIDTH = 128

NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 40

TEST_SIZE = 0.20
RANDOM_STATE = 42

# IMPORTANT:
# This experiment DOES NOT use data augmentation.
DATA_AUGMENTATION = False

CLASS_NAMES = [
    "Overripe",
    "Rippen",
    "Rotten",
    "Unripe"
]

# ============================================================
# PRINT INFORMATION
# ============================================================

print("=" * 65)
print("ALEXNET - WITHOUT DATA AUGMENTATION")
print("=" * 65)

print("TensorFlow:", tf.__version__)
print("Dataset:", PATH)
print(f"Image size: {IMG_WIDTH} x {IMG_HEIGHT}")
print("Classes:", NUM_CLASSES)
print("Data augmentation:", DATA_AUGMENTATION)
print()

# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{PATH}\n\n"
        "Please check that Banana_Dataset exists at this location."
    )

print("Dataset found.")

# Check the four required folders
for class_name in CLASS_NAMES:
    class_path = os.path.join(PATH, class_name)

    if not os.path.isdir(class_path):
        raise FileNotFoundError(
            f"\nRequired class folder not found:\n{class_path}\n\n"
            f"Expected folders:\n"
            f"  Overripe\n"
            f"  Rippen\n"
            f"  Rotten\n"
            f"  Unripe"
        )

print("All four class folders found:")
for class_name in CLASS_NAMES:
    class_path = os.path.join(PATH, class_name)
    files = [
        f for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f))
    ]
    print(f"  {class_name}: {len(files)} images")

print()

# ============================================================
# ALPHANUMERIC SORT
# ============================================================

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [
        convert(c) for c in re.split("([0-9]+)", key)
    ]
    return sorted(data, key=alphanum_key)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(dataset_path):

    images = []
    labels = []

    print("=" * 65)
    print("LOADING BANANA DATASET")
    print("=" * 65)

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_path = os.path.join(
            dataset_path,
            class_name
        )

        file_names = sorted_alphanumeric(
            os.listdir(class_path)
        )

        loaded_count = 0
        skipped_count = 0

        print(f"\nClass: {class_name}")

        for file_name in file_names:

            file_path = os.path.join(
                class_path,
                file_name
            )

            # Ignore directories
            if not os.path.isfile(file_path):
                continue

            try:

                # Load image
                img = image.load_img(
                    file_path,
                    target_size=(IMG_HEIGHT, IMG_WIDTH)
                )

                # Convert to NumPy array
                img_array = image.img_to_array(img)

                # Normalize pixels to 0-1
                img_array = img_array / 255.0

                images.append(img_array)
                labels.append(class_index)

                loaded_count += 1

            except Exception as e:

                skipped_count += 1

        print(f"  Loaded: {loaded_count}")
        print(f"  Skipped: {skipped_count}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print()
    print("=" * 65)
    print("DATASET LOADED")
    print("=" * 65)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Verify classes
    unique, counts = np.unique(y, return_counts=True)

    print("\nClass distribution:")

    for class_index, count in zip(unique, counts):
        print(
            f"  {CLASS_NAMES[class_index]}: {count}"
        )

    return X, y


# ============================================================
# LOAD DATA
# ============================================================

X, y = load_dataset(PATH)

# ============================================================
# VERIFY FOUR CLASSES
# ============================================================

unique_classes = np.unique(y)

if len(unique_classes) != NUM_CLASSES:

    raise ValueError(
        f"\nExpected {NUM_CLASSES} classes, "
        f"but found {len(unique_classes)} classes."
    )

print()
print("SUCCESS: All 4 banana classes are present.")

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 65)
print("CREATING TRAIN / TEST SPLIT")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training images:", len(X_train))
print("Testing images :", len(X_test))

# ============================================================
# CHECK SPLIT DISTRIBUTION
# ============================================================

print("\nTraining distribution:")

train_unique, train_counts = np.unique(
    y_train,
    return_counts=True
)

for class_index, count in zip(
    train_unique,
    train_counts
):
    print(
        f"  {CLASS_NAMES[class_index]}: {count}"
    )

print("\nTesting distribution:")

test_unique, test_counts = np.unique(
    y_test,
    return_counts=True
)

for class_index, count in zip(
    test_unique,
    test_counts
):
    print(
        f"  {CLASS_NAMES[class_index]}: {count}"
    )

# ============================================================
# BUILD ALEXNET
# ============================================================

def build_alexnet():

    model = Sequential(
        [

            Input(
                shape=(
                    IMG_HEIGHT,
                    IMG_WIDTH,
                    3
                )
            ),

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            Conv2D(
                96,
                kernel_size=(5, 5),
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),

            MaxPooling2D(
                pool_size=(2, 2)
            ),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            Conv2D(
                256,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),

            MaxPooling2D(
                pool_size=(2, 2)
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            Conv2D(
                384,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),

            Conv2D(
                384,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),

            Conv2D(
                256,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),

            MaxPooling2D(
                pool_size=(2, 2)
            ),

            # ------------------------------------------------
            # Fully connected section
            # ------------------------------------------------

            Flatten(),

            Dense(
                1024,
                activation="relu"
            ),

            Dropout(0.5),

            Dense(
                512,
                activation="relu"
            ),

            Dropout(0.5),

            # ------------------------------------------------
            # Four banana classes
            # ------------------------------------------------

            Dense(
                NUM_CLASSES,
                activation="softmax"
            )

        ],
        name="AlexNet_Banana_No_Augmentation"
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

model = build_alexnet()

print()
print("=" * 65)
print("ALEXNET MODEL")
print("=" * 65)

model.summary()

# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),

    ModelCheckpoint(
        "AlexNet_NDA_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    CSVLogger(
        "AlexNet_NDA_history.csv"
    )

]

# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 65)
print("STARTING TRAINING")
print("=" * 65)

print("Data augmentation: DISABLED")
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)

start_time = time.time()

history = model.fit(

    X_train,
    y_train,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=(
        X_test,
        y_test
    ),

    shuffle=True,

    callbacks=callbacks,

    verbose=1
)

training_time = time.time() - start_time

print()
print(
    f"Training completed in "
    f"{training_time:.2f} seconds."
)

# ============================================================
# EVALUATE
# ============================================================

print()
print("=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print()
print(f"Test Loss     : {test_loss:.6f}")
print(f"Test Accuracy : {test_accuracy:.6f}")
print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 65)
print("GENERATING CONFUSION MATRIX")
print("=" * 65)

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

# ============================================================
# ACCURACY FROM CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

cm_accuracy = np.trace(cm) / np.sum(cm)

print()
print(
    "Accuracy calculated from "
    "confusion matrix:"
)

print(
    f"{cm_accuracy * 100:.2f}%"
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("Confusion Matrix:")
print(cm)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

# ============================================================
# PLOT ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
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

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "AlexNet_NDA_Accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(
    figsize=(10, 6)
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

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "AlexNet_NDA_Loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# BLUE CONFUSION MATRIX
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
    "Banana Ripeness - AlexNet Confusion Matrix",
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

# Write numbers inside cells
threshold = cm.max() / 2.0

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
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
    "AlexNet_NDA_Confusion_Matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 65)
print("ALEXNET WITHOUT DATA AUGMENTATION - FINAL RESULT")
print("=" * 65)

print(
    f"Training Accuracy (last epoch): "
    f"{history.history['accuracy'][-1] * 100:.2f}%"
)

print(
    f"Validation Accuracy (last epoch): "
    f"{history.history['val_accuracy'][-1] * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: "
    f"{test_loss:.6f}"
)

print(
    f"Confusion Matrix Accuracy: "
    f"{cm_accuracy * 100:.2f}%"
)

print()
print("Class order:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i} = {name}")

print()
print("Files created:")
print("  AlexNet_NDA_best.keras")
print("  AlexNet_NDA_history.csv")
print("  AlexNet_NDA_Accuracy.png")
print("  AlexNet_NDA_Loss.png")
print("  AlexNet_NDA_Confusion_Matrix.png")

print()
print("=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)