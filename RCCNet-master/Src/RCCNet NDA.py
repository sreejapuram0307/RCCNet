
# ============================================================
# RCCNet - Banana Ripeness Classification
# WITHOUT DATA AUGMENTATION
# ============================================================

import os

# Disable oneDNN messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = r"C:\Users\E028.6\Downloads\Banana_Dataset"

IMG_SIZE = (32, 32)

NUM_CLASSES = 4

BATCH_SIZE = 64

EPOCHS = 30

RANDOM_STATE = 42


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Overripe",
    "Rippen",
    "Rotten",
    "Unripe"
]


# ============================================================
# CHECK DATASET
# ============================================================

print()
print("=" * 70)
print("RCCNet - BANANA RIPENESS CLASSIFICATION")
print("WITHOUT DATA AUGMENTATION")
print("=" * 70)
print()
print("Dataset path:")
print(DATASET_PATH)

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset path does not exist:\n{DATASET_PATH}"
    )


# ============================================================
# FIND CLASS FOLDERS
# ============================================================

subfolders = [

    folder

    for folder in os.listdir(DATASET_PATH)

    if os.path.isdir(
        os.path.join(
            DATASET_PATH,
            folder
        )
    )
]


# Sort classes using the desired order
subfolders = [

    class_name

    for class_name in CLASS_NAMES

    if class_name in subfolders
]


print()
print("Classes found:")

for i, class_name in enumerate(subfolders):

    print(
        f"   {i}: {class_name}"
    )


if len(subfolders) != NUM_CLASSES:

    raise ValueError(
        f"\nExpected {NUM_CLASSES} classes "
        f"but found {len(subfolders)}."
    )


# ============================================================
# LOAD IMAGES
# ============================================================

images = []

labels = []

print()
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)


for label_idx, class_name in enumerate(subfolders):

    folder_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    files = os.listdir(folder_path)

    image_count = 0

    print()
    print(
        f"Loading class {label_idx}: {class_name}"
    )

    for filename in files:

        # Only image files
        if not filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            )
        ):

            continue

        image_path = os.path.join(
            folder_path,
            filename
        )

        img = cv2.imread(
            image_path
        )

        if img is None:

            continue

        # Convert BGR -> RGB
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # Resize to RCCNet input size
        img = cv2.resize(
            img,
            IMG_SIZE
        )

        images.append(img)

        labels.append(
            label_idx
        )

        image_count += 1

    print(
        f"Loaded: {image_count} images"
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(
    images,
    dtype="float32"
)

y = np.array(
    labels,
    dtype="int64"
)


print()
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(
    "Total images:",
    len(X)
)

print(
    "Image shape:",
    X.shape
)


# ============================================================
# NORMALIZATION
# ============================================================

# No augmentation.
# Only standard pixel normalization.

X = X / 255.0


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("Class distribution:")

for class_index, class_name in enumerate(subfolders):

    count = np.sum(
        y == class_index
    )

    print(
        f"   {class_name}: {count}"
    )


# ============================================================
# ONE-HOT ENCODING
# ============================================================

y_cat = to_categorical(
    y,
    num_classes=NUM_CLASSES
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print()
print("=" * 70)
print("TRAIN / VALIDATION SPLIT")
print("=" * 70)

X_train, X_val, y_train, y_val = train_test_split(

    X,
    y_cat,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    "Training images:",
    len(X_train)
)

print(
    "Validation images:",
    len(X_val)
)


# ============================================================
# RCCNet MODEL
# ============================================================

def build_rccnet(
    input_shape=(32, 32, 3),
    num_classes=4
):

    inputs = Input(
        shape=input_shape
    )


    # ========================================================
    # CONVOLUTIONAL BLOCK 1
    # ========================================================

    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(inputs)

    x = BatchNormalization()(x)

    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        (2, 2)
    )(x)


    # ========================================================
    # CONVOLUTIONAL BLOCK 2
    # ========================================================

    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        (2, 2)
    )(x)


    # ========================================================
    # CLASSIFIER
    # ========================================================

    x = Flatten()(x)

    x = Dense(
        256,
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    x = Dropout(
        0.5
    )(x)


    # ========================================================
    # FOUR BANANA CLASSES
    # ========================================================

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)


    model = Model(

        inputs=inputs,

        outputs=outputs,

        name="RCCNet_Banana_NDA"
    )


    return model


# ============================================================
# CREATE MODEL
# ============================================================

model = build_rccnet(
    input_shape=(32, 32, 3),
    num_classes=NUM_CLASSES
)


print()
print("=" * 70)
print("RCCNet MODEL")
print("=" * 70)

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=1e-3
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

        "RCCNet_NDA_best.keras",

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

        "RCCNet_NDA_history.csv",

        append=False
    )
]


# ============================================================
# IMPORTANT
# ============================================================

print()
print("=" * 70)
print("DATA AUGMENTATION = DISABLED")
print("=" * 70)

print()
print("No rotation")
print("No zoom")
print("No horizontal flip")
print("No vertical flip")
print("No width shift")
print("No height shift")
print("No shear")
print("No brightness augmentation")
print()

print(
    "Training directly on original training images."
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING RCCNet NO-AUGMENTATION TRAINING")
print("=" * 70)

start_time = time.time()


history = model.fit(

    X_train,

    y_train,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=(

        X_val,

        y_val
    ),

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
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Training time: "
    f"{training_time / 60:.2f} minutes"
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST RCCNet NDA MODEL")
print("=" * 70)


model = tf.keras.models.load_model(
    "RCCNet_NDA_best.keras"
)


print(
    "Best model loaded successfully."
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL VALIDATION RESULTS")
print("=" * 70)


val_loss, val_accuracy = model.evaluate(

    X_val,

    y_val,

    verbose=1
)


print()
print(
    f"Validation Loss     : "
    f"{val_loss:.6f}"
)

print(
    f"Validation Accuracy : "
    f"{val_accuracy:.6f}"
)

print(
    f"Validation Accuracy : "
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)


probabilities = model.predict(

    X_val,

    batch_size=BATCH_SIZE,

    verbose=1
)


y_pred = np.argmax(
    probabilities,
    axis=1
)

y_true = np.argmax(
    y_val,
    axis=1
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=np.arange(
        NUM_CLASSES
    )
)


print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(cm)


# ============================================================
# ACCURACY FROM CONFUSION MATRIX
# ============================================================

cm_accuracy = (

    np.trace(cm)
    /
    np.sum(cm)

)


print()

print(
    "Accuracy from confusion matrix:",
    f"{cm_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(

    classification_report(

        y_true,

        y_pred,

        target_names=subfolders,

        digits=4
    )

)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(8, 7)
)


plt.imshow(
    cm,
    interpolation="nearest",
    cmap=plt.cm.Blues
)


plt.title(
    "RCCNet - Banana Ripeness\n"
    "Without Data Augmentation",
    fontsize=15
)


plt.colorbar()


tick_marks = np.arange(
    NUM_CLASSES
)


plt.xticks(
    tick_marks,
    subfolders,
    rotation=45
)


plt.yticks(
    tick_marks,
    subfolders
)


threshold = cm.max() / 2.0


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(

            j,

            i,

            str(cm[i, j]),

            ha="center",

            va="center",

            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),

            fontsize=12
        )


plt.ylabel(
    "Actual Class"
)

plt.xlabel(
    "Predicted Class"
)


plt.tight_layout()


plt.savefig(

    "RCCNet_NDA_Confusion_Matrix.png",

    dpi=300,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# ACCURACY CURVE
# ============================================================

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
    "RCCNet Banana Ripeness - "
    "Without Data Augmentation"
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

    "RCCNet_NDA_Accuracy.png",

    dpi=300,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# LOSS CURVE
# ============================================================

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
    "RCCNet Banana Ripeness - "
    "Without Data Augmentation"
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

    "RCCNet_NDA_Loss.png",

    dpi=300,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# BEST RESULTS
# ============================================================

best_train_accuracy = max(
    history.history["accuracy"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_train_loss = min(
    history.history["loss"]
)

best_val_loss = min(
    history.history["val_loss"]
)


best_val_epoch = (

    np.argmax(
        history.history["val_accuracy"]
    )
    + 1

)


print()
print("=" * 70)
print("BEST RESULTS")
print("=" * 70)


print(
    f"Best Training Accuracy   : "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Training Loss       : "
    f"{best_train_loss:.6f}"
)

print(
    f"Best Validation Loss     : "
    f"{best_val_loss:.6f}"
)

print(
    f"Best Validation Epoch    : "
    f"{best_val_epoch}"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    "RCCNet_NDA_final.keras"
)


print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    "RCCNet_NDA_best.keras"
)

print(
    "RCCNet_NDA_final.keras"
)

print(
    "RCCNet_NDA_history.csv"
)

print(
    "RCCNet_NDA_Confusion_Matrix.png"
)

print(
    "RCCNet_NDA_Accuracy.png"
)

print(
    "RCCNet_NDA_Loss.png"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("RCCNet NO-DATA-AUGMENTATION EXPERIMENT FINISHED")
print("=" * 70)

print()
print(
    "Data augmentation was NOT used."
)

print(
    "The existing DA model files were not modified."
)

print()
