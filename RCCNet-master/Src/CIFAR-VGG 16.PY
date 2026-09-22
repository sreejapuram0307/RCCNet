# ================================================================
# VGG16 BANANA RIPENESS CLASSIFICATION
# DATA AUGMENTATION = TRUE
# ================================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    classification_report,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import seaborn as sns


# ================================================================
# 1. SETTINGS
# ================================================================

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("=" * 70)
print("VGG16 - BANANA RIPENESS CLASSIFICATION")
print("DATA AUGMENTATION: TRUE")
print("=" * 70)

print("TensorFlow:", tf.__version__)


# ================================================================
# 2. DATASET PATH
# ================================================================

DATASET_PATH = r"C:\Users\E028.6\Downloads\Banana_Dataset"

IMG_SIZE = (128, 128)

NUM_CLASSES = 4

BATCH_SIZE = 32

WARMUP_EPOCHS = 8

FINETUNE_EPOCHS = 22

TOTAL_EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS


# ================================================================
# 3. CHECK DATASET
# ================================================================

print("\nDataset:")
print(DATASET_PATH)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        "\nDataset not found:\n"
        + DATASET_PATH
        + "\n\nPlease change DATASET_PATH to your actual Banana_Dataset folder."
    )


# ================================================================
# 4. FIND THE FOUR CLASSES
# ================================================================

classes = [
    "Overripe",
    "Rippen",
    "Rotten",
    "Unripe"
]

print("\nExpected classes:")

for i, class_name in enumerate(classes):
    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(class_path):
        raise FileNotFoundError(
            f"\nClass folder not found:\n{class_path}\n\n"
            f"Your dataset must contain these folders:\n"
            f"Overripe\n"
            f"Rippen\n"
            f"Rotten\n"
            f"Unripe"
        )

    number_of_files = len(
        [
            f for f in os.listdir(class_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        ]
    )

    print(f"{i}: {class_name} -> {number_of_files} images")


# ================================================================
# 5. LOAD IMAGES
# ================================================================

print("\nLoading images...")

images = []
labels = []

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

for class_index, class_name in enumerate(classes):

    folder_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    print(f"\nLoading: {class_name}")

    files = os.listdir(folder_path)

    count = 0

    for filename in files:

        if not filename.lower().endswith(valid_extensions):
            continue

        image_path = os.path.join(
            folder_path,
            filename
        )

        try:

            img = tf.keras.utils.load_img(
                image_path,
                target_size=IMG_SIZE
            )

            img = tf.keras.utils.img_to_array(img)

            images.append(img)

            labels.append(class_index)

            count += 1

        except Exception as e:

            print(
                "Skipped:",
                filename,
                "|",
                str(e)
            )

    print(
        f"Loaded {count} images from {class_name}"
    )


X = np.array(
    images,
    dtype=np.float32
)

y = np.array(
    labels,
    dtype=np.int32
)


print("\n================================================")
print("DATASET INFORMATION")
print("================================================")

print("Images:", X.shape)
print("Labels:", y.shape)

print("\nClass distribution:")

for i, class_name in enumerate(classes):

    count = np.sum(y == i)

    print(
        f"{class_name}: {count}"
    )


# ================================================================
# 6. TRAIN / VALIDATION SPLIT
# ================================================================

print("\nCreating train/validation split...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining images:", len(X_train))
print("Validation images:", len(X_val))


# ================================================================
# 7. ONE-HOT ENCODING
# ================================================================

y_train_cat = to_categorical(
    y_train,
    NUM_CLASSES
)

y_val_cat = to_categorical(
    y_val,
    NUM_CLASSES
)


# ================================================================
# 8. CLASS WEIGHTS
# ================================================================
#
# This is important because your previous confusion matrix
# showed that OVERripe was being classified poorly.
#
# Class weights give more importance to classes with fewer images.
# ================================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    i: class_weights_array[i]
    for i in range(NUM_CLASSES)
}

print("\nClass weights:")

for i, class_name in enumerate(classes):

    print(
        f"{class_name}: "
        f"{class_weights[i]:.4f}"
    )


# ================================================================
# 9. DATA AUGMENTATION
# ================================================================
#
# DATA AUGMENTATION = TRUE
#
# Validation images are NOT augmented.
# Only training images are augmented.
# ================================================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=20,

    width_shift_range=0.10,

    height_shift_range=0.10,

    shear_range=0.10,

    zoom_range=0.15,

    horizontal_flip=True,

    brightness_range=(0.85, 1.15),

    fill_mode="nearest"
)


# Validation:
# NO augmentation
#
# Only VGG16 preprocessing is applied.

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# ================================================================
# 10. CREATE DATA GENERATORS
# ================================================================

train_generator = train_datagen.flow(
    X_train,
    y_train_cat,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


val_generator = val_datagen.flow(
    X_val,
    y_val_cat,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ================================================================
# 11. BUILD VGG16
# ================================================================

print("\n================================================")
print("BUILDING VGG16")
print("================================================")


base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)


# ------------------------------------------------
# Freeze VGG16 initially
# ------------------------------------------------

base_model.trainable = False


# ================================================================
# 12. ADD CLASSIFIER
# ================================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    512,
    activation="relu"
)(x)

x = Dropout(
    0.5
)(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(
    0.3
)(x)

outputs = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = Model(
    inputs=base_model.input,
    outputs=outputs,
    name="VGG16_Banana_Ripeness"
)


# ================================================================
# 13. FIRST COMPILE
# ================================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.0001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


model.summary()


# ================================================================
# 14. CALLBACKS
# ================================================================

checkpoint = ModelCheckpoint(

    "VGG16_Banana_Best.keras",

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=3,

    min_lr=1e-7,

    verbose=1
)


early_stop = EarlyStopping(

    monitor="val_loss",

    patience=7,

    restore_best_weights=True,

    verbose=1
)


csv_logger = CSVLogger(
    "VGG16_Banana_history.csv"
)


callbacks = [
    checkpoint,
    reduce_lr,
    early_stop,
    csv_logger
]


# ================================================================
# 15. PHASE 1 - TRAIN CLASSIFIER
# ================================================================

print("\n================================================")
print("PHASE 1")
print("VGG16 BASE FROZEN")
print("DATA AUGMENTATION ENABLED")
print("================================================")


start_time = time.time()


history1 = model.fit(

    train_generator,

    epochs=WARMUP_EPOCHS,

    validation_data=val_generator,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


print(
    "\nPhase 1 completed in "
    f"{time.time() - start_time:.2f} seconds"
)


# ================================================================
# 16. PHASE 2 - FINE TUNING
# ================================================================

print("\n================================================")
print("PHASE 2")
print("FINE-TUNING VGG16")
print("================================================")


# Unfreeze VGG16

base_model.trainable = True


# Freeze the earlier layers.
#
# Only the last convolutional block is allowed
# to learn new banana-specific features.

for layer in base_model.layers:

    layer.trainable = False


# Unfreeze last few VGG16 layers

for layer in base_model.layers[-8:]:

    layer.trainable = True


print("\nTrainable layers:")

for layer in base_model.layers:

    if layer.trainable:

        print(
            "Trainable:",
            layer.name
        )


# ================================================================
# 17. RECOMPILE WITH SMALL LEARNING RATE
# ================================================================

model.compile(

    optimizer=Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ================================================================
# 18. FINE-TUNE
# ================================================================

start_time = time.time()


history2 = model.fit(

    train_generator,

    initial_epoch=WARMUP_EPOCHS,

    epochs=TOTAL_EPOCHS,

    validation_data=val_generator,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


print(
    "\nPhase 2 completed in "
    f"{time.time() - start_time:.2f} seconds"
)


# ================================================================
# 19. LOAD BEST MODEL
# ================================================================

print("\nLoading best VGG16 model...")

model = tf.keras.models.load_model(
    "VGG16_Banana_Best.keras"
)


# ================================================================
# 20. EVALUATE
# ================================================================

print("\n================================================")
print("FINAL MODEL EVALUATION")
print("================================================")


results = model.evaluate(
    val_generator,
    verbose=1
)


test_loss = results[0]

test_accuracy = results[1]


print(
    f"\nValidation Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ================================================================
# 21. PREDICTIONS
# ================================================================

print("\nGenerating predictions...")


val_generator.reset()


predictions = model.predict(
    val_generator,
    verbose=1
)


y_pred = np.argmax(
    predictions,
    axis=1
)


y_true = y_val


final_accuracy = accuracy_score(
    y_true,
    y_pred
)


print(
    "\nConfusion Matrix Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)


# ================================================================
# 22. CLASSIFICATION REPORT
# ================================================================

print("\n================================================")
print("CLASSIFICATION REPORT")
print("================================================")


print(
    classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4
    )
)


# ================================================================
# 23. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\nConfusion Matrix:")
print(cm)


plt.figure(
    figsize=(9, 7)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=classes,

    yticklabels=classes,

    linewidths=0.5,

    cbar=True

)


plt.title(
    "Banana Ripeness - VGG16 Confusion Matrix",
    fontsize=16
)

plt.xlabel(
    "Predicted Class",
    fontsize=12
)

plt.ylabel(
    "Actual Class",
    fontsize=12
)

plt.tight_layout()


plt.savefig(
    "VGG16_Banana_Confusion_Matrix.png",
    dpi=300
)


plt.show()


# ================================================================
# 24. COMBINE TRAINING HISTORIES
# ================================================================

acc = (
    history1.history["accuracy"]
    +
    history2.history["accuracy"]
)

val_acc = (
    history1.history["val_accuracy"]
    +
    history2.history["val_accuracy"]
)

loss = (
    history1.history["loss"]
    +
    history2.history["loss"]
)

val_loss = (
    history1.history["val_loss"]
    +
    history2.history["val_loss"]
)


epochs_range = range(
    1,
    len(acc) + 1
)


# ================================================================
# 25. ACCURACY GRAPH
# ================================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    epochs_range,
    acc,
    label="Training Accuracy",
    linewidth=2
)


plt.plot(
    epochs_range,
    val_acc,
    label="Validation Accuracy",
    linewidth=2
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Banana Ripeness - VGG16 Accuracy"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "VGG16_Banana_Accuracy.png",
    dpi=300
)


plt.show()


# ================================================================
# 26. LOSS GRAPH
# ================================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    epochs_range,
    loss,
    label="Training Loss",
    linewidth=2
)


plt.plot(
    epochs_range,
    val_loss,
    label="Validation Loss",
    linewidth=2
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Banana Ripeness - VGG16 Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "VGG16_Banana_Loss.png",
    dpi=300
)


plt.show()


# ================================================================
# 27. SAVE MODEL
# ================================================================

model.save(
    "VGG16_Banana_Final.keras"
)


print("\n================================================")
print("TRAINING FINISHED")
print("================================================")

print(
    f"Final validation accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Final validation loss: "
    f"{test_loss:.4f}"
)

print("\nFiles created:")

print("VGG16_Banana_Best.keras")

print("VGG16_Banana_Final.keras")

print("VGG16_Banana_history.csv")

print("VGG16_Banana_Accuracy.png")

print("VGG16_Banana_Loss.png")

print("VGG16_Banana_Confusion_Matrix.png")

print("\n================================================")