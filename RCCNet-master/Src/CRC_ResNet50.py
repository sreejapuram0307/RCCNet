# ============================================================
# BANANA RIPENESS - RESNET50
# WITH DATA AUGMENTATION
# FAST / BATCH-WISE VERSION
# ============================================================

import os

# Disable oneDNN messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger
)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import seaborn as sns


# ============================================================
# SETTINGS
# ============================================================

PATH = r"C:\Users\E028.6\Downloads\Banana_Dataset"

IMG_HEIGHT = 224
IMG_WIDTH = 224

BATCH_SIZE = 32

NUM_CLASSES = 4

# Faster than your previous 15 + 25 setup
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

RANDOM_STATE = 42

RESULTS_DIR = "ResNet50_DA_Results"

os.makedirs(RESULTS_DIR, exist_ok=True)


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
# HEADER
# ============================================================

print("=" * 70)
print("BANANA RIPENESS - RESNET50")
print("WITH DATA AUGMENTATION")
print("FAST BATCH-WISE VERSION")
print("=" * 70)

print("TensorFlow:", tf.__version__)
print("Dataset:", PATH)
print("Image size:", IMG_WIDTH, "x", IMG_HEIGHT)
print("Batch size:", BATCH_SIZE)
print("Stage 1 epochs:", INITIAL_EPOCHS)
print("Fine-tuning epochs:", FINE_TUNE_EPOCHS)
print("Classes:", CLASS_NAMES)
print("Data augmentation: TRUE")
print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(PATH):

    raise FileNotFoundError(
        f"Dataset not found:\n{PATH}"
    )

print("\nDataset folder found successfully.")


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# COLLECT IMAGE PATHS
# ============================================================

def collect_dataset(dataset_path):

    image_paths = []
    labels = []

    print("\n" + "=" * 70)
    print("COUNTING DATASET IMAGES")
    print("=" * 70)

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_folder = os.path.join(
            dataset_path,
            class_name
        )

        if not os.path.isdir(class_folder):

            raise FileNotFoundError(
                f"Class folder not found:\n{class_folder}"
            )

        class_images = []

        for root, dirs, files in os.walk(class_folder):

            for filename in files:

                if filename.lower().endswith(
                    IMAGE_EXTENSIONS
                ):

                    full_path = os.path.join(
                        root,
                        filename
                    )

                    class_images.append(
                        full_path
                    )

        class_images.sort()

        print(
            f"{class_name:<12}: {len(class_images):>6} images"
        )

        image_paths.extend(class_images)

        labels.extend(
            [class_index] * len(class_images)
        )

    image_paths = np.array(
        image_paths
    )

    labels = np.array(
        labels,
        dtype=np.int32
    )

    print(
        "\nTotal images:",
        len(image_paths)
    )

    return image_paths, labels


# ============================================================
# LOAD IMAGE PATHS
# ============================================================

image_paths, labels = collect_dataset(PATH)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TOTAL CLASS DISTRIBUTION")
print("=" * 70)

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(labels == i)

    print(
        f"{class_name:<12}: {count:>6}"
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING DATASET SPLIT")
print("=" * 70)

# 70% training
# 30% temporary

X_train, X_temp, y_train, y_temp = train_test_split(

    image_paths,
    labels,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=labels
)


# 15% validation
# 15% testing

X_val, X_test, y_val, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=y_temp
)


print("\nDataset split:")

print(
    f"Training   : {len(X_train)}"
)

print(
    f"Validation : {len(X_val)}"
)

print(
    f"Testing    : {len(X_test)}"
)


# ============================================================
# CLASS DISTRIBUTION AFTER SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION AFTER SPLIT")
print("=" * 70)

for i, class_name in enumerate(CLASS_NAMES):

    train_count = np.sum(y_train == i)
    val_count = np.sum(y_val == i)
    test_count = np.sum(y_test == i)

    print(
        f"{class_name:<10}"
        f" Train={train_count:>5}"
        f" Val={val_count:>5}"
        f" Test={test_count:>5}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

class_weight_values = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(y_train),

    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight
    in zip(
        np.unique(y_train),
        class_weight_values
    )
}

for class_id, weight in class_weights.items():

    print(
        f"{CLASS_NAMES[class_id]:<12}: {weight:.4f}"
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path, label):

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        [
            IMG_HEIGHT,
            IMG_WIDTH
        ]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = preprocess_input(
        image
    )

    label = tf.one_hot(
        label,
        depth=NUM_CLASSES
    )

    return image, label


# ============================================================
# DATA AUGMENTATION
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

    name="banana_data_augmentation"
)


# ============================================================
# CREATE TF.DATA DATASET
# ============================================================

def create_dataset(
    image_paths,
    labels,
    training=False
):

    dataset = tf.data.Dataset.from_tensor_slices(

        (
            image_paths,
            labels
        )
    )

    # Shuffle file paths before loading
    if training:

        dataset = dataset.shuffle(

            buffer_size=min(
                len(image_paths),
                5000
            ),

            seed=RANDOM_STATE,

            reshuffle_each_iteration=True
        )

    dataset = dataset.map(

        load_image,

        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False
    )

    # --------------------------------------------------------
    # APPLY AUGMENTATION BATCH-WISE
    # --------------------------------------------------------

    if training:

        dataset = dataset.map(

            lambda x, y:
            (
                data_augmentation(
                    x,
                    training=True
                ),
                y
            ),

            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# CREATE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("CREATING TF.DATA DATASETS")
print("=" * 70)

print("Training dataset...")

train_dataset = create_dataset(
    X_train,
    y_train,
    training=True
)

print("Validation dataset...")

val_dataset = create_dataset(
    X_val,
    y_val,
    training=False
)

print("Testing dataset...")

test_dataset = create_dataset(
    X_test,
    y_test,
    training=False
)

print("\nDatasets ready.")

print("Training augmentation : ENABLED")
print("Validation augmentation: DISABLED")
print("Testing augmentation   : DISABLED")


# ============================================================
# BUILD RESNET50
# ============================================================

print("\n" + "=" * 70)
print("BUILDING RESNET50")
print("=" * 70)

inputs = Input(

    shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    )
)


base_model = ResNet50(

    weights="imagenet",

    include_top=False,

    input_tensor=inputs
)


# ============================================================
# FREEZE RESNET
# ============================================================

base_model.trainable = False


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    512,
    activation="relu"
)(x)

x = BatchNormalization()(x)

x = Dropout(
    0.5
)(x)

outputs = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = Model(

    inputs=inputs,

    outputs=outputs,

    name="Banana_ResNet50_DA"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary(
    expand_nested=False,
    show_trainable=True
)


# ============================================================
# STAGE 1 COMPILE
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
# STAGE 1 CALLBACKS
# ============================================================

stage1_checkpoint = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_Stage1_Best.keras"
)


callbacks_stage1 = [

    ModelCheckpoint(

        stage1_checkpoint,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),

    EarlyStopping(

        monitor="val_accuracy",

        patience=3,

        mode="max",

        restore_best_weights=True,

        verbose=1
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=1,

        min_lr=1e-7,

        verbose=1
    )
]


# ============================================================
# STAGE 1 TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1")
print("RESNET50 CLASSIFICATION HEAD")
print("DATA AUGMENTATION ENABLED")
print("=" * 70)

start_time = time.time()


history1 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_stage1,

    verbose=1
)


stage1_time = time.time() - start_time


print(
    "\nStage 1 time:",
    round(stage1_time / 60, 2),
    "minutes"
)


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2")
print("FINE-TUNING RESNET50")
print("DATA AUGMENTATION ENABLED")
print("=" * 70)


# Unfreeze ResNet
base_model.trainable = True


# Freeze early layers
for layer in base_model.layers[:-30]:

    layer.trainable = False


# Freeze BatchNormalization layers
for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# COMPILE FINE-TUNING
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# STAGE 2 CALLBACKS
# ============================================================

final_checkpoint = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_Best.keras"
)


csv_path = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_History.csv"
)


callbacks_stage2 = [

    ModelCheckpoint(

        final_checkpoint,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),

    EarlyStopping(

        monitor="val_accuracy",

        patience=4,

        mode="max",

        restore_best_weights=True,

        verbose=1
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=1,

        min_lr=1e-7,

        verbose=1
    ),

    CSVLogger(

        csv_path,

        append=False
    )
]


# ============================================================
# STAGE 2 TRAINING
# ============================================================

start_time = time.time()


history2 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_stage2,

    verbose=1
)


stage2_time = time.time() - start_time


print(
    "\nStage 2 time:",
    round(stage2_time / 60, 2),
    "minutes"
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(

    test_dataset,

    verbose=1
)


print(
    f"\nTest Loss     : {test_loss:.6f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

y_true = []
y_pred = []


for images, labels_batch in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    true_classes = np.argmax(

        labels_batch.numpy(),

        axis=1
    )

    predicted_classes = np.argmax(

        predictions,

        axis=1
    )

    y_true.extend(
        true_classes
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(
    y_true
)

y_pred = np.array(
    y_pred
)


# ============================================================
# FINAL METRICS
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


print("\n" + "=" * 70)
print("FINAL PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    digits=4
)


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


report_path = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_Classification_Report.txt"
)


with open(
    report_path,
    "w"
) as f:

    f.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


plt.figure(
    figsize=(9, 8)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=CLASS_NAMES,

    yticklabels=CLASS_NAMES,

    cbar=True
)


plt.title(
    "Banana Ripeness - ResNet50 Data Augmentation",
    fontsize=16
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()


cm_path = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_Confusion_Matrix.png"
)


plt.savefig(

    cm_path,

    dpi=300,

    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMBINE HISTORY
# ============================================================

acc = []

val_acc = []

loss = []

val_loss = []


acc.extend(
    history1.history["accuracy"]
)

val_acc.extend(
    history1.history["val_accuracy"]
)

loss.extend(
    history1.history["loss"]
)

val_loss.extend(
    history1.history["val_loss"]
)


acc.extend(
    history2.history["accuracy"]
)

val_acc.extend(
    history2.history["val_accuracy"]
)

loss.extend(
    history2.history["loss"]
)

val_loss.extend(
    history2.history["val_loss"]
)


epochs = range(
    1,
    len(acc) + 1
)


# ============================================================
# ACCURACY PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)


plt.plot(

    epochs,

    acc,

    marker="o",

    label="Training Accuracy"
)


plt.plot(

    epochs,

    val_acc,

    marker="o",

    label="Validation Accuracy"
)


plt.title(
    "Banana Ripeness - ResNet50 Data Augmentation Accuracy",
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


accuracy_path = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_Accuracy.png"
)


plt.savefig(

    accuracy_path,

    dpi=300,

    bbox_inches="tight"
)

plt.close()


# ============================================================
# LOSS PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)


plt.plot(

    epochs,

    loss,

    marker="o",

    label="Training Loss"
)


plt.plot(

    epochs,

    val_loss,

    marker="o",

    label="Validation Loss"
)


plt.title(
    "Banana Ripeness - ResNet50 Data Augmentation Loss",
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


loss_path = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_Loss.png"
)


plt.savefig(

    loss_path,

    dpi=300,

    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE COMPLETE HISTORY
# ============================================================

history_df = pd.DataFrame({

    "epoch":
        list(epochs),

    "accuracy":
        acc,

    "val_accuracy":
        val_acc,

    "loss":
        loss,

    "val_loss":
        val_loss

})


history_complete_path = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_Complete_History.csv"
)


history_df.to_csv(

    history_complete_path,

    index=False
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics_path = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_Final_Metrics.txt"
)


with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "BANANA RIPENESS - RESNET50\n"
    )

    f.write(
        "WITH DATA AUGMENTATION\n\n"
    )

    f.write(
        f"Test Loss: {test_loss:.6f}\n"
    )

    f.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall: {recall * 100:.2f}%\n"
    )

    f.write(
        f"F1 Score: {f1 * 100:.2f}%\n"
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_Final.keras"
)


model.save(
    final_model_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    "\nData augmentation : ENABLED"
)

print(
    f"Test Accuracy     : {accuracy * 100:.2f}%"
)

print(
    f"Precision          : {precision * 100:.2f}%"
)

print(
    f"Recall             : {recall * 100:.2f}%"
)

print(
    f"F1 Score           : {f1 * 100:.2f}%"
)


print("\nResults saved in:")

print(
    os.path.abspath(
        RESULTS_DIR
    )
)


print("\nSaved files:")

print(
    "1. ResNet50_DA_Accuracy.png"
)

print(
    "2. ResNet50_DA_Loss.png"
)

print(
    "3. ResNet50_DA_Confusion_Matrix.png"
)

print(
    "4. ResNet50_DA_Classification_Report.txt"
)

print(
    "5. ResNet50_DA_Final_Metrics.txt"
)

print(
    "6. ResNet50_DA_Complete_History.csv"
)

print(
    "7. ResNet50_DA_Final.keras"
)

print(
    "8. ResNet50_DA_Best.keras"
)

print(
    "\n" + "=" * 70
)

print(
    "DONE"
)

print(
    "=" * 70
)