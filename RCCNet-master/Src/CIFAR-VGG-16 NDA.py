'''import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ==========================================
# 1. PATH SETUP (Update this path)
# ==========================================
PATH = r"C:/Users/E028.6/Downloads/Banana/Banana/Test Set"  # Replace with your dataset directory
print("Dataset Directory:", PATH)

# ==========================================
# 2. METRICS & UTILITIES
# ==========================================
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

# ==========================================
# 3. DATA PREPROCESSING & LOADING
# ==========================================
if not os.path.exists(PATH):
    raise FileNotFoundError(f"Directory '{PATH}' not found. Please update the PATH variable.")

data_dir_list = sorted_alphanumeric(os.listdir(PATH))
print("Found subdirectories:", data_dir_list)

img_data_list = []

for dataset in data_dir_list:
    dir_path = os.path.join(PATH, dataset)
    if not os.path.isdir(dir_path):
        continue
    img_list = sorted_alphanumeric(os.listdir(dir_path))
    print(f"Loading images from folder: {dataset}")
    for img_name in img_list:
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        img_path = os.path.join(dir_path, img_name)
        img = image.load_img(img_path, target_size=(32, 32))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_data_list.append(x)

img_data = np.array(img_data_list)
img_data = np.squeeze(img_data, axis=1)
print("Image Dataset Shape:", img_data.shape)

num_classes = 4
num_of_samples = img_data.shape[0]
print("Total Samples:", num_of_samples)

# Assign class labels
labels = np.ones((num_of_samples,), dtype='int64')
labels[0:7722] = 0
labels[7722:13434] = 1
labels[13434:20225] = 2
labels[20225:] = 3

Y = to_categorical(labels, num_classes)

# Fixed seed (42) guarantees identical dataset splits across all models for fair comparison
x, y = shuffle(img_data, Y, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ==========================================
# 4. VGG16 ARCHITECTURE (13 Conv + 3 Dense)
# ==========================================
def build_vgg16(input_shape=(32, 32, 3), num_classes=4):
    inputs = Input(shape=input_shape)

    # Block 1 (2 Conv) -> Output: 32x32x64
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)

    # Block 2 (2 Conv + Pool) -> Output: 16x16x128
    x = MaxPooling2D((2, 2), name='block1_pool')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)

    # Block 3 (3 Conv + Pool) -> Output: 8x8x256
    x = MaxPooling2D((2, 2), name='block2_pool')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)

    # Block 4 (3 Conv + Pool) -> Output: 4x4x512
    x = MaxPooling2D((2, 2), name='block3_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)

    # Block 5 (3 Conv + Pool) -> Output: 2x2x512
    x = MaxPooling2D((2, 2), name='block4_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)

    # Classification Head (FC1, FC2, Output)
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    return Model(inputs=inputs, outputs=outputs, name='VGG16')

model = build_vgg16(input_shape=(32, 32, 3), num_classes=num_classes)
model.summary()

# ==========================================
# 5. COMPILATION & CALLBACKS
# ==========================================
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy', f1]
)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6)
csv_logger = CSVLogger('vgg16_history.csv', append=False, separator=';')
model_checkpoint = ModelCheckpoint('vgg16_best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

# ==========================================
# 6. TRAINING EXECUTION
# ==========================================
batch_size = 64
epochs = 50

start_time = time.time()

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    shuffle=True,
    callbacks=[lr_reducer, csv_logger, model_checkpoint, early_stopping]
)

total_time = time.time() - start_time
print(f"\n--- Total VGG16 Training Time: {total_time:.2f} seconds ---")

# ==========================================
# 7. EVALUATION & PLOTTING
# ==========================================
scores = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {scores[0]:.4f}")
print(f"Test Accuracy: {scores[1]:.4f}")
if len(scores) > 2:
    print(f"Test F1-Score: {scores[2]:.4f}")

# Plot performance curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('VGG16 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('VGG16 Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()'''

# ============================================================
# VGG16 - BANANA RIPENESS CLASSIFICATION
# WITHOUT DATA AUGMENTATION
# ============================================================

import os

# Disable oneDNN messages before TensorFlow import
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
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
# SETTINGS
# ============================================================

DATASET_PATH = r"C:\Users\E028.6\Downloads\Banana_Dataset"

IMG_HEIGHT = 32
IMG_WIDTH = 32

NUM_CLASSES = 4

BATCH_SIZE = 64

EPOCHS = 30

RANDOM_STATE = 42

# ------------------------------------------------------------
# IMPORTANT:
# Data augmentation is completely disabled.
# ------------------------------------------------------------

DATA_AUGMENTATION = False

RESULTS_DIR = "VGG16_NDA_Results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


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

print()
print("=" * 75)
print("VGG16 - BANANA RIPENESS CLASSIFICATION")
print("WITHOUT DATA AUGMENTATION")
print("=" * 75)

print("TensorFlow:", tf.__version__)
print("Dataset:", DATASET_PATH)
print("Image size:", IMG_WIDTH, "x", IMG_HEIGHT)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Classes:", CLASS_NAMES)
print("Data augmentation:", DATA_AUGMENTATION)

print("=" * 75)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}\n\n"
        "Please check the dataset path."
    )


print()
print("Dataset path found.")


# ============================================================
# CHECK CLASS FOLDERS
# ============================================================

for class_name in CLASS_NAMES:

    class_folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_folder):

        raise FileNotFoundError(
            f"\nMissing class folder:\n{class_folder}\n\n"
            "Expected folders:\n"
            "  Overripe\n"
            "  Rippen\n"
            "  Rotten\n"
            "  Unripe"
        )


print("All four class folders found.")


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# LOAD DATASET
# ============================================================

images = []
labels = []


print()
print("=" * 75)
print("LOADING DATASET")
print("=" * 75)


for class_index, class_name in enumerate(CLASS_NAMES):

    class_folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    class_files = [
        file_name
        for file_name in os.listdir(class_folder)
        if file_name.lower().endswith(IMAGE_EXTENSIONS)
    ]

    class_files.sort()

    loaded_count = 0
    skipped_count = 0

    print()
    print(
        f"Class: {class_name}"
    )

    for file_name in class_files:

        image_path = os.path.join(
            class_folder,
            file_name
        )

        try:

            img = cv2.imread(
                image_path
            )

            if img is None:

                skipped_count += 1
                continue

            # BGR -> RGB
            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            # Resize
            img = cv2.resize(
                img,
                (IMG_WIDTH, IMG_HEIGHT)
            )

            images.append(
                img
            )

            labels.append(
                class_index
            )

            loaded_count += 1

        except Exception as error:

            skipped_count += 1

            print(
                "Skipped:",
                file_name,
                "|",
                error
            )

    print(
        "Loaded:",
        loaded_count
    )

    print(
        "Skipped:",
        skipped_count
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(
    images,
    dtype=np.float32
)

y = np.array(
    labels,
    dtype=np.int64
)


print()
print("=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print(
    "Total images:",
    len(X)
)

print(
    "Image shape:",
    X.shape
)


if len(X) == 0:

    raise RuntimeError(
        "No images were loaded."
    )


# ============================================================
# NORMALIZATION
# ============================================================

# No augmentation.
# Only pixel normalization.

X = X / 255.0


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("Total class distribution:")

for class_index, class_name in enumerate(CLASS_NAMES):

    count = np.sum(
        y == class_index
    )

    print(
        f"{class_name:<10}: {count}"
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
# ============================================================

print()
print("=" * 75)
print("CREATING TRAIN / VALIDATION / TEST SPLIT")
print("=" * 75)


# ------------------------------------------------------------
# First split:
# 70% training
# 30% temporary
# ------------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=y
)


# ------------------------------------------------------------
# Second split:
# 15% validation
# 15% test
# ------------------------------------------------------------

X_val, X_test, y_val, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=y_temp
)


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

print(
    "Total      :",
    len(X_train) + len(X_val) + len(X_test)
)


# ============================================================
# SPLIT CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 75)
print("CLASS DISTRIBUTION AFTER SPLIT")
print("=" * 75)


for class_index, class_name in enumerate(CLASS_NAMES):

    train_count = np.sum(
        y_train == class_index
    )

    val_count = np.sum(
        y_val == class_index
    )

    test_count = np.sum(
        y_test == class_index
    )

    print(
        f"{class_name:<10}"
        f" Train={train_count:>6}"
        f" Val={val_count:>6}"
        f" Test={test_count:>6}"
    )


# ============================================================
# BUILD VGG16
# ============================================================

def build_vgg16(
    input_shape=(32, 32, 3),
    num_classes=4
):

    inputs = Input(
        shape=input_shape,
        name="input_layer"
    )


    # ========================================================
    # BLOCK 1
    # ========================================================

    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        name="block1_conv1"
    )(inputs)

    x = BatchNormalization(
        name="block1_bn1"
    )(x)

    x = Conv2D(
        64,
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


    # ========================================================
    # BLOCK 2
    # ========================================================

    x = Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        name="block2_conv1"
    )(x)

    x = BatchNormalization(
        name="block2_bn1"
    )(x)

    x = Conv2D(
        128,
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


    # ========================================================
    # BLOCK 3
    # ========================================================

    x = Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        name="block3_conv1"
    )(x)

    x = BatchNormalization(
        name="block3_bn1"
    )(x)

    x = Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        name="block3_conv2"
    )(x)

    x = BatchNormalization(
        name="block3_bn2"
    )(x)

    x = Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        name="block3_conv3"
    )(x)

    x = BatchNormalization(
        name="block3_bn3"
    )(x)

    x = MaxPooling2D(
        (2, 2),
        name="block3_pool"
    )(x)


    # ========================================================
    # BLOCK 4
    # ========================================================

    x = Conv2D(
        512,
        (3, 3),
        padding="same",
        activation="relu",
        name="block4_conv1"
    )(x)

    x = BatchNormalization(
        name="block4_bn1"
    )(x)

    x = Conv2D(
        512,
        (3, 3),
        padding="same",
        activation="relu",
        name="block4_conv2"
    )(x)

    x = BatchNormalization(
        name="block4_bn2"
    )(x)

    x = Conv2D(
        512,
        (3, 3),
        padding="same",
        activation="relu",
        name="block4_conv3"
    )(x)

    x = BatchNormalization(
        name="block4_bn3"
    )(x)

    x = MaxPooling2D(
        (2, 2),
        name="block4_pool"
    )(x)


    # ========================================================
    # BLOCK 5
    # ========================================================

    x = Conv2D(
        512,
        (3, 3),
        padding="same",
        activation="relu",
        name="block5_conv1"
    )(x)

    x = BatchNormalization(
        name="block5_bn1"
    )(x)

    x = Conv2D(
        512,
        (3, 3),
        padding="same",
        activation="relu",
        name="block5_conv2"
    )(x)

    x = BatchNormalization(
        name="block5_bn2"
    )(x)

    x = Conv2D(
        512,
        (3, 3),
        padding="same",
        activation="relu",
        name="block5_conv3"
    )(x)

    x = BatchNormalization(
        name="block5_bn3"
    )(x)


    # ========================================================
    # CLASSIFIER
    # ========================================================

    x = Flatten(
        name="flatten"
    )(x)

    x = Dense(
        512,
        activation="relu",
        name="fc1"
    )(x)

    x = BatchNormalization(
        name="fc1_bn"
    )(x)

    x = Dropout(
        0.5,
        name="dropout1"
    )(x)

    x = Dense(
        512,
        activation="relu",
        name="fc2"
    )(x)

    x = BatchNormalization(
        name="fc2_bn"
    )(x)

    x = Dropout(
        0.5,
        name="dropout2"
    )(x)


    # ========================================================
    # OUTPUT
    # ========================================================

    outputs = Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)


    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="VGG16_Banana_NDA"
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

model = build_vgg16(
    input_shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    ),
    num_classes=NUM_CLASSES
)


print()
print("=" * 75)
print("VGG16 MODEL")
print("=" * 75)

model.summary()


# ============================================================
# COMPILE
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
# CALLBACKS
# ============================================================

best_model_path = os.path.join(
    RESULTS_DIR,
    "VGG16_NDA_Best.keras"
)

csv_path = os.path.join(
    RESULTS_DIR,
    "VGG16_NDA_history.csv"
)


callbacks = [

    ModelCheckpoint(

        best_model_path,

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

        patience=8,

        restore_best_weights=True,

        verbose=1
    ),


    CSVLogger(

        csv_path,

        append=False
    )
]


# ============================================================
# DATA AUGMENTATION CHECK
# ============================================================

print()
print("=" * 75)
print("DATA AUGMENTATION CHECK")
print("=" * 75)

print(
    "Data augmentation:",
    DATA_AUGMENTATION
)

print()
print("No rotation")
print("No zoom")
print("No horizontal flip")
print("No vertical flip")
print("No translation")
print("No shear")
print("No brightness modification")
print("No contrast augmentation")

print()
print(
    "Training will use the original images only."
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 75)
print("STARTING VGG16 WITHOUT DATA AUGMENTATION")
print("=" * 75)


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
print(
    "Training time:",
    f"{training_time / 60:.2f} minutes"
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 75)
print("LOADING BEST VGG16 MODEL")
print("=" * 75)


model = tf.keras.models.load_model(
    best_model_path
)


print(
    "Best model loaded successfully."
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 75)
print("FINAL VALIDATION EVALUATION")
print("=" * 75)


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
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)


test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=1
)


print()
print(
    f"Test Loss     : "
    f"{test_loss:.6f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 75)
print("GENERATING TEST PREDICTIONS")
print("=" * 75)


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
# METRICS
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
print("=" * 75)
print("FINAL PERFORMANCE")
print("=" * 75)

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
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=np.arange(NUM_CLASSES)
)


print()
print("=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(cm)


cm_accuracy = (
    np.trace(cm)
    /
    np.sum(cm)
)


print()
print(
    "Confusion Matrix Accuracy:",
    f"{cm_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    labels=np.arange(NUM_CLASSES),

    target_names=CLASS_NAMES,

    digits=4,

    zero_division=0
)


print()
print("=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(report)


# Save classification report
report_path = os.path.join(
    RESULTS_DIR,
    "VGG16_NDA_Classification_Report.txt"
)


with open(
    report_path,
    "w"
) as file:

    file.write(report)


# ============================================================
# CONFUSION MATRIX IMAGE
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
    "Banana Ripeness - VGG16\n"
    "Without Data Augmentation",
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

            str(cm[i, j]),

            ha="center",

            va="center",

            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),

            fontsize=13
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
    "VGG16_NDA_Confusion_Matrix.png"
)


plt.savefig(

    cm_path,

    dpi=300,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(12, 7)
)


plt.plot(

    history.history["accuracy"],

    marker="o",

    linewidth=2,

    label="Training Accuracy"
)


plt.plot(

    history.history["val_accuracy"],

    marker="o",

    linewidth=2,

    label="Validation Accuracy"
)


plt.title(
    "Banana Ripeness - VGG16 Accuracy\n"
    "Without Data Augmentation",
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
    "VGG16_NDA_Accuracy.png"
)


plt.savefig(

    accuracy_path,

    dpi=300,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(12, 7)
)


plt.plot(

    history.history["loss"],

    marker="o",

    linewidth=2,

    label="Training Loss"
)


plt.plot(

    history.history["val_loss"],

    marker="o",

    linewidth=2,

    label="Validation Loss"
)


plt.title(
    "Banana Ripeness - VGG16 Loss\n"
    "Without Data Augmentation",
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
    "VGG16_NDA_Loss.png"
)


plt.savefig(

    loss_path,

    dpi=300,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# BEST TRAINING RESULTS
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
print("=" * 75)
print("BEST TRAINING RESULTS")
print("=" * 75)

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

final_model_path = os.path.join(
    RESULTS_DIR,
    "VGG16_NDA_Final.keras"
)


model.save(
    final_model_path
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "VGG16_NDA_Final_Metrics.txt"
)


with open(
    metrics_path,
    "w"
) as file:

    file.write(
        "BANANA RIPENESS - VGG16\n"
    )

    file.write(
        "WITHOUT DATA AUGMENTATION\n\n"
    )

    file.write(
        f"Total Images: "
        f"{len(X)}\n"
    )

    file.write(
        f"Training Images: "
        f"{len(X_train)}\n"
    )

    file.write(
        f"Validation Images: "
        f"{len(X_val)}\n"
    )

    file.write(
        f"Test Images: "
        f"{len(X_test)}\n\n"
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
        f"Epochs Configured: "
        f"{EPOCHS}\n\n"
    )

    file.write(
        f"Validation Loss: "
        f"{val_loss:.6f}\n"
    )

    file.write(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        f"Test Loss: "
        f"{test_loss:.6f}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: "
        f"{precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall: "
        f"{recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score: "
        f"{f1 * 100:.2f}%\n"
    )

    file.write(
        f"Confusion Matrix Accuracy: "
        f"{cm_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Data Augmentation: DISABLED\n"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "VGG16_NDA_Summary.txt"
)


with open(
    summary_path,
    "w"
) as file:

    file.write(
        "VGG16 - BANANA RIPENESS\n"
    )

    file.write(
        "WITHOUT DATA AUGMENTATION\n"
    )

    file.write(
        "====================================\n\n"
    )

    file.write(
        f"Total Images       : {len(X)}\n"
    )

    file.write(
        f"Training Images    : {len(X_train)}\n"
    )

    file.write(
        f"Validation Images  : {len(X_val)}\n"
    )

    file.write(
        f"Test Images        : {len(X_test)}\n\n"
    )

    file.write(
        f"Test Accuracy      : "
        f"{test_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Test Loss          : "
        f"{test_loss:.6f}\n"
    )

    file.write(
        f"Precision          : "
        f"{precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall             : "
        f"{recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score           : "
        f"{f1 * 100:.2f}%\n"
    )

    file.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Best Validation Epoch: "
        f"{best_val_epoch}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 75)
print("VGG16 WITHOUT DATA AUGMENTATION - COMPLETE")
print("=" * 75)

print()
print("DATA AUGMENTATION: DISABLED")

print()
print("Final test accuracy:")
print(
    f"{test_accuracy * 100:.2f}%"
)

print()
print("Final test loss:")
print(
    f"{test_loss:.6f}"
)

print()
print("Precision:")
print(
    f"{precision * 100:.2f}%"
)

print()
print("Recall:")
print(
    f"{recall * 100:.2f}%"
)

print()
print("F1 Score:")
print(
    f"{f1 * 100:.2f}%"
)

print()
print("Images used:")
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

print()
print("Results folder:")
print(
    os.path.abspath(
        RESULTS_DIR
    )
)

print()
print("Files created:")

print(
    "1. VGG16_NDA_Best.keras"
)

print(
    "2. VGG16_NDA_Final.keras"
)

print(
    "3. VGG16_NDA_history.csv"
)

print(
    "4. VGG16_NDA_Accuracy.png"
)

print(
    "5. VGG16_NDA_Loss.png"
)

print(
    "6. VGG16_NDA_Confusion_Matrix.png"
)

print(
    "7. VGG16_NDA_Classification_Report.txt"
)

print(
    "8. VGG16_NDA_Final_Metrics.txt"
)

print(
    "9. VGG16_NDA_Summary.txt"
)

print()
print("=" * 75)
print("TRAINING FINISHED")
print("=" * 75)