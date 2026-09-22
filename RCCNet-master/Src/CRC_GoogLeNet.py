'''from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge,Conv2D,MaxPooling2D,BatchNormalization,Concatenate
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import os
import re
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator


# In[2]:
PATH = "/home/hareesh/Desktop/Shabbeer/CV_Course_Shabbeer/crchistophenotypes"
print("PWD", PATH)


from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

data_path = PATH
data_dir_list = sorted_alphanumeric(os.listdir(data_path))
print(data_dir_list)

img_data_list = []

for dataset in sorted_alphanumeric(data_dir_list):
    img_list = sorted_alphanumeric(os.listdir(data_path + '/' + dataset))
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        print(img)
        img_path = data_path + '/' + dataset + '/' + img
        img = image.load_img(img_path, target_size=(32, 32))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        #     x = x/255
        print('Input image shape:', x.shape)
        img_data_list.append(x)

img_data = np.array(img_data_list)
# img_data = img_data.astype('float32')
print(img_data.shape)
img_data = np.rollaxis(img_data, 1, 0)
print(img_data.shape)
img_data = img_data[0]
print(img_data.shape)

# backend
import tensorflow as tf
from keras import backend as k

# Don't pre-allocate memory; allocate as-needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

# Hyperparameters
batch_size = 64
num_classes = 4
epochs = 500

# Load CIFAR10 Data
num_classes = 4
num_of_samples = img_data.shape[0]
print("sample", num_of_samples)
labels = np.ones((num_of_samples,), dtype='int64')
labels[0:7721] = 0
labels[7722:13433] = 1
labels[13434:20224] = 2
labels[20225:] = 3
names = ['epithelial', 'fibroblast', 'inflammatory', 'others']

Y = np_utils.to_categorical(labels, num_classes)
x, y = shuffle(img_data, Y, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]
input = Input(shape=(img_height, img_width, channel,))


Conv2D_1 = Conv2D(64, (3,3), activation='relu', padding='same')(input)
MaxPool2D_1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(Conv2D_1)
BatchNorm_1 = BatchNormalization()(MaxPool2D_1)


tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_1)
tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_1)
tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(BatchNorm_1)
tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

Conv2D_2 = Conv2D(128, (3,3), activation='relu', padding='same')(output)
MaxPool2D_2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(Conv2D_2)
BatchNorm_2 = BatchNormalization()(MaxPool2D_2)


tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_2)
tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_2)
tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(BatchNorm_2)
tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

Conv2D_3 = Conv2D(256, (3,3), activation='relu', padding='same')(output)
MaxPool2D_3 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(Conv2D_3)
BatchNorm_3 = BatchNormalization()(MaxPool2D_3)

tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_3)
tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(BatchNorm_3)
tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(BatchNorm_3)
tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

Output = Flatten()(output)
Output = Dense(num_classes, activation='softmax')(Output)
model = Model(inputs=[input], outputs=[Output])
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
csv_logger = CSVLogger('/home/hareesh/Desktop/Shabbeer/CV_Course_Shabbeer/GoogLeNet_history_DA_acc1.csv', append=True, separator=';')

data_augmentation = False

import time
start = time.time()
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,callbacks=[lr_reducer, csv_logger])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)  # randomly flip images
    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                            batch_size=batch_size),
                               epochs=epochs,
                               validation_data=(x_test, y_test),callbacks=[lr_reducer,csv_logger])

print('------------Training time is seconds:%s',time.time()-start)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
import matplotlib.pyplot as plt
print("Max Test accuracy", max(history.history['val_acc']))
import matplotlib.pyplot  as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()'''

import os

# -------------------------------------------------------------
# IMPORTANT: Set this BEFORE importing TensorFlow
# -------------------------------------------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

# =============================================================
# SETTINGS
# =============================================================

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

# =============================================================
# REPRODUCIBILITY
# =============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("=" * 65)
print("BANANA RIPENESS - GOOGLENET")
print("=" * 65)

print("TensorFlow:", tf.__version__)
print("Dataset:", DATASET_PATH)
print("Image size:", IMG_WIDTH, "x", IMG_HEIGHT)
print("Classes:", CLASS_NAMES)

# =============================================================
# CHECK DATASET
# =============================================================

dataset_path = Path(DATASET_PATH)

if not dataset_path.exists():
    raise FileNotFoundError(
        "\nDataset not found:\n"
        + DATASET_PATH
        + "\n\nPlease check the dataset path."
    )

# Check the four required folders
for class_name in CLASS_NAMES:
    class_folder = dataset_path / class_name

    if not class_folder.exists():
        raise FileNotFoundError(
            f"\nMissing class folder:\n{class_folder}"
        )

print("\nDataset folders found successfully.")

# =============================================================
# COLLECT IMAGE PATHS
# =============================================================

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_paths = []
labels = []

print("\nCounting images...")

for class_index, class_name in enumerate(CLASS_NAMES):

    class_folder = dataset_path / class_name

    files = [
        p for p in class_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]

    print(
        f"{class_name:10s}: {len(files):5d} images"
    )

    for file_path in files:
        image_paths.append(str(file_path))
        labels.append(class_index)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("\nTotal images:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError("No images were found.")

# =============================================================
# TRAIN / VALIDATION / TEST SPLIT
#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
#
# STRATIFIED = each class is represented in all three sets
# =============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=y_temp
)

print("\nDataset split:")
print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Testing    :", len(X_test))

# =============================================================
# SHOW CLASS DISTRIBUTION
# =============================================================

print("\nClass distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    train_count = np.sum(y_train == i)
    val_count = np.sum(y_val == i)
    test_count = np.sum(y_test == i)

    print(
        f"{class_name:10s} "
        f"Train={train_count:4d} "
        f"Val={val_count:4d} "
        f"Test={test_count:4d}"
    )

# =============================================================
# IMAGE LOADING
# =============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMG_HEIGHT, IMG_WIDTH]
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# =============================================================
# DATA AUGMENTATION
#
# Applied ONLY during training
# =============================================================

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

    layers.RandomContrast(
        0.10
    )
], name="data_augmentation")


# =============================================================
# CREATE TF.DATA DATASETS
# =============================================================

def create_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:

        dataset = dataset.map(
            lambda x, y: (
                data_augmentation(x, training=True),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


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

# =============================================================
# GOOGLENET / INCEPTION MODULE
# =============================================================

def inception_module(
    x,
    filters_1x1,
    filters_3x3_reduce,
    filters_3x3,
    filters_5x5_reduce,
    filters_5x5,
    filters_pool_proj
):

    # ---------------------------------------------------------
    # Branch 1: 1x1 convolution
    # ---------------------------------------------------------

    branch1 = layers.Conv2D(
        filters_1x1,
        (1, 1),
        padding="same",
        activation="relu"
    )(x)

    branch1 = layers.BatchNormalization()(branch1)

    # ---------------------------------------------------------
    # Branch 2: 1x1 -> 3x3
    # ---------------------------------------------------------

    branch2 = layers.Conv2D(
        filters_3x3_reduce,
        (1, 1),
        padding="same",
        activation="relu"
    )(x)

    branch2 = layers.BatchNormalization()(branch2)

    branch2 = layers.Conv2D(
        filters_3x3,
        (3, 3),
        padding="same",
        activation="relu"
    )(branch2)

    branch2 = layers.BatchNormalization()(branch2)

    # ---------------------------------------------------------
    # Branch 3: 1x1 -> 5x5
    # ---------------------------------------------------------

    branch3 = layers.Conv2D(
        filters_5x5_reduce,
        (1, 1),
        padding="same",
        activation="relu"
    )(x)

    branch3 = layers.BatchNormalization()(branch3)

    branch3 = layers.Conv2D(
        filters_5x5,
        (5, 5),
        padding="same",
        activation="relu"
    )(branch3)

    branch3 = layers.BatchNormalization()(branch3)

    # ---------------------------------------------------------
    # Branch 4: MaxPool -> 1x1
    # ---------------------------------------------------------

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

    branch4 = layers.BatchNormalization()(branch4)

    # ---------------------------------------------------------
    # Concatenate all branches
    # ---------------------------------------------------------

    output = layers.Concatenate(
        axis=-1
    )([
        branch1,
        branch2,
        branch3,
        branch4
    ])

    return output


# =============================================================
# BUILD GOOGLENET
# =============================================================

def build_googlenet():

    inputs = layers.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        )
    )

    # ---------------------------------------------------------
    # Initial convolution
    # ---------------------------------------------------------

    x = layers.Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        padding="same"
    )(x)

    # ---------------------------------------------------------
    # Conv
    # ---------------------------------------------------------

    x = layers.Conv2D(
        64,
        (1, 1),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        192,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        padding="same"
    )(x)

    # ---------------------------------------------------------
    # Inception 3a
    # ---------------------------------------------------------

    x = inception_module(
        x,
        64,
        96,
        128,
        16,
        32,
        32
    )

    # ---------------------------------------------------------
    # Inception 3b
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Inception 4a
    # ---------------------------------------------------------

    x = inception_module(
        x,
        192,
        96,
        208,
        16,
        48,
        64
    )

    # ---------------------------------------------------------
    # Inception 4b
    # ---------------------------------------------------------

    x = inception_module(
        x,
        160,
        112,
        224,
        24,
        64,
        64
    )

    # ---------------------------------------------------------
    # Inception 4c
    # ---------------------------------------------------------

    x = inception_module(
        x,
        128,
        128,
        256,
        24,
        64,
        64
    )

    # ---------------------------------------------------------
    # Inception 4d
    # ---------------------------------------------------------

    x = inception_module(
        x,
        112,
        144,
        288,
        32,
        64,
        64
    )

    # ---------------------------------------------------------
    # Inception 4e
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Inception 5a
    # ---------------------------------------------------------

    x = inception_module(
        x,
        256,
        160,
        320,
        32,
        128,
        128
    )

    # ---------------------------------------------------------
    # Inception 5b
    # ---------------------------------------------------------

    x = inception_module(
        x,
        384,
        192,
        384,
        48,
        128,
        128
    )

    # ---------------------------------------------------------
    # Classification head
    # ---------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(
        0.40
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="banana_classification"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="GoogleNet_Banana"
    )

    return model


# =============================================================
# CREATE MODEL
# =============================================================

model = build_googlenet()

print("\n")
model.summary()

# =============================================================
# COMPILE
# =============================================================

model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)

# =============================================================
# CALLBACKS
# =============================================================

callbacks = [

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    ModelCheckpoint(
        "GoogLeNet_best.keras",
        monitor="val_accuracy",
        mode="max",
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
        "GoogLeNet_history.csv",
        append=False
    )
]

# =============================================================
# TRAINING
# =============================================================

print("\n")
print("=" * 65)
print("STARTING GOOGLENET TRAINING")
print("=" * 65)

start_time = time.time()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

training_time = time.time() - start_time

print("\nTraining completed.")
print(
    f"Training time: {training_time / 60:.2f} minutes"
)

# =============================================================
# LOAD BEST MODEL
# =============================================================

print("\nLoading best GoogleNet model...")

model = tf.keras.models.load_model(
    "GoogLeNet_best.keras"
)

# =============================================================
# FINAL TEST EVALUATION
# =============================================================

print("\n")
print("=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nFINAL TEST RESULTS")
print("-" * 40)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)

# =============================================================
# PREDICTIONS
# =============================================================

print("\nGenerating test predictions...")

y_probability = model.predict(
    test_dataset,
    verbose=1
)

y_pred = np.argmax(
    y_probability,
    axis=1
)

# =============================================================
# CONFUSION MATRIX
# =============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

print("\nConfusion Matrix:")
print(cm)

# =============================================================
# ACCURACY CALCULATED DIRECTLY FROM CONFUSION MATRIX
# =============================================================

cm_accuracy = (
    np.trace(cm) /
    np.sum(cm)
)

print("\nAccuracy from confusion matrix:")
print(
    f"{cm_accuracy * 100:.2f}%"
)

# =============================================================
# CLASSIFICATION REPORT
# =============================================================

print("\n")
print("=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)

# =============================================================
# CONFUSION MATRIX PLOT
# BLUE THEME
# =============================================================

plt.figure(
    figsize=(8, 7)
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

# Add numbers inside cells
threshold = cm.max() / 2.0

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white"
            if cm[i, j] > threshold
            else "black",
            fontsize=12
        )

plt.tight_layout()

plt.savefig(
    "GoogLeNet_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =============================================================
# ACCURACY GRAPH
# =============================================================

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
    "GoogLeNet_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =============================================================
# LOSS GRAPH
# =============================================================

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
    "GoogLeNet_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =============================================================
# SAVE FINAL RESULTS
# =============================================================

with open(
    "GoogLeNet_final_results.txt",
    "w"
) as f:

    f.write(
        "Banana Ripeness - GoogleNet Results\n"
    )

    f.write(
        "====================================\n\n"
    )

    f.write(
        f"Test Loss: {test_loss:.6f}\n"
    )

    f.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Confusion Matrix Accuracy: "
        f"{cm_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

print("\n")
print("=" * 65)
print("GOOGLENET COMPLETE")
print("=" * 65)

print("\nFiles created:")

print("1. GoogLeNet_best.keras")
print("2. GoogLeNet_history.csv")
print("3. GoogLeNet_confusion_matrix.png")
print("4. GoogLeNet_accuracy.png")
print("5. GoogLeNet_loss.png")
print("6. GoogLeNet_final_results.txt")

print("\nFinal Test Accuracy:")
print(
    f"{test_accuracy * 100:.2f}%"
)

print("\nFinal Test Loss:")
print(
    f"{test_loss:.4f}"
)

print("\nConfusion Matrix Accuracy:")
print(
    f"{cm_accuracy * 100:.2f}%"
)