'''from __future__ import print_function
import numpy as np
import os
import time
from keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import h5py
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger
import keras
from keras.preprocessing.image import ImageDataGenerator

PATH = "C:/Users/E028.6/Downloads/Banana/Banana/Test Set"
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


# Define data path
data_path = PATH
data_dir_list = sorted_alphanumeric(os.listdir(data_path))
print(data_dir_list)

img_data_list = []

for dataset in sorted_alphanumeric(data_dir_list):
    img_list = sorted_alphanumeric(os.listdir(data_path + '/' + dataset))
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        # Add these two lines to skip non-image files:
        if not img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        print(img)
        img_path = data_path + '/' + dataset + '/' + img
        img = load_img(img_path, target_size=(32,32))
        x = img_to_array(img)
        # ... rest of the loop
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

num_classes = 4
num_of_samples = img_data.shape[0]
print("sample", num_of_samples)
labels = np.ones((num_of_samples,), dtype='int64')
labels[0:7722] = 0
labels[7722:13434] = 1
labels[13434:20225] = 2
labels[20225:] = 3

names = ['Unripe', 'Ripe', 'Overripe', 'Rotten']

Y = np_utils.to_categorical(labels, num_classes)
x, y = shuffle(img_data, Y, random_state=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32,32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))
model.summary()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=6, min_lr=0.5e-15)
csv_logger = CSVLogger('./RCCNet_F1score_NoDA_dummy.csv', append=True, separator=';') 

adam = Adam(lr=0.00006, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1.0e-6,amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1])


batch_size =64
data_augmentation = True
epochs = 100

import time
start_time = time.time()

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True, callbacks=[lr_reducer,csv_logger])
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
                                  validation_data=(x_test, y_test), callbacks=[lr_reducer, csv_logger])

print("---  Training time in seconds ---%s " % (time.time() - start_time))
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
import matplotlib.pyplot as plt
print('Max Test accuracy:', max(history.history['val_f1']))
# # visualizing losses and accuracy
print(history.history.keys())
# # summarize history for accuracy
plt.plot(history.history['f1'])
plt.plot(history.history['val_f1'])
plt.title('model accuracy')
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
plt.show() '''

# ============================================================
# RCCNet - BANANA RIPENESS CLASSIFICATION
# WITH DATA AUGMENTATION
# ============================================================

import os

# Disable oneDNN messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import cv2
import numpy as np
import pandas as pd
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
# GPU CHECK
# ============================================================

print()
print("=" * 75)
print("RCCNet - GPU CHECK")
print("=" * 75)

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("GPU detected:")

    for gpu in gpus:
        print("   ", gpu)

    try:

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

        print("GPU memory growth: ENABLED")

    except RuntimeError as error:

        print("GPU memory setting message:")
        print(error)

else:

    print("No GPU detected.")
    print("Training will use CPU.")

print("=" * 75)


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = r"C:\Users\E028.6\Downloads\Banana_Dataset"

IMG_SIZE = (32, 32)

IMG_HEIGHT = 32
IMG_WIDTH = 32

NUM_CLASSES = 4

BATCH_SIZE = 64

EPOCHS = 30

RANDOM_STATE = 42

RESULTS_DIR = "RCCNet_DA_Results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


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
print("RCCNet - BANANA RIPENESS CLASSIFICATION")
print("WITH DATA AUGMENTATION")
print("=" * 75)

print("Dataset:", DATASET_PATH)
print("Image size:", IMG_WIDTH, "x", IMG_HEIGHT)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Classes:", CLASS_NAMES)
print("Data augmentation: TRUE")

print("=" * 75)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset path does not exist:\n{DATASET_PATH}"
    )


for class_name in CLASS_NAMES:

    folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(folder):

        raise FileNotFoundError(
            f"\nMissing class folder:\n{folder}"
        )


print("\nAll dataset folders found successfully.")


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
# LOAD DATASET
# ============================================================

images = []
labels = []

print()
print("=" * 75)
print("LOADING DATASET")
print("=" * 75)


for class_index, class_name in enumerate(CLASS_NAMES):

    folder_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    files = [

        filename

        for filename in os.listdir(folder_path)

        if filename.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ]

    files.sort()

    loaded = 0
    skipped = 0

    print()
    print(f"Loading class: {class_name}")

    for filename in files:

        image_path = os.path.join(
            folder_path,
            filename
        )

        try:

            img = cv2.imread(
                image_path
            )

            if img is None:

                skipped += 1
                continue

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            img = cv2.resize(
                img,
                IMG_SIZE
            )

            images.append(
                img
            )

            labels.append(
                class_index
            )

            loaded += 1

        except Exception as error:

            skipped += 1

            print(
                "Skipped:",
                filename,
                error
            )

    print(
        "Loaded :",
        loaded
    )

    print(
        "Skipped:",
        skipped
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


# ============================================================
# NORMALIZATION
# ============================================================

X = X / 255.0


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("Class distribution:")

for class_index, class_name in enumerate(CLASS_NAMES):

    count = np.sum(
        y == class_index
    )

    print(
        f"{class_name:<12}: {count}"
    )


# ============================================================
# 70 / 15 / 15 SPLIT
# ============================================================

print()
print("=" * 75)
print("CREATING TRAIN / VALIDATION / TEST SPLIT")
print("=" * 75)


# First split:
# 70% training
# 30% temporary

X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=y
)


# Second split:
# 15% validation
# 15% testing

X_val, X_test, y_val, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=y_temp
)


print()
print(
    "Training images   :",
    len(X_train)
)

print(
    "Validation images :",
    len(X_val)
)

print(
    "Testing images    :",
    len(X_test)
)

print(
    "Total images      :",
    len(X_train) +
    len(X_val) +
    len(X_test)
)


# ============================================================
# CLASS DISTRIBUTION AFTER SPLIT
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
        f"{class_name:<12}"
        f"Train={train_count:<6}"
        f"Val={val_count:<6}"
        f"Test={test_count:<6}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomFlip(
            mode="horizontal"
        ),

        tf.keras.layers.RandomRotation(
            factor=0.10
        ),

        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        tf.keras.layers.RandomContrast(
            factor=0.10
        )

    ],

    name="RCCNet_Data_Augmentation"
)


print()
print("=" * 75)
print("DATA AUGMENTATION")
print("=" * 75)

print("Horizontal flip     : ENABLED")
print("Rotation            : ENABLED")
print("Zoom                : ENABLED")
print("Translation         : ENABLED")
print("Contrast            : ENABLED")


# ============================================================
# BUILD RCCNet
# ============================================================

def build_rccnet(
    input_shape=(32, 32, 3),
    num_classes=4
):

    inputs = Input(
        shape=input_shape,
        name="input"
    )


    # ========================================================
    # DATA AUGMENTATION
    # ========================================================

    x = data_augmentation(
        inputs
    )


    # ========================================================
    # CONVOLUTIONAL BLOCK 1
    # ========================================================

    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        pool_size=(2, 2)
    )(x)


    # ========================================================
    # CONVOLUTIONAL BLOCK 2
    # ========================================================

    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        pool_size=(2, 2)
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
    # OUTPUT
    # ========================================================

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)


    model = Model(

        inputs=inputs,

        outputs=outputs,

        name="RCCNet_Banana_DA"
    )


    return model


# ============================================================
# CREATE MODEL
# ============================================================

model = build_rccnet(

    input_shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    ),

    num_classes=NUM_CLASSES
)


print()
print("=" * 75)
print("RCCNet MODEL")
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
# CALLBACK PATHS
# ============================================================

best_model_path = os.path.join(

    RESULTS_DIR,

    "RCCNet_DA_Best.keras"
)


history_csv_path = os.path.join(

    RESULTS_DIR,

    "RCCNet_DA_History.csv"
)


final_model_path = os.path.join(

    RESULTS_DIR,

    "RCCNet_DA_Final.keras"
)


# ============================================================
# CALLBACKS
# ============================================================

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

        patience=7,

        restore_best_weights=True,

        verbose=1
    ),

    CSVLogger(

        history_csv_path,

        append=False
    )
]


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 75)
print("STARTING RCCNet DATA AUGMENTATION TRAINING")
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
    - start_time
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
print("LOADING BEST RCCNet DA MODEL")
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
print("VALIDATION EVALUATION")
print("=" * 75)


val_loss, val_accuracy = model.evaluate(

    X_val,

    y_val,

    verbose=1
)


print()
print(
    f"Validation Loss     : {val_loss:.6f}"
)

print(
    f"Validation Accuracy : "
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 75)
print("TEST EVALUATION")
print("=" * 75)


test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=1
)


print()
print(
    f"Test Loss     : {test_loss:.6f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# TEST PREDICTIONS
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
print("FINAL TEST METRICS")
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

    labels=np.arange(
        NUM_CLASSES
    )
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
    f"Confusion Matrix Accuracy: "
    f"{cm_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
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
print("=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(report)


report_path = os.path.join(

    RESULTS_DIR,

    "RCCNet_DA_Classification_Report.txt"
)


with open(
    report_path,
    "w"
) as file:

    file.write(
        report
    )


# ============================================================
# CONFUSION MATRIX PLOT
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

    "RCCNet - Banana Ripeness\n"
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


cm_path = os.path.join(

    RESULTS_DIR,

    "RCCNet_DA_Confusion_Matrix.png"
)


plt.savefig(

    cm_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# ACCURACY PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)


epochs = range(

    1,

    len(
        history.history["accuracy"]
    ) + 1
)


plt.plot(

    epochs,

    history.history["accuracy"],

    marker="o",

    label="Training Accuracy"
)


plt.plot(

    epochs,

    history.history["val_accuracy"],

    marker="o",

    label="Validation Accuracy"
)


plt.title(

    "RCCNet Banana Ripeness - "
    "Accuracy with Data Augmentation",

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

    "RCCNet_DA_Accuracy.png"
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

    history.history["loss"],

    marker="o",

    label="Training Loss"
)


plt.plot(

    epochs,

    history.history["val_loss"],

    marker="o",

    label="Validation Loss"
)


plt.title(

    "RCCNet Banana Ripeness - "
    "Loss with Data Augmentation",

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

    "RCCNet_DA_Loss.png"
)


plt.savefig(

    loss_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# BEST TRAINING RESULTS
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

    np.argmax(

        history.history[
            "val_accuracy"
        ]

    )

    + 1

)


# ============================================================
# SAVE COMPLETE HISTORY CSV
# ============================================================

history_df = pd.DataFrame({

    "epoch":
        list(
            range(
                1,
                len(
                    history.history[
                        "accuracy"
                    ]
                ) + 1
            )
        ),

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

})


complete_history_path = os.path.join(

    RESULTS_DIR,

    "RCCNet_DA_Complete_History.csv"
)


history_df.to_csv(

    complete_history_path,

    index=False
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics_path = os.path.join(

    RESULTS_DIR,

    "RCCNet_DA_Final_Metrics.txt"
)


with open(
    metrics_path,
    "w"
) as file:

    file.write(
        "RCCNet - BANANA RIPENESS\n"
    )

    file.write(
        "WITH DATA AUGMENTATION\n\n"
    )

    file.write(
        f"TensorFlow Version: "
        f"{tf.__version__}\n"
    )

    file.write(
        f"GPU Detected: "
        f"{bool(gpus)}\n\n"
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
        f"Best Training Accuracy: "
        f"{best_train_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%\n"
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
        f"{best_val_epoch}\n"
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    final_model_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 75)
print("RCCNet DATA AUGMENTATION EXPERIMENT COMPLETE")
print("=" * 75)

print()

print(
    "Data augmentation: ENABLED"
)

print(
    "Training images   :",
    len(X_train)
)

print(
    "Validation images :",
    len(X_val)
)

print(
    "Test images       :",
    len(X_test)
)

print()

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss     : "
    f"{test_loss:.6f}"
)

print(
    f"Precision     : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall        : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score      : "
    f"{f1 * 100:.2f}%"
)

print()

print(
    "Results folder:"
)

print(
    os.path.abspath(
        RESULTS_DIR
    )
)

print()

print("Saved files:")

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

print()
print("=" * 75)
print("DONE")
print("=" * 75)