''' from __future__ import print_function
import numpy as np
import os
import time,keras
from keras.callbacks import CSVLogger
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, ZeroPadding2D,MaxPooling2D, BatchNormalization,Input
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint
from keras.utils import np_utils
import re
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import h5py
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

PATH = "/home/shabbeer/CV_Course/Shabbeer/crchistophenotypes32_32"
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
        print(img)
        img_path = data_path + '/' + dataset + '/' + img
        img = image.load_img(img_path, target_size=(33,33))
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

num_classes = 4
num_of_samples = img_data.shape[0]
print("sample", num_of_samples)
labels = np.ones((num_of_samples,), dtype='int64')
labels[0:7722] = 0
labels[7722:13434] = 1
labels[13434:20225] = 2
labels[20225:] = 3
names = ['epithelial', 'fibroblast', 'inflammatory', 'others']

Y = np_utils.to_categorical(labels, num_classes)
x, y = shuffle(img_data, Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

model = Sequential()
model.add(ZeroPadding2D(1, input_shape=(33,33,3)))
model.add(Conv2D(96, (5,5), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(256, (5, 5), activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.3, name='dropout_1'))
model.add(MaxPooling2D(pool_size=(2, 2), name='block1_maxpooling2'))
model.add(Conv2D(384, (3, 3), activation='relu', name='block1_conv3', padding='same'))

model.add(Conv2D(384, (3, 3), activation='relu', name='block1_conv4', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', name='block1_conv5', padding='same'))
model.add(Dropout(0.4, name='dropout_2'))
model.add(Flatten(name='flatten'))
# model.add(Dense(512, activation='relu', name='fc'))
# model.add(Dropout(0.5, name='dropout_3'))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dropout(0.5, name='dropout_4'))
model.add(Dense(4096, activation='relu', name='fc1'))
model.add(Dropout(0.5, name='dropout_5'))
model.add(Dense(4, activation='softmax', name='predictions'))
model.summary()

batch_size = 32
num_classes = 4
epochs = 500
data_augmentation = False

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
csv_logger = CSVLogger('/home/ravi/Desktop/Shabbeer/AlexNet_acc_DA.csv')

# save_dir = os.path.join(os.getcwd(), 'saved_models')
adam = Adam(lr=0.00006, beta_1=0.9, beta_2=0.99, epsilon=None, decay=1e-6, amsgrad=False)

# Let's train the model using Adam
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

import time
start = time.time()

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              shuffle=True, callbacks=[lr_reducer,csv_logger])
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)  # randomly flip images

    datagen.fit(X_train)
    history = model.fit_generator(datagen.flow(X_train, y_train,
                                            batch_size=batch_size),
                               epochs=epochs,
                               validation_data=(X_test, y_test), callbacks=[lr_reducer, csv_logger])

print('------------Training time is seconds:%s',time.time()-start)

scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
import matplotlib.pyplot as plt
print('Max test accuracy:', max(history.history['val_acc']))
print(history.history.keys())
# # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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

import os

# Disable oneDNN messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    ModelCheckpoint,
    EarlyStopping
)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

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

IMG_HEIGHT = 128
IMG_WIDTH = 128

BATCH_SIZE = 32

EPOCHS = 40

NUM_CLASSES = 4

RANDOM_STATE = 42


# ============================================================
# PRINT INFORMATION
# ============================================================

print("\n==============================================")
print("ENHANCED ALEXNET")
print("==============================================")

print("TensorFlow:", tf.__version__)

print("Dataset:", PATH)

print("Image size:", IMG_HEIGHT, "x", IMG_WIDTH)

print("Classes:", NUM_CLASSES)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(PATH):

    raise FileNotFoundError(
        "\nDataset not found:\n" + PATH
    )


# ============================================================
# SORT FILES
# ============================================================

def sorted_alphanumeric(data):

    convert = lambda text: (
        int(text)
        if text.isdigit()
        else text.lower()
    )

    alphanum_key = lambda key: [
        convert(c)
        for c in re.split(
            "([0-9]+)",
            key
        )
    ]

    return sorted(
        data,
        key=alphanum_key
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    dataset_path,
    img_size=(128, 128)
):

    class_names = []

    for name in sorted_alphanumeric(
        os.listdir(dataset_path)
    ):

        folder_path = os.path.join(
            dataset_path,
            name
        )

        if os.path.isdir(folder_path):

            class_names.append(name)


    print("\n==============================================")
    print("CLASSES")
    print("==============================================")

    for i, name in enumerate(class_names):

        print(
            i,
            "->",
            name
        )


    if len(class_names) != NUM_CLASSES:

        raise ValueError(
            f"\nExpected {NUM_CLASSES} classes "
            f"but found {len(class_names)}"
        )


    img_data_list = []

    labels_list = []


    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )


    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    for class_index, class_name in enumerate(
        class_names
    ):

        dir_path = os.path.join(
            dataset_path,
            class_name
        )


        files = [
            f
            for f in sorted_alphanumeric(
                os.listdir(dir_path)
            )
            if f.lower().endswith(
                valid_extensions
            )
        ]


        print(
            f"{class_name}: "
            f"{len(files)} images"
        )


        for img_name in files:

            img_path = os.path.join(
                dir_path,
                img_name
            )


            try:

                img = image.load_img(
                    img_path,
                    target_size=img_size
                )


                x = image.img_to_array(
                    img
                )


                x = preprocess_input(
                    x
                )


                img_data_list.append(
                    x
                )


                labels_list.append(
                    class_index
                )


            except Exception as e:

                print(
                    "Skipped:",
                    img_path
                )


    X = np.array(
        img_data_list,
        dtype="float32"
    )


    labels = np.array(
        labels_list,
        dtype="int64"
    )


    print("\nTotal images:", len(X))

    print("Dataset shape:", X.shape)


    Y = tf.keras.utils.to_categorical(
        labels,
        NUM_CLASSES
    )


    return X, Y, labels, class_names


# ============================================================
# LOAD DATA
# ============================================================

X, Y, labels, class_names = load_dataset(
    PATH,
    img_size=(
        IMG_HEIGHT,
        IMG_WIDTH
    )
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n==============================================")
print("DATA SPLIT")
print("==============================================")


# First:
# 80% training
# 20% temporary

X_train, X_temp, y_train, y_temp, labels_train, labels_temp = train_test_split(

    X,
    Y,
    labels,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=labels
)


# Then:
# 10% validation
# 10% test

X_val, X_test, y_val, y_test, labels_val, labels_test = train_test_split(

    X_temp,
    y_temp,
    labels_temp,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=labels_temp
)


print(
    "Training:",
    len(X_train)
)

print(
    "Validation:",
    len(X_val)
)

print(
    "Testing:",
    len(X_test)
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(
        labels_train
    ),

    y=labels_train
)


class_weights = {

    i: float(weight)

    for i, weight in enumerate(
        class_weights_array
    )

}


print("\n==============================================")
print("CLASS WEIGHTS")
print("==============================================")

print(class_weights)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = Sequential([

    RandomFlip(
        "horizontal"
    ),

    RandomRotation(
        0.10
    ),

    RandomZoom(
        0.15
    ),

    RandomContrast(
        0.10
    )

], name="AlexNet_Augmentation")


# ============================================================
# BUILD ENHANCED ALEXNET
# ============================================================

def build_alexnet(
    input_shape=(128, 128, 3),
    num_classes=4
):

    model = Sequential(

        [

            Input(
                shape=input_shape
            ),


            # -----------------------------
            # AUGMENTATION
            # -----------------------------

            data_augmentation,


            # -----------------------------
            # BLOCK 1
            # -----------------------------

            Conv2D(
                64,
                kernel_size=(
                    5,
                    5
                ),
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),

            MaxPooling2D(
                pool_size=(
                    2,
                    2
                )
            ),


            # -----------------------------
            # BLOCK 2
            # -----------------------------

            Conv2D(
                128,
                kernel_size=(
                    3,
                    3
                ),
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),

            MaxPooling2D(
                pool_size=(
                    2,
                    2
                )
            ),


            # -----------------------------
            # BLOCK 3
            # -----------------------------

            Conv2D(
                256,
                kernel_size=(
                    3,
                    3
                ),
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),


            Conv2D(
                256,
                kernel_size=(
                    3,
                    3
                ),
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),


            MaxPooling2D(
                pool_size=(
                    2,
                    2
                )
            ),


            # -----------------------------
            # BLOCK 4
            # -----------------------------

            Conv2D(
                384,
                kernel_size=(
                    3,
                    3
                ),
                padding="same",
                activation="relu"
            ),

            BatchNormalization(),


            MaxPooling2D(
                pool_size=(
                    2,
                    2
                )
            ),


            # -----------------------------
            # CLASSIFIER
            # -----------------------------

            GlobalAveragePooling2D(),


            Dense(
                512,
                activation="relu"
            ),

            BatchNormalization(),

            Dropout(
                0.5
            ),


            Dense(
                256,
                activation="relu"
            ),

            Dropout(
                0.3
            ),


            Dense(
                num_classes,
                activation="softmax"
            )

        ],

        name="Enhanced_AlexNet"

    )


    return model


# ============================================================
# BUILD MODEL
# ============================================================

print("\n==============================================")
print("BUILDING ALEXNET")
print("==============================================")


model = build_alexnet(

    input_shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    ),

    num_classes=NUM_CLASSES
)


model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.0001
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

        "AlexNet_best.keras",

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1

    ),


    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=3,

        min_lr=1e-7,

        verbose=1

    ),


    EarlyStopping(

        monitor="val_loss",

        patience=8,

        restore_best_weights=True,

        verbose=1

    ),


    CSVLogger(

        "AlexNet_history.csv"

    )

]


# ============================================================
# TRAIN
# ============================================================

print("\n==============================================")
print("TRAINING ALEXNET")
print("==============================================")


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

    class_weight=class_weights,

    shuffle=True,

    callbacks=callbacks,

    verbose=1

)


training_time = (
    time.time()
    -
    start_time
)


print(
    "\nTraining completed in:",
    round(
        training_time,
        2
    ),
    "seconds"
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n==============================================")
print("FINAL TEST RESULTS")
print("==============================================")


test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=1

)


print(
    "\nTest Loss:",
    round(
        test_loss,
        4
    )
)


print(
    "Test Accuracy:",
    round(
        test_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==============================================")
print("CLASSIFICATION REPORT")
print("==============================================")


predictions = model.predict(

    X_test,

    verbose=1

)


predicted_classes = np.argmax(

    predictions,

    axis=1

)


print(

    classification_report(

        labels_test,

        predicted_classes,

        target_names=class_names,

        digits=4

    )

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n==============================================")
print("CONFUSION MATRIX")
print("==============================================")


cm = confusion_matrix(

    labels_test,

    predicted_classes

)


print(cm)


plt.figure(
    figsize=(7, 6)
)


plt.imshow(cm)


plt.title(
    "Banana / Dataset Classification - AlexNet"
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

            cm[i, j],

            ha="center",

            va="center"

        )


plt.tight_layout()


plt.savefig(

    "AlexNet_confusion_matrix.png",

    dpi=300

)


plt.show()


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
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
    "AlexNet Training and Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    "AlexNet_accuracy.png",

    dpi=300

)


plt.show()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
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
    "AlexNet Training and Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    "AlexNet_loss.png",

    dpi=300

)


plt.show()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(

    "AlexNet_FINAL.keras"

)


print("\n==============================================")
print("ALEXNET COMPLETE")
print("==============================================")


print(
    "\nSaved files:"
)

print(
    "AlexNet_best.keras"
)

print(
    "AlexNet_FINAL.keras"
)

print(
    "AlexNet_history.csv"
)

print(
    "AlexNet_accuracy.png"
)

print(
    "AlexNet_loss.png"
)

print(
    "AlexNet_confusion_matrix.png"
)