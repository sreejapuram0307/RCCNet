import os

# Disable oneDNN messages BEFORE importing TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    ModelCheckpoint,
    EarlyStopping
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# ============================================================
# SETTINGS
# ============================================================

PATH = r"C:/Users/E028.6/Downloads/Banana_Dataset"

IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 4

BATCH_SIZE = 32
EPOCHS = 20

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(dataset_path, img_size=(224, 224)):

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}"
        )

    class_names = sorted([
        folder for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ])

    print("\n========================================")
    print("BANANA RIPENESS DATASET")
    print("========================================")
    print("Classes found:", class_names)

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"but found {len(class_names)} classes."
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

    for class_index, class_name in enumerate(class_names):

        dir_path = os.path.join(dataset_path, class_name)

        image_files = [
            f for f in os.listdir(dir_path)
            if f.lower().endswith(valid_extensions)
        ]

        print(
            f"Loading {len(image_files)} images "
            f"from: {class_name}"
        )

        for img_name in image_files:

            img_path = os.path.join(dir_path, img_name)

            try:

                img = image.load_img(
                    img_path,
                    target_size=img_size
                )

                x = image.img_to_array(img)

                x = preprocess_input(x)

                img_data_list.append(x)
                labels_list.append(class_index)

            except Exception as e:

                print(
                    f"Skipped image: {img_path}"
                )

    X = np.array(
        img_data_list,
        dtype=np.float32
    )

    labels = np.array(
        labels_list,
        dtype=np.int64
    )

    if len(X) == 0:
        raise ValueError(
            "No images were loaded from the dataset."
        )

    print("\nDataset loaded successfully.")
    print("Dataset shape:", X.shape)
    print("Number of images:", len(X))
    print("Number of classes:", len(class_names))

    Y = to_categorical(
        labels,
        num_classes=NUM_CLASSES
    )

    return X, Y, class_names


# ============================================================
# BUILD RESNET50 MODEL
# ============================================================

def build_resnet50(
    input_shape=(224, 224, 3),
    num_classes=4
):

    inputs = Input(shape=input_shape)

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )

    # First training stage:
    # keep pretrained ResNet50 frozen
    base_model.trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        512,
        activation="relu"
    )(x)

    x = Dropout(0.5)(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Banana_Ripeness_ResNet50"
    )

    return model


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("BANANA RIPENESS - RESNET50")
    print("========================================")

    print("TensorFlow version:", tf.__version__)
    print("Dataset:", PATH)
    print("Image size:", IMG_HEIGHT, "x", IMG_WIDTH)
    print("Classes:", NUM_CLASSES)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    X, Y, class_names = load_dataset(
        PATH,
        img_size=(IMG_HEIGHT, IMG_WIDTH)
    )

    # --------------------------------------------------------
    # 2. Shuffle dataset
    # --------------------------------------------------------

    X, Y = shuffle(
        X,
        Y,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # 3. Train/Test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=np.argmax(Y, axis=1)
    )

    print("\nTraining images:", len(X_train))
    print("Testing images :", len(X_test))

    # --------------------------------------------------------
    # 4. Build model
    # --------------------------------------------------------

    model = build_resnet50(
        input_shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        ),
        num_classes=NUM_CLASSES
    )

    model.summary()

    # --------------------------------------------------------
    # 5. Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=Adam(
            learning_rate=0.0001
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------------------------------------------
    # 6. Callbacks
    # --------------------------------------------------------

    callbacks = [

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),

        ModelCheckpoint(
            "Banana_ResNet50_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        CSVLogger(
            "Banana_ResNet50_history.csv"
        )
    ]

    # --------------------------------------------------------
    # 7. Train
    # --------------------------------------------------------

    print("\n========================================")
    print("STARTING TRAINING")
    print("========================================\n")

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.20,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    print(
        f"\nTraining completed in "
        f"{training_time / 60:.2f} minutes."
    )

    # --------------------------------------------------------
    # 8. Evaluate
    # --------------------------------------------------------

    print("\n========================================")
    print("FINAL TEST EVALUATION")
    print("========================================")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print(
        f"\nTest Loss     : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy : {test_accuracy:.4f}"
    )

    print(
        f"Test Accuracy : {test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 9. Save final model
    # --------------------------------------------------------

    model.save(
        "Banana_ResNet50_final.keras"
    )

    # --------------------------------------------------------
    # 10. Plot accuracy
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(
        "Banana Ripeness - ResNet50 Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "Banana_ResNet50_accuracy.png",
        dpi=300
    )

    plt.show()

    # --------------------------------------------------------
    # 11. Plot loss
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "Banana Ripeness - ResNet50 Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "Banana_ResNet50_loss.png",
        dpi=300
    )

    plt.show()

    print("\n========================================")
    print("TRAINING FINISHED")
    print("========================================")

    print("\nFiles created:")

    print("1. Banana_ResNet50_best.keras")
    print("2. Banana_ResNet50_final.keras")
    print("3. Banana_ResNet50_history.csv")
    print("4. Banana_ResNet50_accuracy.png")
    print("5. Banana_ResNet50_loss.png")

