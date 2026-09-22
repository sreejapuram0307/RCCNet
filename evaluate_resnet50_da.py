import os

# ============================================================
# RESNET50 DATA AUGMENTATION - POST TRAINING EVALUATION
# NO TRAINING
# ============================================================

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

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

MODEL_PATH = r"C:\Users\E028.6\Downloads\RCCNet-master\Banana_ResNet50_best.keras"

RESULTS_DIR = r"C:\Users\E028.6\Downloads\RCCNet-master\ResNet50_DA_Evaluation"

IMG_HEIGHT = 224
IMG_WIDTH = 224

BATCH_SIZE = 32

NUM_CLASSES = 4

RANDOM_STATE = 42


CLASS_NAMES = [
    "Overripe",
    "Rippen",
    "Rotten",
    "Unripe"
]


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BANANA RIPENESS - RESNET50")
print("WITH DATA AUGMENTATION")
print("POST-TRAINING EVALUATION")
print("NO TRAINING WILL BE PERFORMED")
print("=" * 70)

print(
    "\nTensorFlow:",
    tf.__version__
)

print(
    "Dataset:",
    DATASET_PATH
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Results:",
    RESULTS_DIR
)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


print("\nDataset and model found successfully.")


# ============================================================
# COLLECT DATASET IMAGE PATHS
# ============================================================

print("\n" + "=" * 70)
print("COUNTING DATASET IMAGES")
print("=" * 70)


image_paths = []

labels = []


for class_index, class_name in enumerate(CLASS_NAMES):

    class_folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_folder):

        raise FileNotFoundError(
            f"\nClass folder not found:\n{class_folder}"
        )


    class_images = []


    for root, dirs, files in os.walk(
        class_folder
    ):

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
        f"{class_name:<12}: "
        f"{len(class_images):>6} images"
    )


    image_paths.extend(
        class_images
    )


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


# ============================================================
# RECREATE ORIGINAL DATASET SPLIT
# ============================================================

print("\n" + "=" * 70)
print("RECREATING ORIGINAL 70 / 15 / 15 SPLIT")
print("=" * 70)


# ------------------------------------------------------------
# 70% TRAINING
# 30% TEMPORARY
# ------------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(

    image_paths,

    labels,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=labels
)


# ------------------------------------------------------------
# 15% VALIDATION
# 15% TESTING
# ------------------------------------------------------------

X_val, X_test, y_val, y_test = train_test_split(

    X_temp,

    y_temp,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=y_temp
)


print(
    "\nTraining   :",
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


# ============================================================
# TEST CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TEST CLASS DISTRIBUTION")
print("=" * 70)


for class_index, class_name in enumerate(
    CLASS_NAMES
):

    count = np.sum(
        y_test == class_index
    )

    print(
        f"{class_name:<12}: "
        f"{count:>6}"
    )


# ============================================================
# IMAGE LOADING FUNCTION
# ============================================================

def load_test_image(
    image_path,
    label
):

    image = tf.io.read_file(
        image_path
    )


    image = tf.image.decode_image(

        image,

        channels=3,

        expand_animations=False
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


    # --------------------------------------------------------
    # RESNET50 IMAGENET PREPROCESSING
    # --------------------------------------------------------

    image = preprocess_input(
        image
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # MODEL USES CATEGORICAL CROSSENTROPY
    # THEREFORE LABEL MUST BE ONE-HOT
    # --------------------------------------------------------

    label = tf.one_hot(

        label,

        depth=NUM_CLASSES
    )


    return image, label


# ============================================================
# CREATE TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("CREATING TEST DATASET")
print("=" * 70)


test_dataset = tf.data.Dataset.from_tensor_slices(

    (
        X_test,
        y_test
    )
)


test_dataset = test_dataset.map(

    load_test_image,

    num_parallel_calls=tf.data.AUTOTUNE
)


test_dataset = test_dataset.batch(
    BATCH_SIZE
)


test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


print(
    "Test dataset ready."
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED RESNET50 MODEL")
print("=" * 70)


print(
    "\nLoading:"
)

print(
    MODEL_PATH
)


model = load_model(
    MODEL_PATH
)


print(
    "\nModel loaded successfully."
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)


test_loss, test_accuracy = model.evaluate(

    test_dataset,

    verbose=1
)


# ============================================================
# GENERATE PREDICTIONS
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


    predicted_classes = np.argmax(

        predictions,

        axis=1
    )


    true_classes = np.argmax(

        labels_batch.numpy(),

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


print(
    "\nPredictions generated:"
)

print(
    "Test samples:",
    len(y_true)
)


# ============================================================
# CALCULATE METRICS
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


# ============================================================
# FINAL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("RESNET50 WITH DATA AUGMENTATION")
print("FINAL TEST PERFORMANCE")
print("=" * 70)


print(
    f"\nTest Loss     : "
    f"{test_loss:.6f}"
)


print(
    f"Test Accuracy : "
    f"{accuracy * 100:.2f}%"
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


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    digits=4
)


print(
    report
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_Classification_Report.txt"
)


with open(

    report_path,

    "w"

) as f:


    f.write(
        "BANANA RIPENESS - RESNET50\n"
    )


    f.write(
        "WITH DATA AUGMENTATION\n"
    )


    f.write(
        "POST-TRAINING EVALUATION\n\n"
    )


    f.write(
        f"Test Loss     : "
        f"{test_loss:.6f}\n"
    )


    f.write(
        f"Test Accuracy : "
        f"{accuracy * 100:.2f}%\n"
    )


    f.write(
        f"Precision     : "
        f"{precision * 100:.2f}%\n"
    )


    f.write(
        f"Recall        : "
        f"{recall * 100:.2f}%\n"
    )


    f.write(
        f"F1 Score      : "
        f"{f1 * 100:.2f}%\n\n"
    )


    f.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(

    y_true,

    y_pred
)


print(
    cm
)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

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

    "Banana Ripeness - ResNet50 "
    "with Data Augmentation",

    fontsize=15
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
        "WITH DATA AUGMENTATION\n"
    )


    f.write(
        "POST-TRAINING EVALUATION\n\n"
    )


    f.write(
        f"Test Loss     : "
        f"{test_loss:.6f}\n"
    )


    f.write(
        f"Test Accuracy : "
        f"{accuracy * 100:.2f}%\n"
    )


    f.write(
        f"Precision     : "
        f"{precision * 100:.2f}%\n"
    )


    f.write(
        f"Recall        : "
        f"{recall * 100:.2f}%\n"
    )


    f.write(
        f"F1 Score      : "
        f"{f1 * 100:.2f}%\n"
    )


# ============================================================
# SAVE CONFUSION MATRIX VALUES
# ============================================================

cm_txt_path = os.path.join(

    RESULTS_DIR,

    "ResNet50_DA_Confusion_Matrix.txt"
)


with open(

    cm_txt_path,

    "w"

) as f:


    f.write(
        "BANANA RIPENESS - RESNET50\n"
    )

    f.write(
        "WITH DATA AUGMENTATION\n\n"
    )

    f.write(
        "Class order:\n"
    )

    f.write(
        "0 = Overripe\n"
    )

    f.write(
        "1 = Rippen\n"
    )

    f.write(
        "2 = Rotten\n"
    )

    f.write(
        "3 = Unripe\n\n"
    )

    f.write(
        "Confusion Matrix:\n\n"
    )

    f.write(
        str(cm)
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)


print(
    "\nNO TRAINING WAS PERFORMED."
)


print(
    "\nExisting model used:"
)


print(
    MODEL_PATH
)


print(
    "\nResults saved in:"
)


print(
    os.path.abspath(
        RESULTS_DIR
    )
)


print("\nSaved files:")


print(
    "1. ResNet50_DA_Confusion_Matrix.png"
)


print(
    "2. ResNet50_DA_Confusion_Matrix.txt"
)


print(
    "3. ResNet50_DA_Classification_Report.txt"
)


print(
    "4. ResNet50_DA_Final_Metrics.txt"
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)