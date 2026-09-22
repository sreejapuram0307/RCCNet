import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# BANANA RIPENESS - RESULTS FROM CONFUSION MATRIX
# ============================================================

# Confusion matrix from your actual result
cm = np.array([
    [233, 1,   1,   0],
    [14,  672, 18,  5],
    [10,  7, 753,  2],
    [0,   0,   4, 196]
])

class_names = [
    "Overripe",
    "Rippen",
    "Rotten",
    "Unripe"
]

# ============================================================
# CALCULATE ACCURACY
# ============================================================

total_correct = np.trace(cm)

total_samples = np.sum(cm)

test_accuracy = total_correct / total_samples

print("==============================================")
print("BANANA RIPENESS RESULTS")
print("==============================================")

print("Correct predictions:", total_correct)
print("Total test images:", total_samples)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================================
# CLASS-WISE ACCURACY
# ============================================================

print("\nClass-wise Accuracy:")

for i, class_name in enumerate(class_names):

    class_accuracy = cm[i, i] / np.sum(cm[i])

    print(
        f"{class_name}: "
        f"{class_accuracy * 100:.2f}%"
    )


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    ["Test Accuracy"],
    [test_accuracy * 100]
)

plt.ylim(0, 100)

plt.ylabel("Accuracy (%)")

plt.title(
    "Banana Ripeness - GoogLeNet Test Accuracy"
)

plt.text(
    0,
    test_accuracy * 100 + 2,
    f"{test_accuracy * 100:.2f}%",
    ha="center",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "Banana_GoogLeNet_test_accuracy.png",
    dpi=300
)

plt.show()


# ============================================================
# CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(cm)

plt.title(
    "Banana Ripeness - GoogLeNet Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)

for i in range(len(class_names)):

    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=12
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "Banana_GoogLeNet_confusion_matrix.png",
    dpi=300
)

plt.show()


print("\n==============================================")
print("RESULTS GENERATED")
print("==============================================")