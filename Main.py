from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import load_iris
import numpy as np

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into three partitions A, B, and C
X_A, X_temp, y_A, y_temp = train_test_split(X, y, test_size=0.66, random_state=42, stratify=y)
X_B, X_C, y_B, y_C = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Function to evaluate a classifier and print metrics
def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred, average='weighted')
    precision = metrics.precision_score(y_test, y_pred, average='weighted')

    # Specificity is not a direct metric in scikit-learn; you may need to calculate it manually
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    specificity = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print()

# First experiment: Training (A+B) and Testing (C)
X_train_1 = np.concatenate((X_A, X_B))
y_train_1 = np.concatenate((y_A, y_B))
X_test_1, y_test_1 = X_C, y_C

# Second experiment: Training (A+C) and Testing (B)
X_train_2 = np.concatenate((X_A, X_C))
y_train_2 = np.concatenate((y_A, y_C))
X_test_2, y_test_2 = X_B, y_B

# Third experiment: Training (C+B) and Testing (A)
X_train_3 = np.concatenate((X_C, X_B))
y_train_3 = np.concatenate((y_C, y_B))
X_test_3, y_test_3 = X_A, y_A

# Decision Trees
print("Decision Trees:")
print("Experiment 1:")
evaluate_classifier(DecisionTreeClassifier(criterion='entropy'), X_train_1, y_train_1, X_test_1, y_test_1)
print("Experiment 2:")
evaluate_classifier(DecisionTreeClassifier(criterion='entropy'), X_train_2, y_train_2, X_test_2, y_test_2)
print("Experiment 3:")
evaluate_classifier(DecisionTreeClassifier(criterion='entropy'), X_train_3, y_train_3, X_test_3, y_test_3)

# KNN
print("KNN:")
print("Experiment 1:")
evaluate_classifier(KNeighborsClassifier(), X_train_1, y_train_1, X_test_1, y_test_1)
print("Experiment 2:")
evaluate_classifier(KNeighborsClassifier(), X_train_2, y_train_2, X_test_2, y_test_2)
print("Experiment 3:")
evaluate_classifier(KNeighborsClassifier(), X_train_3, y_train_3, X_test_3, y_test_3)
