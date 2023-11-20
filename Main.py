from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import numpy as np

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into three partitions A, B, and C
X_A, X_temp, y_A, y_temp = train_test_split(X, y, test_size=0.66, stratify=y)
X_B, X_C, y_B, y_C = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

# Function to evaluate a classifier and print metrics
def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred, average='weighted')
    precision = metrics.precision_score(y_test, y_pred, average='weighted')

    # Specificity is not a direct metric in scikit-learn ,need to calculate it manually
    tn= metrics.confusion_matrix(y_test, y_pred).ravel()
    specificity = tn[0]/(tn[0]+tn[4])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print()
    return (classifier,accuracy, sensitivity, specificity, precision)



# Function to calculete and print the avarage metrics
def calculate_avarege(results):
    total_accuracy = 0
    total_sensitivity = 0
    total_specificity = 0
    total_precision = 0

    print("Average values")

    for i in results:
        total_accuracy += i[1]
        total_sensitivity += i[2]
        total_specificity += i[3]
        total_precision += i[4]

    print(f"Accuracy: {total_accuracy/3:.4f}")
    print(f"Sensitivity: {total_sensitivity/3:.4f}")
    print(f"Specificity: {total_specificity/3:.4f}")
    print(f"Precision: {total_precision/3:.4f}")
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

# Initialize variables to store models for later tree visualization
tree_models = []

# Decision Trees
print("Decision Trees:")
print("Experiment 1:")
tree_models.append(evaluate_classifier(DecisionTreeClassifier(criterion='entropy'), X_train_1, y_train_1, X_test_1, y_test_1))
#Plot the structure of the decision tree
fig = plt.figure(figsize=(25,20))
tree.plot_tree(tree_models[0][0],feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
#Save the structure of the decision tree as a image
fig.savefig("decistion_tree.png")

print("Experiment 2:")
tree_models.append(evaluate_classifier(DecisionTreeClassifier(criterion='entropy'), X_train_2, y_train_2, X_test_2, y_test_2))
tree.plot_tree(tree_models[1][0],feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
fig.savefig("decistion_tree2.png")

print("Experiment 3:")
tree_models.append(evaluate_classifier(DecisionTreeClassifier(criterion='entropy'), X_train_3, y_train_3, X_test_3, y_test_3))
tree.plot_tree(tree_models[2][0],feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
fig.savefig("decistion_tree3.png")
print()

calculate_avarege(tree_models)


# Initialize variables to store models for later KNN visualization
knn_models = []

# KNN
print("KNN:")
print("Experiment 1:")
knn_models.append(evaluate_classifier(KNeighborsClassifier(), X_train_1, y_train_1, X_test_1, y_test_1))
print("Experiment 2:")
knn_models.append(evaluate_classifier(KNeighborsClassifier(), X_train_2, y_train_2, X_test_2, y_test_2))
print("Experiment 3:")
knn_models.append(evaluate_classifier(KNeighborsClassifier(), X_train_3, y_train_3, X_test_3, y_test_3))

calculate_avarege(knn_models)