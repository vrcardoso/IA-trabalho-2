# AI Algorithms for evaluate the Decision Trees (tree generation with the entropy metric) and KNN methods to classify iris data.

This repository explores the application of Decision Trees (tree generation with the entropy metric) and KNN methods to classify the public database iris.
The division into training and testing is done as follows:the Iris database is divide into approximately three partitions with 1 third of the records: A, B and C.
Subsequently, the partitions is grouped into training and testing, resulting in 3 experiments:

     First: Training (A+B) and Testing (C)
     Second: Training (A+C) and Testing (B)
     Third: Training (C+B) and Testing (A)
     
     1- For each experiment, the metrics (Accuracy, Sensitivity, Specificity, Precision) must be calculated for the Test basis. This must be done for each classification method investigated (KNN and Tree).
     2- For each experiment, the structure of the decision tree obtained during training must be presented.
     3- At the end, the average value of the metrics (considering the 3 experiments) for each classification method must be presented.

For this implementation is used Scikit-learn: http://scikit-learn.org/, an python open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection, model evaluation, and many other utilities.

## Introduction

### Decision Trees (tree generation with the entropy metric)

Decision Trees are a popular and surprisingly effective technique, particularly for classification problems.The criterion for selecting variables and hierarchy can be tricky to get, not to mention Gini index, Entropyand information gain.

Decision Trees offer tremendous flexibility in that we can use both numeric and categorical variables for splitting the target data. Categoric data is split along the different classes in the variable.

### KNN - K-Nearest Neighbors

K-Nearest Neighbors (KNN) is a simple and widely used algorithm in machine learning for classification and regression tasks. It's a type of instance-based learning, where the algorithm makes predictions by finding the "k" training examples that are nearest to a given input data point and then combining the labels or values of those neighbors to make a prediction.



## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python (version 3.11 recommended)
- Dependencies: {List of dependencies or a reference to `requirements.txt`}

### Installation

1. Clone the repository to your local machine.
2. Install dependencies from `requirements.txt` with `pip install -r requirements.txt` 
