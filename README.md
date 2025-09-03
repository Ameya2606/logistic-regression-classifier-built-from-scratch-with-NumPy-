###README.md

````markdown
# 📉 Customer Churn Prediction using From-Scratch Logistic Regression

## 📝 Project Overview

This repository contains a **from-scratch implementation of a Logistic Regression model** built in Python. The model is designed to solve a binary classification problem: predicting whether a Netflix customer will churn (cancel their subscription).

The primary goal is to demonstrate a deep understanding of the logistic regression algorithm by building it from the ground up, rather than relying on high-level libraries like Scikit-learn for the core model logic. The project also showcases best practices in machine learning, including data preprocessing, feature scaling, and model evaluation with multiple metrics.

## 🚀 Features

* **From-Scratch Implementation:** The `LogisticRegression` class is implemented using NumPy, including the sigmoid function, cost calculation (cross-entropy), and gradient descent for weight optimization.
* **Early Stopping:** The model includes an early stopping mechanism to prevent overfitting and ensure it finds the optimal set of weights during training.
* **Comprehensive Evaluation:** The `main_model.py` script not only trains the model but also evaluates its performance using key metrics like **accuracy**, **precision**, **recall**, and the **F1-score**.
* **Data Visualization:** A plot of the training and validation costs is generated to visually demonstrate the learning process and the point at which early stopping is triggered.
* **Hyperparameter Demonstration:** The code includes a section that shows how different hyperparameters (e.g., learning rate) can impact the model's final performance.
* **Standardized Workflow:** The project follows a typical machine learning workflow, including data loading, preprocessing (one-hot encoding), data splitting, and feature scaling.

## ⚙️ How to Run the Code

To run this project, follow these simple steps.

### Prerequisites

You'll need Python installed on your system. The required libraries can be installed using `pip`.

```bash
pip install -r requirements.txt
````

### Execution

1.  Clone this repository to your local machine.
2.  Ensure you have the `netflix_customer_churn.csv` dataset in the same directory as the `main_model.py` script.
3.  Run the main script from your terminal:

<!-- end list -->

```bash
python main_model.py
```

The script will handle all data loading, preprocessing, model training, and evaluation, displaying the results in the console and generating a plot of the cost function.

## 📂 File Structure

  * `main_model.py`: The main Python script containing the `LogisticRegression` class and the full machine learning pipeline.
  * `netflix_customer_churn.csv`: The dataset used for training and testing the model.
  * `README.md`: This file, providing an overview of the project.
  * `requirements.txt`: A list of all necessary Python libraries.

## 📊 Expected Output

When you run the script, you should see output similar to the following, showcasing the training progress and final evaluation metrics:

```
Training the Logistic Regression model...
Epoch 100/5000 | Train Cost: 0.6931 | Val Cost: 0.6931
...
--- Early stopping triggered at epoch 1200 ---
Model restored to best state based on validation cost.

--- Final Model Evaluation ---
Training Set Accuracy: 81.25%
Test Set Accuracy:     80.50%

--- Additional Metrics ---
Training Set Precision: 0.81
Test Set Precision:     0.80
...
```

A plot will also be generated, showing the training and validation costs over time.

```
```
