# logistic-regression-classifier-built-from-scratch-with-NumPy-
A logistic regression model built from scratch with NumPy to predict Netflix customer churn. It uses gradient descent, sigmoid activation, and cross-entropy loss with early stopping. Data is preprocessed via one-hot encoding and standardization, and performance is evaluated on train/test splits.
The model uses:

Gradient Descent for optimization

Sigmoid Activation Function to map linear outputs into probabilities

Cross-Entropy Loss to measure prediction error

Early Stopping to prevent overfitting by monitoring validation loss and restoring the best weights

 Data Preprocessing

Dropped non-informative columns like customer_id

Applied One-Hot Encoding to convert categorical features into numeric form

Scaled numerical features using StandardScaler to improve gradient descent convergence

Split dataset into training and testing sets with stratification for balanced churn classes

 Model Evaluation

The model outputs probabilities of churn for each customer and converts them into binary predictions:

0 → Customer stays

1 → Customer churns

Performance is evaluated on both the training and testing sets. While accuracy is reported, the project can be extended with additional metrics like precision, recall, F1-score, and ROC-AUC for deeper insight into churn prediction.
