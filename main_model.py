import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    """
    A simple Logistic Regression model implemented from scratch for binary classification.
    """
    def __init__(self, learning_rate=0.01, epochs=1000, patience=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.patience = patience # For early stopping

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, X_val, y_val):
        """
        Train the logistic regression model using gradient descent with early stopping.

        Args:
            X_val: Validation features for early stopping.
            y_val: Validation target for early stopping.
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # --- Early Stopping Initialization ---
        best_val_cost = float('inf')
        patience_counter = 0
        best_weights = None
        best_bias = None

        # Gradient Descent
        for epoch in range(self.epochs):
            # Linear model
            z = np.dot(X, self.weights) + self.bias
            # Predicted probabilities
            y_hat = self._sigmoid(z)

            # Calculate gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / num_samples) * np.sum(y_hat - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # --- Monitoring and Early Stopping (checked every 100 epochs) ---
            if (epoch + 1) % 100 == 0:
                # Add a small epsilon to prevent log(0) which results in NaN
                epsilon = 1e-9
                train_cost = -np.mean(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))

                # Calculate validation cost
                val_z = np.dot(X_val, self.weights) + self.bias
                val_y_hat = self._sigmoid(val_z)
                val_cost = -np.mean(y_val * np.log(val_y_hat + epsilon) + (1 - y_val) * np.log(1 - val_y_hat + epsilon))

                print(f"Epoch {epoch + 1}/{self.epochs} | Train Cost: {train_cost:.4f} | Val Cost: {val_cost:.4f}")

                # Check for improvement
                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    patience_counter = 0
                    # Save the best model weights
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"\n--- Early stopping triggered at epoch {epoch + 1} ---")
                    break
        
        # Restore the best model found
        if best_weights is not None:
            self.weights = best_weights
            self.bias = best_bias
            print("Model restored to best state based on validation cost.")

    def predict(self, X):
        """Predict binary labels for a given dataset."""
        z = np.dot(X, self.weights) + self.bias
        y_pred_proba = self._sigmoid(z)
        return (y_pred_proba >= 0.5).astype(int)

def main():
    # 1) Build the path to the CSV file next to this script
    csv_path = Path(__file__).parent / "netflix_customer_churn.csv"

    # 2) Read the CSV into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # 3) Drop the 'customer_id' column from the DataFrame (reassign instead of inplace)
    df = df.drop('customer_id', axis=1)
    
    # 4) Identify all non-numeric columns in the DataFrame.
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # 5) Perform one-hot encoding on the identified categorical columns.
    encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 6) Separate the features (X) and the target variable (y).
    X = encoded_df.drop('churned', axis=1)
    y = encoded_df['churned']
    
    # 7) Split data into training and testing sets, ensuring class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8) Scale numerical features
    # Gradient-based algorithms perform better with scaled features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 9) Convert pandas Series to NumPy arrays for the algorithm
    # This avoids potential index alignment issues and is more efficient.
    y_train_np = y_train.values
    y_test_np = y_test.values

    # --- Step 4: Train the model ---
    print("Training the Logistic Regression model...")
    # We can still set a high epoch count, but early stopping will likely halt it much sooner.
    model = LogisticRegression(learning_rate=0.01, epochs=5000, patience=20)
    
    # For simplicity, we use the test set as the validation set for early stopping.
    # In a more formal setup, you might create a separate validation split from the training data.
    model.fit(X_train_scaled, y_train_np, X_test_scaled, y_test_np)

    # --- Step 5: Evaluate the model ---
    print("\n--- Final Model Evaluation ---")
    # Predict on both training and test sets to check for overfitting
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Calculate and print accuracy for both sets
    train_accuracy = np.mean(y_pred_train == y_train_np)
    test_accuracy = np.mean(y_pred_test == y_test_np)

    print(f"Training Set Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Set Accuracy:     {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
