# -Alfido-tech-TASK2-
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore") # To ignore convergence warnings
# Step 1: Load dataset
data = load_iris()
X = data.data
y = data.target
# Step 2: Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 3: Train a Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
# Step 4: Train a Logistic Regression Classifier
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
# Step 5: Evaluate both modelsX
print("=== Decision Tree Evaluation ===")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("Classification Report:\n", classification_report(y_test, dt_predictions))
print("\n=== Logistic Regression Evaluation ===")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print("Classification Report:\n", classification_report(y_test, lr_predictions))
 
 OUTPUT
=== Decision Tree Evaluation ===
Accuracy: 1.0
Classification Report:
 precision recall f1-score support
...
=== Logistic Regression Evaluation ===
Accuracy: 0.9667
Classification Report:
 precision recall f1-score support
