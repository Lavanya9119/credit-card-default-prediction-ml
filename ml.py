import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
import seaborn as sns 
import matplotlib.pyplot as plt 
 
7 
 
 
# Load the dataset 
file_path = 'default of credit card clients.xls'  # Update with your file path 
try: 
    data = pd.read_excel(file_path, header=1) 
except ImportError: 
    print("Please install 'xlrd' for reading .xls files using 'pip install xlrd'.") 
    exit() 
except ValueError: 
    print("Unable to read .xls file. Convert it to .xlsx format and try again.") 
    exit() 
 
# Analyze dataset 
print("Dataset Columns:") 
print(data.columns) 
 
# Target variable 
target_column = 'default payment next month' 
if target_column not in data.columns: 
    print(f"Target column '{target_column}' not found.") 
    exit() 
 
# Check target distribution 
print("\nTarget Variable Distribution:") 
print(data[target_column].value_counts()) 
 
# Split features and target 
X = data.drop(columns=[target_column]) 
y = data[target_column] 
 
# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 
 
# Scale features 
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
 
# Models 
models = { 
    "Logistic Regression": LogisticRegression(), 
    "KNN": KNeighborsClassifier(n_neighbors=5), 
    "Decision Tree": DecisionTreeClassifier(), 
    "Random Forest": RandomForestClassifier(), 
    "SVM": SVC() 
} 
 
# Train and evaluate models 
results = [] 
plt.figure(figsize=(15, 5))  # For confusion matrix heatmaps 
 
8 
 
for i, (name, model) in enumerate(models.items()): 
    model.fit(X_train_scaled, y_train) 
    y_pred = model.predict(X_test_scaled) 
 
    # Calculate metrics 
    cm = confusion_matrix(y_test, y_pred) 
    accuracy = accuracy_score(y_test, y_pred) 
    report = classification_report(y_test, y_pred, output_dict=True) 
 
    # Store metrics 
    results.append({ 
        "Model": name, 
        "Accuracy": accuracy, 
        "Precision": report['weighted avg']['precision'], 
        "Recall": report['weighted avg']['recall'], 
        "F1 Score": report['weighted avg']['f1-score'] 
    }) 
 
    # Plot confusion matrix 
    plt.subplot(1, len(models), i + 1) 
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues") 
    plt.title(f"{name}") 
    plt.xlabel("Predicted") 
    plt.ylabel("Actual") 
 
plt.tight_layout() 
plt.show() 
 
# Create DataFrame for metrics 
metrics_df = pd.DataFrame(results) 
 
# Display metrics 
print("Performance Metrics:") 
print(metrics_df) 
 
# Plot metrics 
metrics_df.set_index("Model").plot(kind='bar', figsize=(10, 6)) 
plt.title("Model Performance Comparison") 
plt.ylabel("Score") 
plt.xticks(rotation=45) 
plt.legend(loc='lower right') 
plt.show() 