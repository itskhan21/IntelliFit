import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'C:\\Users\\abdul\\OneDrive\\Desktop\\HeHe!\\7th Semester\\AI\\Project\\gym_members_exercise_tracking.csv'
df = pd.read_csv(file_path)

# Encode categorical features
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df = pd.get_dummies(df, columns=['Workout_Type'], prefix='Workout')
df = df.astype({col: 'int' for col in df.select_dtypes(include='bool').columns})

# Normalize/Standardize numerical features
numerical_features = [
    'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
    'Session_Duration (hours)', 'Fat_Percentage',
    'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'BMI'
]
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split the dataset into features and target variables
X = df.drop(columns=['Calories_Burned'])  # Features
y = df['Calories_Burned']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert target variable to categorical for classification
y_class = pd.cut(y, bins=4, labels=[0, 1, 2, 3])  # Creating bins for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# 1. Linear Regression Evaluation
print("\nEvaluating Linear Regression model...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
cross_val_scores_lr = cross_val_score(linear_model, X, y, cv=5, scoring='neg_mean_squared_error')
std_dev_lr = np.std(cross_val_scores_lr)
print(f"Linear Regression Mean Squared Error: {mse_lr}")
print(f"Linear Regression Cross-Validation MSE Scores: {-cross_val_scores_lr}")
print(f"Linear Regression Standard Deviation of MSE: {std_dev_lr}")

# 2. Naive Bayes Classifier Evaluation
print("\nEvaluating Naive Bayes Classifier...")
nb_model = GaussianNB()
nb_model.fit(X_train_class, y_train_class)
y_pred_nb = nb_model.predict(X_test_class)
accuracy_nb = accuracy_score(y_test_class, y_pred_nb)
precision_nb = precision_score(y_test_class, y_pred_nb, average='weighted')
recall_nb = recall_score(y_test_class, y_pred_nb, average='weighted')
f1_nb = f1_score(y_test_class, y_pred_nb, average='weighted')
cross_val_scores_nb = cross_val_score(nb_model, X_train_class, y_train_class, cv=5, scoring='accuracy')
std_dev_nb = np.std(cross_val_scores_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")
print(f"Naive Bayes Precision: {precision_nb}")
print(f"Naive Bayes Recall: {recall_nb}")
print(f"Naive Bayes F1-Score: {f1_nb}")
print(f"Naive Bayes Cross-Validation Accuracy Scores: {cross_val_scores_nb}")
print(f"Naive Bayes Standard Deviation of Accuracy: {std_dev_nb}")

# 3. Neural Network Evaluation
print("\nEvaluating Neural Network model...")
# Increase max_iter to 1000 and use solver 'lbfgs' for better convergence on smaller datasets
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, solver='lbfgs', random_state=42)
nn_model.fit(X_train_class, y_train_class)
y_pred_nn = nn_model.predict(X_test_class)
accuracy_nn = accuracy_score(y_test_class, y_pred_nn)
precision_nn = precision_score(y_test_class, y_pred_nn, average='weighted')
recall_nn = recall_score(y_test_class, y_pred_nn, average='weighted')
f1_nn = f1_score(y_test_class, y_pred_nn, average='weighted')
cross_val_scores_nn = cross_val_score(nn_model, X_train_class, y_train_class, cv=5, scoring='accuracy')
std_dev_nn = np.std(cross_val_scores_nn)

# Output evaluation metrics
print(f"Neural Network Accuracy: {accuracy_nn}")
print(f"Neural Network Precision: {precision_nn}")
print(f"Neural Network Recall: {recall_nn}")
print(f"Neural Network F1-Score: {f1_nn}")
print(f"Neural Network Cross-Validation Accuracy Scores: {cross_val_scores_nn}")
print(f"Neural Network Standard Deviation of Accuracy: {std_dev_nn}")

# Plotting ROC Curves for Naive Bayes and Neural Network
y_test_class_binary = pd.get_dummies(y_test_class, drop_first=False)  # Convert to binary format for ROC Curve
plt.figure(figsize=(12, 6))

for (i, col) in enumerate(y_test_class_binary.columns):
    y_score_nb = nb_model.predict_proba(X_test_class)[:, i]
    fpr_nb, tpr_nb, _ = roc_curve(y_test_class_binary.iloc[:, i], y_score_nb)
    plt.plot(fpr_nb, tpr_nb, linestyle='--', label=f'Naive Bayes (Class {col})')

    y_score_nn = nn_model.predict_proba(X_test_class)[:, i]  # Corrected 'mlp_model' to 'nn_model'
    fpr_nn, tpr_nn, _ = roc_curve(y_test_class_binary.iloc[:, i], y_score_nn)
    plt.plot(fpr_nn, tpr_nn, linestyle='-', label=f'Neural Network (Class {col})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Naive Bayes and Neural Network')
plt.legend()
plt.show()