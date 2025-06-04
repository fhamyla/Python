# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Step 2: Load and inspect the dataset
data = pd.read_csv('autoscout24-germany-dataset.csv')

print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe())
print("\nMissing values in each column:")
print(data.isnull().sum())

# Step 3: Data Preprocessing
data = data.dropna()  # Drop rows with missing values
encoder = LabelEncoder()
for column in data.select_dtypes(include='object').columns:
    data[column] = encoder.fit_transform(data[column])

X = data.drop('fuel', axis=1)  # Features
y = data['fuel']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (for models like Logistic Regression and SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Experiment 1: Baseline Model Performance (All Features) ---
print("\nExperiment 1: Baseline Model Performance (All Features)\n")

# Train and evaluate models on all features
models = [
    ("Logistic Regression", LogisticRegression()),
    ("KNN", KNeighborsClassifier(n_neighbors=10)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=10)),
    ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=10)),
    ("SVM", SVC(C=1.0, kernel='rbf', gamma='scale'))
]

results_all_features = []
for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results_all_features.append((model_name, accuracy))

# Display results for all models
df_results_all_features = pd.DataFrame(results_all_features, columns=['Model', 'Accuracy'])
df_results_all_features.set_index('Model', inplace=True)
df_results_all_features.plot(kind='bar', figsize=(10, 6), color=['lightgreen'])
plt.title("Model Accuracy Comparison (All Features)")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()

# --- Experiment 2: Feature Selection (Reduced Features using SelectKBest) ---
print("\nExperiment 2: Feature Selection (Reduced Features using SelectKBest)\n")

# Feature selection using SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_reduced = selector.fit_transform(X, y)

# Train-test split with reduced features
X_train_reduced, X_test_reduced = train_test_split(X_reduced, test_size=0.2, random_state=42)

# Train and evaluate models on reduced features
results_reduced_features = []
for model_name, model in models:
    model.fit(X_train_reduced, y_train)
    y_pred = model.predict(X_test_reduced)
    accuracy = accuracy_score(y_test, y_pred)
    results_reduced_features.append((model_name, accuracy))

# Display results for reduced features
df_results_reduced_features = pd.DataFrame(results_reduced_features, columns=['Model', 'Accuracy'])
df_results_reduced_features.set_index('Model', inplace=True)
df_results_reduced_features.plot(kind='bar', figsize=(10, 6), color=['lightcoral'])
plt.title("Model Accuracy Comparison (Reduced Features)")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()

# --- Experiment 3: Hyperparameter Tuning (KNN and Random Forest) ---
print("\nExperiment 3: Hyperparameter Tuning (KNN and Random Forest)\n")

# Hyperparameter tuning for KNN and Random Forest
# KNN: Testing different values for n_neighbors
knn_tune = [3, 5, 7, 10]
knn_results = []
for k in knn_tune:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_reduced, y_train)
    y_pred_knn = knn.predict(X_test_reduced)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    knn_results.append((k, accuracy_knn))

# Random Forest: Testing different values for n_estimators and max_depth
rf_tune = [(100, 5), (100, 10), (200, 5), (200, 10)]
rf_results = []
for n_estimators, max_depth in rf_tune:
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(X_train_reduced, y_train)
    y_pred_rf = rf.predict(X_test_reduced)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    rf_results.append((f"{n_estimators}-{max_depth}", accuracy_rf))

# Display results
df_knn_results = pd.DataFrame(knn_results, columns=['K', 'Accuracy'])
df_rf_results = pd.DataFrame(rf_results, columns=['Params', 'Accuracy'])

# Plot results for KNN
df_knn_results.plot(kind='bar', x='K', y='Accuracy', figsize=(10, 6), color='skyblue', title="KNN Hyperparameter Tuning")
plt.ylabel("Accuracy")
plt.xticks(rotation=0)
plt.show()

# Plot results for Random Forest
df_rf_results.plot(kind='bar', x='Params', y='Accuracy', figsize=(10, 6), color='lightgreen', title="Random Forest Hyperparameter Tuning")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()