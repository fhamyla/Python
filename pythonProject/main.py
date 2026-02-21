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
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('AirQualityUCI.csv')

print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe())
print("\nMissing values in each column:")
print(data.isnull().sum())

data = data.dropna()

encoder = LabelEncoder()
for column in data.select_dtypes(include='object').columns:
    data[column] = encoder.fit_transform(data[column])
    
X = data.drop('fuel', axis=1)
y = data['fuel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=10)
X_reduced = selector.fit_transform(X, y)
print("Selected Features Shape:", X_reduced.shape)

X_train_reduced, X_test_reduced = train_test_split(
    X_reduced, test_size=0.2, random_state=42
)

def plot_confusion_matrix(y_actual, y_predicted, title):
    cm = confusion_matrix(y_actual, y_predicted)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

lr_reduced = LogisticRegression()
lr_reduced.fit(X_train_reduced, y_train)
y_pred_lr_reduced = lr_reduced.predict(X_test_reduced)
print("Logistic Regression (Reduced Features):")
print(classification_report(y_test, y_pred_lr_reduced))
plot_confusion_matrix(y_test, y_pred_lr_reduced, "Logistic Regression (Reduced Features)")

knn_reduced = KNeighborsClassifier(n_neighbors=10, weights='distance', metric='manhattan')
knn_reduced.fit(X_train_reduced, y_train)
y_pred_knn_reduced = knn_reduced.predict(X_test_reduced)
print("\nKNN (Reduced Features):")
print(classification_report(y_test, y_pred_knn_reduced))
plot_confusion_matrix(y_test, y_pred_knn_reduced, "KNN (Reduced Features)")

dt_reduced = DecisionTreeClassifier(max_depth=10, min_samples_split=5, criterion='entropy')
dt_reduced.fit(X_train_reduced, y_train)
y_pred_dt_reduced = dt_reduced.predict(X_test_reduced)
print("\nDecision Tree (Reduced Features):")
print(classification_report(y_test, y_pred_dt_reduced))
plot_confusion_matrix(y_test, y_pred_dt_reduced, "Decision Tree (Reduced Features)")

rf_reduced = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced, y_train)
y_pred_rf_reduced = rf_reduced.predict(X_test_reduced)
print("\nRandom Forest (Reduced Features):")
print(classification_report(y_test, y_pred_rf_reduced))
plot_confusion_matrix(y_test, y_pred_rf_reduced, "Random Forest (Reduced Features)")

svm_reduced = SVC(C=1.0, kernel='rbf', gamma='scale')
svm_reduced.fit(X_train_reduced, y_train)
y_pred_svm_reduced = svm_reduced.predict(X_test_reduced)
print("\nSVM (Reduced Features):")
print(classification_report(y_test, y_pred_svm_reduced))
plot_confusion_matrix(y_test, y_pred_svm_reduced, "SVM (Reduced Features)")

from sklearn.metrics import accuracy_score

accuracy_lr_reduced = accuracy_score(y_test, y_pred_lr_reduced)
accuracy_knn_reduced = accuracy_score(y_test, y_pred_knn_reduced)
accuracy_dt_reduced = accuracy_score(y_test, y_pred_dt_reduced)
accuracy_rf_reduced = accuracy_score(y_test, y_pred_rf_reduced)
accuracy_svm_reduced = accuracy_score(y_test, y_pred_svm_reduced)

results = {
    'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'SVM'],
    'Accuracy (Reduced Features)': [
        accuracy_lr_reduced, accuracy_knn_reduced, accuracy_dt_reduced,
        accuracy_rf_reduced, accuracy_svm_reduced
    ]
}

df_results = pd.DataFrame(results)
df_results.set_index('Model', inplace=True)
df_results.plot(kind='bar', figsize=(10, 6), color=['skyblue'])
plt.title("Model Accuracy Comparison (Reduced Features)")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()