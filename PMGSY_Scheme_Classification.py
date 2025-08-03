# PMGSY Scheme Classification using Random Forest + SMOTE + Scaling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib


df = pd.read_csv("/content/PMGSY_DATASET.csv")  # Update path as needed

if 'PMGSY_SCHEME' in df.columns:
    df.rename(columns={'PMGSY_SCHEME': 'Target'}, inplace=True)
elif 'Label' in df.columns:
    df.rename(columns={'Label': 'Target'}, inplace=True)
elif 'Target' not in df.columns:
    raise ValueError("Could not find a target column.")

df.drop(columns=['STATE_NAME', 'DISTRICT_NAME', 'Unnamed: 14'], inplace=True, errors='ignore')

df.dropna(inplace=True)

X = df.drop(columns=['Target'])
y = df['Target']

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

print(f"Dataset size after SMOTE: {len(X_resampled)} samples")


X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.4, random_state=42
)
print(f"Training Set: {len(X_train)} samples")
print(f"Testing Set : {len(X_test)} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

train_acc = model.score(X_train_scaled, y_train)
test_acc = accuracy_score(y_test, y_pred)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy : {test_acc * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

actual_counts = pd.Series(y_test).value_counts().sort_index()
predicted_counts = pd.Series(y_pred).value_counts().sort_index()

actual_class_names = target_encoder.inverse_transform(actual_counts.index)
predicted_class_names = target_encoder.inverse_transform(predicted_counts.index)

actual_counts.index = actual_class_names
predicted_counts.index = predicted_class_names

print("\nActual Class Counts:")
print(actual_counts.sort_index())

print("\nPredicted Class Counts:")
print(predicted_counts.sort_index())

# Bar Plot
df_compare = pd.DataFrame({
    'Actual': actual_counts,
    'Predicted': predicted_counts
})
df_compare.plot(kind='bar', figsize=(10, 6))
plt.title("Actual vs Predicted Class Counts")
plt.xlabel("PMGSY Scheme Class")
plt.ylabel("Number of Projects")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1][:10]  # Top 10

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Save Model Artifacts
joblib.dump(model, "pmgsy_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(target_encoder, "label_encoder.pkl")
print("Model, Scaler, and LabelEncoder saved successfully.")

# Predict a Single Sample
i = 11
sample = X_test.iloc[i].values.reshape(1, -1)
sample_scaled = scaler.transform(sample)

pred_encoded = model.predict(sample_scaled)[0]
true_encoded = y_test[i]

pred_label = target_encoder.inverse_transform([pred_encoded])[0]
true_label = target_encoder.inverse_transform([true_encoded])[0]

print("\n Demo Sample Index:", i)
print(" True Class      :", true_label)
print(" Predicted Class :", pred_label)
print(" Result          :", " Correct" if pred_label == true_label else " Incorrect")

# Cross-Validation
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=7)
print(f"\n 7-Fold Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")
