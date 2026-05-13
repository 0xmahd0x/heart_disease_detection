# =========================================
# Import Libraries
# =========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

import joblib
import os

# =========================================
# Create Folders
# =========================================

os.makedirs("plots", exist_ok=True)

# =========================================
# Load Dataset
# =========================================

df = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

# =========================================
# Take Sample
# =========================================

df = df.sample(n=20000, random_state=42)

# =========================================
# Display Basic Info
# =========================================

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

# =========================================
# Check Missing Values
# =========================================

print("\nMissing Values:")
print(df.isnull().sum())

# =========================================
# Check Duplicates
# =========================================

print("\nDuplicates:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates:")
print(df.shape)

# =========================================
# Target Distribution
# =========================================

print("\nTarget Distribution:")
print(df['HeartDiseaseorAttack'].value_counts())

# =========================================
# Target Distribution Plot
# =========================================

plt.figure(figsize=(6,4))

sns.countplot(x=df['HeartDiseaseorAttack'])

plt.title("Target Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Count")

plt.savefig("plots/target_distribution.png")

plt.show()

# =========================================
# Correlation Heatmap
# =========================================

plt.figure(figsize=(18,12))

sns.heatmap(
    df.corr(),
    cmap='coolwarm',
    annot=False
)

plt.title("Correlation Heatmap")

plt.savefig("plots/correlation_heatmap.png")

plt.show()

# =========================================
# BMI Distribution
# =========================================

plt.figure(figsize=(8,5))

sns.histplot(df['BMI'], bins=30, kde=True)

plt.title("BMI Distribution")

plt.savefig("plots/bmi_distribution.png")

plt.show()

# =========================================
# Age Distribution
# =========================================

plt.figure(figsize=(8,5))

sns.countplot(x=df['Age'])

plt.title("Age Distribution")

plt.savefig("plots/age_distribution.png")

plt.show()

# =========================================
# Outliers Detection
# =========================================

plt.figure(figsize=(12,6))

sns.boxplot(data=df[['BMI', 'MentHlth', 'PhysHlth']])

plt.title("Outliers Detection")

plt.savefig("plots/outliers_detection.png")

plt.show()

# =========================================
# Features & Target
# =========================================

X = df.drop("HeartDiseaseorAttack", axis=1)

y = df["HeartDiseaseorAttack"]

# =========================================
# Save Feature Names
# =========================================

feature_names = X.columns.tolist()

joblib.dump(feature_names, "features.pkl")

# =========================================
# Train Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# Before SMOTE
# =========================================

print("\nBefore SMOTE:")
print(y_train.value_counts())

# =========================================
# Apply SMOTE
# =========================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# =========================================
# After SMOTE
# =========================================

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# =========================================
# SMOTE Visualization
# =========================================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

sns.countplot(x=y_train)

plt.title("Before SMOTE")

plt.subplot(1,2,2)

sns.countplot(x=y_train_smote)

plt.title("After SMOTE")

plt.tight_layout()

plt.savefig("plots/smote_comparison.png")

plt.show()

# =========================================
# Scaling
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_smote)

X_test_scaled = scaler.transform(X_test)

# =========================================
# Final Shapes
# =========================================

print("\nFinal Shapes:")

print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled :", X_test_scaled.shape)

print("y_train_smote:", y_train_smote.shape)
print("y_test:", y_test.shape)

# =========================================
# Logistic Regression
# =========================================

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train
log_model.fit(X_train_scaled, y_train_smote)

# Predict
y_pred_log = log_model.predict(X_test_scaled)

# Metrics
accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

# Results
print("\n===== Logistic Regression =====")

print(f"Accuracy  : {accuracy_log:.4f}")
print(f"Precision : {precision_log:.4f}")
print(f"Recall    : {recall_log:.4f}")
print(f"F1-Score  : {f1_log:.4f}")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred_log))

# Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_log,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Logistic Regression - Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("plots/logistic_cm.png")

plt.show()

# =========================================
# Decision Tree
# =========================================

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

# Train
tree_model.fit(X_train_scaled, y_train_smote)

# Predict
y_pred_tree = tree_model.predict(X_test_scaled)

# Metrics
accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree)
recall_tree = recall_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)

# Results
print("\n===== Decision Tree =====")

print(f"Accuracy  : {accuracy_tree:.4f}")
print(f"Precision : {precision_tree:.4f}")
print(f"Recall    : {recall_tree:.4f}")
print(f"F1-Score  : {f1_tree:.4f}")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred_tree))

# Confusion Matrix
cm_tree = confusion_matrix(y_test, y_pred_tree)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_tree,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title("Decision Tree - Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("plots/tree_cm.png")

plt.show()

# =========================================
# SVM
# =========================================

from sklearn.svm import SVC

svm_model = SVC(
    probability=True,
    kernel='rbf',
    random_state=42
)

# Train
svm_model.fit(X_train_scaled, y_train_smote)

# Predict
y_pred_svm = svm_model.predict(X_test_scaled)

# Metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

# Results
print("\n===== SVM =====")

print(f"Accuracy  : {accuracy_svm:.4f}")
print(f"Precision : {precision_svm:.4f}")
print(f"Recall    : {recall_svm:.4f}")
print(f"F1-Score  : {f1_svm:.4f}")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred_svm))

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt='d',
    cmap='Reds'
)

plt.title("SVM - Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("plots/svm_cm.png")

plt.show()

# =========================================
# Neural Network
# =========================================

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

# =========================================
# Build Model
# =========================================

model = Sequential()

model.add(
    Dense(
        32,
        activation='relu',
        input_shape=(X_train_scaled.shape[1],)
    )
)

model.add(
    Dense(
        16,
        activation='relu'
    )
)

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)

# =========================================
# Compile Model
# =========================================

model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================================
# Early Stopping
# =========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================================
# Model Summary
# =========================================

model.summary()

# =========================================
# Train Model
# =========================================

history = model.fit(
    X_train_scaled,
    y_train_smote,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# =========================================
# Predict
# =========================================

y_pred_prob = model.predict(X_test_scaled)

y_pred_nn = (y_pred_prob > 0.5).astype(int).flatten()

# =========================================
# Metrics
# =========================================

accuracy_nn = accuracy_score(y_test, y_pred_nn)

precision_nn = precision_score(y_test, y_pred_nn)

recall_nn = recall_score(y_test, y_pred_nn)

f1_nn = f1_score(y_test, y_pred_nn)

# =========================================
# Results
# =========================================

print("\n===== Neural Network =====")

print(f"Accuracy  : {accuracy_nn:.4f}")
print(f"Precision : {precision_nn:.4f}")
print(f"Recall    : {recall_nn:.4f}")
print(f"F1-Score  : {f1_nn:.4f}")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred_nn))

# =========================================
# Confusion Matrix
# =========================================

cm_nn = confusion_matrix(y_test, y_pred_nn)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_nn,
    annot=True,
    fmt='d',
    cmap='Purples'
)

plt.title("Neural Network - Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("plots/nn_cm.png")

plt.show()

# =========================================
# Accuracy Plot
# =========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(['Train', 'Validation'])

plt.savefig("plots/nn_accuracy.png")

plt.show()

# =========================================
# Loss Plot
# =========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(['Train', 'Validation'])

plt.savefig("plots/nn_loss.png")

plt.show()

# =========================================
# Models Comparison
# =========================================

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "SVM",
        "Neural Network"
    ],

    "Accuracy": [
        accuracy_log,
        accuracy_tree,
        accuracy_svm,
        accuracy_nn
    ],

    "Precision": [
        precision_log,
        precision_tree,
        precision_svm,
        precision_nn
    ],

    "Recall": [
        recall_log,
        recall_tree,
        recall_svm,
        recall_nn
    ],

    "F1-Score": [
        f1_log,
        f1_tree,
        f1_svm,
        f1_nn
    ]
})

# =========================================
# Display Results
# =========================================

print("\n===== Models Comparison =====")

print(results)

# =========================================
# Save Results CSV
# =========================================

results.to_csv("models_results.csv", index=False)

# =========================================
# Accuracy Comparison Plot
# =========================================

plt.figure(figsize=(10,5))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=results
)

plt.title("Models Accuracy Comparison")

plt.savefig("plots/models_accuracy.png")

plt.show()

# =========================================
# Save Best Model
# =========================================

joblib.dump(svm_model, "heart_model.pkl")

joblib.dump(scaler, "scaler.pkl")

# =========================================
# Save Best Model Name
# =========================================

best_model = "SVM"

with open("best_model.txt", "w") as f:
    f.write(best_model)

# =========================================
# Saved Files Info
# =========================================

print("\nSaved Files:")
print("heart_model.pkl")
print("scaler.pkl")
print("features.pkl")
print("models_results.csv")
print("best_model.txt")
print("All plots saved in plots folder")

print("\nBest Model Saved Successfully!")

# =========================================
# Project Finished
# =========================================

print("\nHeart Disease Prediction Project Finished Successfully!")