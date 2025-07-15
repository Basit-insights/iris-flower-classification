import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load data
df = pd.read_csv('Iris.csv')

# Step 2: Prepare data
X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 3: Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Step 5: Decode predicted species
predicted_species = le.inverse_transform(y_pred)
print(predicted_species)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming y_test and y_pred are defined

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("Classification Report:")
print(report)

