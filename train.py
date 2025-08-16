import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

data = pd.read_csv("data/StudentsPerformance.csv")

all_features = ["math score", "reading score", "writing score", "gender", 
                "parental level of education", "lunch", "test preparation course"]

X_processed = pd.get_dummies(data[all_features], drop_first=True)
X = X_processed

le = LabelEncoder()
y = le.fit_transform(data["race/ethnicity"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,         
    random_state=42,
    class_weight='balanced',   
    n_jobs=-1                  
)

print(" Training RandomForestClassifier...")
model.fit(X_train, y_train)
print(" Training complete.")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

output_dir = "model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

joblib.dump(model, os.path.join(output_dir, "race_ethnicity_rf.pkl"))
joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))
print(f" Model & encoder saved in '{output_dir}/' folder")