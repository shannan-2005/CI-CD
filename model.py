import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Step 1: Load dataset ---
# Make sure your dataset has a "label" column
data = pd.read_csv('winequality_FS')  

# Features and target
X = data.drop('quality', axis=1)
y = data['quality']

# --- Step 2: Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 3: Train classifier ---
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Step 4: Evaluate model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
