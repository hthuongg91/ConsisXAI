
import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

base = "uci_logs"
model_dir = os.path.join(base, "models")
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(
    os.path.join(base, "preprocessed_not_discretized_dataset_crx.csv")
)

target = "+"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = [
    ("rf", RandomForestClassifier),
    ("gbm", GradientBoostingClassifier),
]

for name, cls in models:
    param_path = os.path.join(
        base, f"optimal_params_{name}_crx.pickle"
    )

    with open(param_path, "rb") as f:
        params = pickle.load(f)

    model = cls(**params, random_state=42)
    model.fit(X_train, y_train)

    model_path = os.path.join(
        model_dir, f"model_{name}_crx.pickle"
    )

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"{name.upper()} saved: {model_path}")
    print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
