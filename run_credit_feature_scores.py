import os
import pickle
import pandas as pd
import preprocessing.Indispensable_feats_gen as ig

base = "uci_logs"
file_name = "crx"
ig.logs_dir = base
ig.file_name = file_name

df = pd.read_csv(os.path.join(base, "preprocessed_not_discretized_dataset_crx.csv"))
target = "+"
X = df.drop(columns=[target])
y = df[target]
features = X.columns.tolist()

df_disc = pd.read_csv(
    os.path.join(base, "discretized_dataset_crx.csv"),
    header=None
)
df_disc.columns = df.columns

X_train, X_test, y_train, y_test = __import__(
    "sklearn.model_selection", fromlist=["train_test_split"]
).train_test_split(X, y, test_size=0.2, random_state=42)

models = {}
for name in ["logit", "xgboost", "rf", "gbm"]:
    path = os.path.join(base, "models", f"model_{name}_crx.pickle")
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

print("Feature count:", len(features))
print("Discretized shape:", df_disc.shape)
print("Target column:", target)

ig.compute_features_importances(
    "Credit", X_train, y_train, features, target,
    models["xgboost"], models["logit"], models["rf"], models["gbm"],
    df_disc, [], features
)

print("Finished feature scoring.")
print("Output exists:", os.path.exists(os.path.join(base, "features_scores_crx.csv")))
