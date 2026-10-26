import json
import pandas as pd
from evaluation.Consistency_ratios import ComputeRatio
from evaluation.Consistency_measures import calculate_abic

base = "uci_logs"
df = pd.read_csv(f"{base}/preprocessed_not_discretized_dataset_crx.csv")
features = [c for c in df.columns if c != "+"]

with open(f"{base}/reds_crx_python.json", encoding="utf-8") as f:
    reds_idx = json.load(f)
with open(f"{base}/core_crx_python.json", encoding="utf-8") as f:
    core_idx = json.load(f)

reds = {k: [features[i] for i in v] for k, v in reds_idx.items()}
core = [features[i] for i in core_idx]
shortest_name = min(reds, key=lambda k: len(reds[k]))
reduct = reds[shortest_name]

# Match the original pipeline's training sample size
sample_size = 552

def get_shap_topk(model, k):
    data = pd.read_csv(f"{base}/shap_values_crx_{model}.csv", sep=";")
    scores = data[features].abs().mean(axis=0).sort_values(ascending=False)
    return {name: float(score) for name, score in scores.head(k).items()}

def get_perm_topk(model, k):
    data = pd.read_csv(
        f"{base}/permutation_importance_crx_{model}_training.csv",
        sep=";"
    )
    # Repo's get_imp_features takes the first k rows.
    data = data[data["Feature"].isin(features)].head(k)
    return dict(zip(data["Feature"], data["importances(mean)"].astype(float)))

rows = []
for model in ["logit", "xgboost"]:
    for xai in ["shap", "perm"]:
        for set_type, reference in [("reduct", reduct), ("core", core)]:
            k = len(reference)
            selected = (
                get_shap_topk(model, k)
                if xai == "shap"
                else get_perm_topk(model, k)
            )

            ratio, intersection = ComputeRatio(
                selected, reference, "experimental"
            )
            params = k - intersection

            rows.append({
                "model": model,
                "xai": xai,
                "set_type": set_type,
                "reference_size": k,
                "selected_count": len(selected),
                "intersection_count": intersection,
                "experimental_ratio": ratio,
                "AIC": calculate_abic(sample_size, ratio, params, "AIC"),
                "BIC": calculate_abic(sample_size, ratio, params, "BIC"),
            })

result = pd.DataFrame(rows)
out = f"{base}/consistency_ratios_crx_xai.csv"
result.to_csv(out, index=False)

print("Target counts:")
print(df["+"].value_counts().to_string())
print(f"Positive class rate: {(df['+'] == 1).mean():.4f}")
print(f"Training sample size used for AIC/BIC: {sample_size}")
print(f"Shortest reduct: {shortest_name} ({len(reduct)} features)")
print("\nFinal experimental results:")
print(result.to_string(index=False))
print(f"\nSaved: {out}")
