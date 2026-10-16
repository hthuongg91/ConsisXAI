import pickle
import pandas as pd
from scipy.io import arff
import json
import os
import numpy as np


def retrieve_vector(folder, fname):
    with open(os.path.join(folder,fname)) as f:
        return json.load(f)


# a function to retrieve artefacts
def retrieve_artefact(folder, filename, file_end, vals_sep=None):
    retrieved_file = filename + file_end
    file = os.path.join(folder, retrieved_file)
    if '.pickle' in file_end:
        with open(file, 'rb') as fin:
            retrieved_artefact = pickle.load(fin)
    elif '.data' in file_end:
        retrieved_artefact = pd.read_csv(file, sep=vals_sep, engine='python')
    elif '.arff' in file_end:
        data = arff.loadarff(file)
        retrieved_artefact = pd.DataFrame(data[0])
    else:
        retrieved_artefact = pd.read_csv(file, sep=vals_sep, encoding='ISO-8859-1', engine='python')
    return retrieved_artefact

def get_imp_features(
    folder,
    file_name,
    ffeatures,
    cls_method,
    num_retrieved_feats,
    xai_type=None
):
    frmt_str = {
        'ALE': 'ALE_pred_explainer_%s_%s' % (file_name, cls_method),
        'shap': 'shap_values_%s_%s' % (file_name, cls_method),
        'perm': 'permutation_importance_%s_%s_training' % (
            file_name, cls_method
        )
    }

    if xai_type is None:
        raise ValueError("xai_type must be ALE, shap, or perm")

    if xai_type not in frmt_str:
        raise ValueError("XAI method not found: " + str(xai_type))

    feats_df = retrieve_artefact(
        folder, frmt_str[xai_type], '.csv', ';'
    )

    local_feats = num_retrieved_feats

    if xai_type == 'shap':
        # Mỗi cột là một đặc trưng, mỗi dòng là một mẫu.
        feature_names = list(feats_df.columns)
        shap_scores = np.mean(np.abs(feats_df.to_numpy()), axis=0)

        feats_df_sorted = pd.DataFrame({
            'Feature': feature_names,
            'Score': shap_scores
        })

        feats_df_sorted = feats_df_sorted.sort_values(
            by='Score', ascending=False
        )

    elif xai_type == 'perm':
        # Sắp xếp theo importance mean giảm dần.
        feats_df_sorted = feats_df[
            ['Feature', 'importances(mean)']
        ].copy()

        feats_df_sorted = feats_df_sorted.sort_values(
            by='importances(mean)', ascending=False
        )

        feats_df_sorted.columns = ['Feature', 'Score']

    elif xai_type == 'ALE':
        # Giữ cách lấy thứ tự như định dạng đầu ra ALE của repo.
        feats_df_sorted = feats_df.copy()

        local_feats = min(
            num_retrieved_feats, len(feats_df_sorted)
        )

        feats_df_sorted = feats_df_sorted.iloc[:local_feats, :]

        feats_df_sorted = pd.DataFrame({
            'Feature': feats_df_sorted.iloc[:, 0],
            'Score': feats_df_sorted.iloc[:, -1]
        })

    local_feats = min(local_feats, len(feats_df_sorted))

    if local_feats <= 0:
        return {}

    selected = feats_df_sorted.iloc[:local_feats]

    return dict(zip(selected['Feature'], selected['Score']))



