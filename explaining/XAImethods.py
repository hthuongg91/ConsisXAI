
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import shap

from sklearn.inspection import permutation_importance


logs_dir = 'uci_logs'


def _prepare_input(X, ffeatures):
    """
    Chuẩn hóa dữ liệu và bảo đảm số lượng, thứ tự feature chính xác.
    """

    if isinstance(X, pd.DataFrame):

        if (
            len(ffeatures) != X.shape[1]
            or set(ffeatures) != set(X.columns)
        ):
            raise ValueError(
                'Feature không khớp giữa dữ liệu và danh sách ffeatures.'
            )

        X_input = X.loc[:, ffeatures].copy()

        X_input = X_input.apply(
            pd.to_numeric,
            errors='raise'
        ).astype(np.float64)

    else:

        X_array = np.asarray(X, dtype=np.float64)

        if X_array.ndim != 2:
            raise ValueError('Dữ liệu đầu vào phải có 2 chiều.')

        if X_array.shape[1] != len(ffeatures):
            raise ValueError(
                'Số feature trong dữ liệu không khớp ffeatures.'
            )

        X_input = pd.DataFrame(
            X_array,
            columns=ffeatures
        )

    return X_input


def _save_pickle(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as fout:
        pickle.dump(obj, fout)


def _save_dataframe(path, df, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, **kwargs)


def shap_global(file_name, cls, X, ffeatures, cls_method):

    # Chuẩn hóa và đồng bộ thứ tự feature.
    X_shap = _prepare_input(X, ffeatures)

    X_values = X_shap.to_numpy(dtype=np.float64)

    print('\n========== KIỂM TRA DỮ LIỆU SHAP ==========')
    print('Kiểu dữ liệu X:', type(X_shap))
    print('Shape:', X_shap.shape)
    print('Dtype:', X_shap.dtypes.unique())

    # Logit được huấn luyện bằng NumPy array.
    # Vì vậy SHAP cũng truyền NumPy array vào Logit.
    if cls_method == 'logit':

        predictions = cls.predict_proba(X_values)

    else:

        predictions = cls.predict_proba(X_shap)

    print('Predict_proba shape:', predictions.shape)
    print('Predict_proba dtype:', predictions.dtype)
    print('================================================')

    # ======================================================
    # TÍNH SHAP
    # ======================================================

    if cls_method in ['xgboost', 'logit']:

        if cls_method == 'logit':

            # Dùng NumPy để khớp với cách Logit được huấn luyện.
            explainer = shap.Explainer(
                cls.predict_proba,
                X_values,
                feature_names=ffeatures
            )

            shap_result = explainer(X_values)

        else:

            # XGBoost cần DataFrame có đúng tên và thứ tự feature.
            explainer = shap.Explainer(
                cls.predict_proba,
                X_shap
            )

            shap_result = explainer(X_shap)

        shap_values = shap_result.values

    else:

        explainer = shap.TreeExplainer(cls)
        shap_result = explainer(X_shap)
        shap_values = shap_result.values

    # Chọn SHAP values của lớp positive (class 1).
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    if shap_values.ndim != 2:
        raise ValueError(
            'SHAP values có shape không hợp lệ: %s'
            % (shap_values.shape,)
        )

    if shap_values.shape[1] != len(ffeatures):
        raise ValueError(
            'Số feature SHAP không khớp danh sách feature.'
        )

    shap_vals = shap.Explanation(
        values=shap_values,
        data=X_values,
        feature_names=ffeatures
    )

    print('\nType of SHAP values:', type(shap_values))
    print('SHAP values shape:', shap_values.shape)

    # ======================================================
    # LƯU EXPLAINER
    # ======================================================

    out1 = os.path.join(
        logs_dir,
        r'shap_explainer\_%s\_%s.pickle'
        % (file_name, cls_method)
    )

    _save_pickle(out1, explainer)

    # ======================================================
    # LƯU SHAP VALUES CSV
    # ======================================================

    shap_csv = os.path.join(
        logs_dir,
        r'shap_values\_%s\_%s.csv'
        % (file_name, cls_method)
    )

    _save_dataframe(
        shap_csv,
        pd.DataFrame(
            shap_values,
            columns=ffeatures
        ),
        sep=';',
        index=False
    )

    # ======================================================
    # LƯU SHAP VALUES PICKLE
    # ======================================================

    shap_data = os.path.join(
        logs_dir,
        r'shap_values\_%s\_%s.pickle'
        % (file_name, cls_method)
    )

    _save_pickle(shap_data, shap_values)

    # ======================================================
    # SUMMARY PLOT - BEESWARM
    # ======================================================

    print('Summary plot of SHAP values - normal')

    shap.plots.beeswarm(
        shap_vals,
        max_display=10,
        show=False
    )

    normal_plot = os.path.join(
        logs_dir,
        r'Shap values_normal\_%s\_%s.png'
        % (file_name, cls_method)
    )

    os.makedirs(os.path.dirname(normal_plot), exist_ok=True)

    plt.savefig(
        normal_plot,
        dpi=300,
        bbox_inches='tight'
    )

    plt.clf()
    plt.close()

    # ======================================================
    # SUMMARY PLOT - BAR
    # ======================================================

    print('-' * 100)
    print('Summary plot of SHAP values - bar')

    shap.plots.bar(
        shap_vals,
        max_display=10,
        show=False
    )

    bar_plot = os.path.join(
        logs_dir,
        r'Shap values_bar\_%s\_%s.png'
        % (file_name, cls_method)
    )

    os.makedirs(os.path.dirname(bar_plot), exist_ok=True)

    plt.savefig(
        bar_plot,
        dpi=300,
        bbox_inches='tight'
    )

    plt.clf()
    plt.close()

    del explainer
    del shap_values


def Permutation_importance_analysis(
    file_name,
    cls,
    ffeatures,
    x,
    y,
    cls_method
):

    permutation_file_name = (
        r'permutation_importance\_%s\_%s'
        % (file_name, cls_method)
    )

    X_perm = _prepare_input(x, ffeatures)

    # Logit được huấn luyện bằng NumPy array.
    if cls_method == 'logit':
        X_perm = X_perm.to_numpy(dtype=np.float64)

    training_result = permutation_importance(
        cls,
        X_perm,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    cols = [
        'Feature',
        'importances(mean)',
        'importances(std)',
        'importances'
    ]

    df_res_train = pd.DataFrame(
        zip(
            ffeatures,
            training_result.importances_mean,
            training_result.importances_std,
            training_result.importances
        ),
        columns=cols
    )

    df_train_sorted = df_res_train.sort_values(
        'importances(mean)',
        ascending=False
    )

    csv_path = os.path.join(
        logs_dir,
        '%s_training.csv' % permutation_file_name
    )

    _save_dataframe(
        csv_path,
        df_train_sorted,
        sep=';',
        index=False
    )

    # ======================================================
    # PLOT PERMUTATION IMPORTANCE
    # ======================================================

    plt.figure(figsize=(18, 8))

    plt.barh(
        np.arange(0, len(df_train_sorted['Feature'])),
        df_train_sorted['importances(mean)'],
        align='center',
        alpha=0.5
    )

    plt.yticks(
        np.arange(0, len(df_train_sorted['Feature'])),
        df_train_sorted['Feature']
    )

    plt.xlabel('Importances')
    plt.title('Permutation Importances (train set)')

    plot_path = os.path.join(
        logs_dir,
        '%s_training2.png' % permutation_file_name
    )

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


def explain_predictions(
    xai_method,
    file_name,
    clf,
    ffeatures,
    X,
    y,
    clf_type
):

    if xai_method == 'perm':

        Permutation_importance_analysis(
            file_name,
            clf,
            ffeatures,
            X,
            y,
            clf_type
        )

    elif xai_method == 'shap':

        shap_global(
            file_name,
            clf,
            X,
            ffeatures,
            clf_type
        )

    elif xai_method == 'ALE':

        print(
            'ALE chưa được triển khai trong explain_predictions.'
        )
