
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from utils.retrieval import retrieve_artefact
from explaining.XAImethods import explain_predictions


# Tham số dự phòng chỉ dùng khi thiếu file optimal_params.
FALLBACK_PARAMS = {
    'logit': {'C': 0},
    'xgboost': {
        'learning_rate': 0.1,
        'subsample': 1.0,
        'max_depth': 6,
        'colsample_bytree': 1.0,
        'min_child_weight': 1
    },
    'rf': {'max_features': 'sqrt'},
    'gbm': {'learning_rate': 0.1}
}


def _load_params(params_dir, cls_method, file_name):
    path = os.path.join(
        params_dir,
        'optimal_params_%s_%s.pickle' % (cls_method, file_name)
    )

    if os.path.isfile(path):
        with open(path, 'rb') as fin:
            params = pickle.load(fin)

        print('  Dùng tham số tối ưu:', path)
        return params, False

    warnings.warn(
        'Thiếu %s; dùng FALLBACK_PARAMS cho %s. '
        'Kết quả không sử dụng tham số tối ưu của tác giả.'
        % (path, file_name)
    )

    return FALLBACK_PARAMS[cls_method].copy(), True


def _align_features(X_train, X_test, expected_features, file_name, model_name):
    """
    Đồng bộ thứ tự cột theo thứ tự feature mà model đã được huấn luyện.
    Chỉ sắp xếp lại cột, không thay đổi giá trị dữ liệu.
    """

    if expected_features is None:
        features = list(X_train.columns)
        return X_train.copy(), X_test.copy(), features

    expected_features = list(expected_features)
    current_features = list(X_train.columns)

    if (
        len(expected_features) != len(current_features)
        or set(expected_features) != set(current_features)
    ):
        missing = [
            f for f in expected_features
            if f not in current_features
        ]
        extra = [
            f for f in current_features
            if f not in expected_features
        ]

        raise ValueError(
            '\nFeature không khớp với model đã lưu!'
            '\nDataset: %s'
            '\nModel: %s'
            '\nThiếu feature: %s'
            '\nFeature thừa: %s'
            % (file_name, model_name, missing, extra)
        )

    X_train_aligned = X_train[expected_features].copy()
    X_test_aligned = X_test[expected_features].copy()

    return X_train_aligned, X_test_aligned, expected_features


def _get_model_features(clf, cls_method, scaler=None):
    """
    Lấy thứ tự feature từ model hoặc scaler đã lưu.
    """

    if cls_method == 'logit' and scaler is not None:
        return getattr(scaler, 'feature_names_in_', None)

    if cls_method == 'xgboost':
        try:
            return clf.get_booster().feature_names
        except Exception:
            return None

    return getattr(clf, 'feature_names_in_', None)


def train_models(
    models_folder,
    datasets,
    info_df,
    params_dir,
    datasets_folder,
    out_dir
):
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    random_state = 42
    performance_rows = []

    methods = ['logit', 'xgboost', 'rf', 'gbm']

    for ds in datasets:

        file_name = info_df.loc[
            info_df['Dataset_name'] == ds,
            'files_names'
        ].values[0]

        target_name = info_df.loc[
            info_df['Dataset_name'] == ds,
            'targets_names'
        ].values[0]

        print('\n========== %s (%s) ==========' % (ds, file_name))

        df = retrieve_artefact(
            datasets_folder,
            'preprocessed_not_discretized_dataset_%s' % file_name,
            '.csv',
            ','
        )

        y = df[target_name]
        X = df.drop(columns=[target_name])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state
        )

        score_path = os.path.join(
            out_dir,
            'classifiers_scores.txt'
        )

        with open(score_path, 'a', encoding='utf-8') as fout:
            fout.write('\n%s\n' % ds)
            fout.write('Training shape: %s\n' % (X_train.shape,))
            fout.write('Testing shape: %s\n' % (X_test.shape,))

        for cls_method in methods:

            print('\nHuấn luyện:', cls_method)

            params, used_fallback = _load_params(
                params_dir,
                cls_method,
                file_name
            )

            model_path = os.path.join(
                models_folder,
                'model_%s_%s.pickle' % (cls_method, file_name)
            )

            scaler_path = os.path.join(
                models_folder,
                'scaler_logit_%s.pickle' % file_name
            )

            model_exists = os.path.exists(model_path)
            scaler_exists = os.path.exists(scaler_path)

            scaler = None

            # ==================================================
            # TÁI SỬ DỤNG MODEL ĐÃ TỒN TẠI
            # ==================================================
            if model_exists and (
                cls_method != 'logit' or scaler_exists
            ):

                print('Tái sử dụng model đã có:', model_path)

                with open(model_path, 'rb') as fin:
                    clf = pickle.load(fin)

                if cls_method == 'logit':
                    with open(scaler_path, 'rb') as fin:
                        scaler = pickle.load(fin)

                # Lấy thứ tự feature của model đã lưu
                expected_features = _get_model_features(
                    clf,
                    cls_method,
                    scaler
                )

                # XGBoost lấy trực tiếp thứ tự feature từ Booster
                if cls_method == 'xgboost':
                    expected_features = clf.get_booster().feature_names

                X_train_aligned, X_test_aligned, model_features = (
                    _align_features(
                        X_train,
                        X_test,
                        expected_features,
                        file_name,
                        cls_method
                    )
                )

                if cls_method == 'logit':
                    X_train_model = scaler.transform(X_train_aligned)
                    X_test_model = scaler.transform(X_test_aligned)
                else:
                    X_train_model = X_train_aligned
                    X_test_model = X_test_aligned

                # Chốt thứ tự feature ngay trước khi dự đoán
                if cls_method == 'xgboost':
                    expected = clf.get_booster().feature_names

                    if expected is not None:
                        X_train_model = X_train_model.loc[:, expected]
                        X_test_model = X_test_model.loc[:, expected]
                        model_features = expected

            # ==================================================
            # HUẤN LUYỆN MODEL MỚI
            # ==================================================
            else:

                print('Huấn luyện model mới:', cls_method)

                X_train_model = X_train.copy()
                X_test_model = X_test.copy()
                model_features = list(X_train.columns)

                if cls_method == 'logit':

                    scaler = StandardScaler()

                    X_train_model = scaler.fit_transform(
                        X_train_model
                    )

                    X_test_model = scaler.transform(
                        X_test_model
                    )

                    clf = LogisticRegression(
                        C=2 ** params['C'],
                        random_state=random_state,
                        max_iter=1000
                    )

                elif cls_method == 'xgboost':

                    clf = xgb.XGBClassifier(
                        objective='binary:logistic',
                        n_estimators=500,
                        learning_rate=params['learning_rate'],
                        subsample=params['subsample'],
                        max_depth=int(params['max_depth']),
                        colsample_bytree=params['colsample_bytree'],
                        min_child_weight=int(params['min_child_weight']),
                        random_state=random_state,
                        eval_metric='logloss',
                        n_jobs=-1
                    )

                elif cls_method == 'rf':

                    clf = RandomForestClassifier(
                        n_estimators=500,
                        max_features=params['max_features'],
                        random_state=random_state,
                        n_jobs=-1
                    )

                else:

                    clf = GradientBoostingClassifier(
                        n_estimators=500,
                        learning_rate=params['learning_rate'],
                        random_state=random_state
                    )

                unfitted_path = os.path.join(
                    models_folder,
                    'unfitted_model_%s_%s.pickle'
                    % (cls_method, file_name)
                )

                with open(unfitted_path, 'wb') as fout:
                    pickle.dump(clf, fout)

                clf.fit(X_train_model, y_train)

                # Cập nhật feature theo model vừa huấn luyện
                if cls_method == 'xgboost':
                    booster_features = clf.get_booster().feature_names
                    if booster_features is not None:
                        model_features = booster_features

                with open(model_path, 'wb') as fout:
                    pickle.dump(clf, fout)

                if scaler is not None:
                    with open(scaler_path, 'wb') as fout:
                        pickle.dump(scaler, fout)

            # ==================================================
            # DỰ ĐOÁN VÀ ĐÁNH GIÁ
            # ==================================================
            y_pred = clf.predict(X_test_model)

            # Giữ nguyên công thức AUC của code hiện tại.
            auc = roc_auc_score(y_test, y_pred)

            f1 = f1_score(
                y_test,
                y_pred,
                average='binary',
                zero_division=0
            )

            cm = confusion_matrix(y_test, y_pred)

            performance_rows.append({
                'Dataset': ds,
                'Model': cls_method,
                'Testing_shape': str(X_test.shape),
                'F1_score': f1,
                'AUC': auc,
                'Used_fallback_params': used_fallback
            })

            with open(score_path, 'a', encoding='utf-8') as fout:

                fout.write('Model: %s\n' % cls_method)
                fout.write('Parameters: %s\n' % params)
                fout.write(
                    'Fallback parameters: %s\n' % used_fallback
                )
                fout.write('AUC: %.8f\n' % auc)
                fout.write('F1: %.8f\n' % f1)
                fout.write('Confusion matrix:\n%s\n' % cm)
                fout.write(
                    '----------------------------------------\n'
                )

            # ==================================================
            # XAI
            # ==================================================
            for xai_method in ['perm', 'shap', 'ALE']:

                explain_predictions(
                    xai_method,
                    file_name,
                    clf,
                    model_features,
                    X_train_model,
                    y_train,
                    cls_method
                )

    # ======================================================
    # TỔNG HỢP TABLE 3
    # ======================================================
    performance_df = pd.DataFrame(performance_rows)

    wide = performance_df.pivot(
        index='Dataset',
        columns='Model',
        values=['F1_score', 'AUC']
    )

    wide.columns = [
        '%s_%s' % (model, metric)
        for metric, model in wide.columns
    ]

    test_shapes = (
        performance_df[['Dataset', 'Testing_shape']]
        .drop_duplicates()
        .set_index('Dataset')
    )

    wide = test_shapes.join(wide).reset_index()

    wide.to_csv(
        os.path.join(
            out_dir,
            'Table3_model_performance.csv'
        ),
        index=False,
        encoding='utf-8-sig'
    )

    performance_df.to_csv(
        os.path.join(
            out_dir,
            'Table3_model_performance_long.csv'
        ),
        index=False,
        encoding='utf-8-sig'
    )

    print('\nĐã lưu Table3_model_performance.csv')

    return performance_df
