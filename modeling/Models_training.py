import os
import pickle
import ast
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from utils.retrieval import retrieve_artefact
from explaining.XAImethods import explain_predictions


# Tham số dự phòng CHỈ dùng khi không có file optimal_params tương ứng.
# Các giá trị này không tương đương với kết quả Hyperopt của tác giả.
FALLBACK_PARAMS = {
    'logit': {'C': 0},  # LogisticRegression(C=2**0 = 1)
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
        params_dir, 'optimal_params_%s_%s.pickle' % (cls_method, file_name)
    )
    if os.path.isfile(path):
        with open(path, 'rb') as fin:
            params = pickle.load(fin)
        print('  Dùng tham số tối ưu:', path)
        return params, False

    warnings.warn(
        'Thiếu %s; dùng FALLBACK_PARAMS cho %s. Kết quả không phải '
        'tái lập chính xác tham số tối ưu của tác giả.' % (path, file_name)
    )
    return FALLBACK_PARAMS[cls_method].copy(), True


def train_models(models_folder, datasets, info_df, params_dir,
                 datasets_folder, out_dir):
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    random_state = 42
    performance_rows = []
    methods = ['logit', 'xgboost', 'rf', 'gbm']

    for ds in datasets:
        file_name = info_df.loc[
            info_df['Dataset_name'] == ds, 'files_names'
        ].values[0]
        target_name = info_df.loc[
            info_df['Dataset_name'] == ds, 'targets_names'
        ].values[0]

        print('\\n========== %s (%s) ==========' % (ds, file_name))
        df = retrieve_artefact(
            datasets_folder,
            'preprocessed_not_discretized_dataset_%s' % file_name,
            '.csv',
            ','
        )
        y = df[target_name]
        X = df.drop(columns=[target_name])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        score_path = os.path.join(out_dir, 'classifiers_scores.txt')
        with open(score_path, 'a', encoding='utf-8') as fout:
            fout.write('\\n%s\\n' % ds)
            fout.write('Training shape: %s\\n' % (X_train.shape,))
            fout.write('Testing shape: %s\\n' % (X_test.shape,))

        for cls_method in methods:
            print('Huấn luyện:', cls_method)
            params, used_fallback = _load_params(
                params_dir, cls_method, file_name
            )

            # Mỗi mô hình nhận bản dữ liệu riêng để bước scale của Logit
            # không làm thay đổi dữ liệu đầu vào của các mô hình còn lại.
                        # Tạo đường dẫn lưu model
            model_path = os.path.join(
                models_folder,
                'model_%s_%s.pickle' % (cls_method, file_name)
            )

            scaler_path = os.path.join(
                models_folder,
                'scaler_logit_%s.pickle' % file_name
            )

            # Kiểm tra model đã tồn tại chưa
            model_exists = os.path.exists(model_path)
            scaler_exists = os.path.exists(scaler_path)

            # Chỉ tái sử dụng Logit khi có cả model và scaler
            if model_exists and (cls_method != 'logit' or scaler_exists):
                print('Tái sử dụng model đã có:', model_path)

                with open(model_path, 'rb') as fin:
                    clf = pickle.load(fin)

                X_train_model = X_train.copy()
                X_test_model = X_test.copy()

                if cls_method == 'logit':
                    with open(scaler_path, 'rb') as fin:
                        scaler = pickle.load(fin)

                    X_train_model = scaler.transform(X_train_model)
                    X_test_model = scaler.transform(X_test_model)

            else:
                print('Huấn luyện model mới:', cls_method)

                X_train_model = X_train.copy()
                X_test_model = X_test.copy()
                scaler = None

                if cls_method == 'logit':
                    scaler = StandardScaler()
                    X_train_model = scaler.fit_transform(X_train_model)
                    X_test_model = scaler.transform(X_test_model)

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

                else:  # gbm
                    clf = GradientBoostingClassifier(
                        n_estimators=500,
                        learning_rate=params['learning_rate'],
                        random_state=random_state
                    )

                # Lưu model chưa huấn luyện
                unfitted_path = os.path.join(
                    models_folder,
                    'unfitted_model_%s_%s.pickle' % (cls_method, file_name)
                )

                with open(unfitted_path, 'wb') as fout:
                    pickle.dump(clf, fout)

                # Huấn luyện và lưu model
                clf.fit(X_train_model, y_train)

                with open(model_path, 'wb') as fout:
                    pickle.dump(clf, fout)

                # Lưu scaler của Logit
                if scaler is not None:
                    with open(scaler_path, 'wb') as fout:
                        pickle.dump(scaler, fout)

            y_pred = clf.predict(X_test_model)
            if hasattr(clf, 'predict_proba'):
                y_score = clf.predict_proba(X_test_model)[:, 1]
            else:
                y_score = clf.decision_function(X_test_model)

            auc = roc_auc_score(y_test, y_score)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
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
                fout.write('Model: %s\\n' % cls_method)
                fout.write('Parameters: %s\\n' % params)
                fout.write('Fallback parameters: %s\\n' % used_fallback)
                fout.write('AUC: %.8f\\n' % auc)
                fout.write('F1: %.8f\\n' % f1)
                fout.write('Confusion matrix:\\n%s\\n' % cm)
                fout.write('----------------------------------------\\n')

            # Tạo giải thích cần cho các bước ConsisXAI tiếp theo.
            # Với Logit, X_train_model là dữ liệu đã scale, đúng với model.
            for xai_method in ['perm', 'shap', 'ALE']:
                explain_predictions(
                    xai_method, file_name, clf,
                    X.columns.tolist(), X_train_model, y_train, cls_method
                )

    performance_df = pd.DataFrame(performance_rows)
    # Dạng rộng, gần bố cục Table 3 trong bài báo.
    wide = performance_df.pivot(
        index='Dataset', columns='Model', values=['F1_score', 'AUC']
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
        os.path.join(out_dir, 'Table3_model_performance.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    # Lưu thêm dạng dài để dễ phân tích/đối chiếu.
    performance_df.to_csv(
        os.path.join(out_dir, 'Table3_model_performance_long.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    print('\\nĐã lưu Table3_model_performance.csv')
    return performance_df
