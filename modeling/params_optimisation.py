import os
import pickle

import hyperopt
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from utils.retrieval import retrieve_artefact


# Thư mục dữ liệu và kết quả
logs_dir = 'uci_logs'
params_dir = 'optimal_params'

os.makedirs(params_dir, exist_ok=True)

info_path = os.path.join(logs_dir, 'datasets_info.csv')
if not os.path.exists(info_path):
    raise FileNotFoundError(
        f"Không tìm thấy {info_path}. Hãy chạy definitions.py trước."
    )

info_df = pd.read_csv(info_path, sep=';')

# 7 dataset được chọn cho thực nghiệm
datasets = [
    'Diabetic',
    'Ionosphere',
    'spect',
    'Kidney',
    'Credit',
    'Climate',
    'Spam'
]

# Số lần thử và random state theo thiết lập của tác giả
n_iter = 50
random_state = 42


def create_and_evaluate_model(args):
    global trial_nr, file_name, cls_method, X, y, fout_all

    trial_nr += 1

    if cls_method == 'xgboost':
        cls = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            learning_rate=args['learning_rate'],
            subsample=args['subsample'],
            max_depth=int(args['max_depth']),
            colsample_bytree=args['colsample_bytree'],
            min_child_weight=int(args['min_child_weight']),
            seed=random_state,
            eval_metric='logloss'
        )

    elif cls_method == 'rf':
        cls = RandomForestClassifier(
            n_estimators=500,
            max_features=args['max_features'],
            random_state=random_state,
            n_jobs=-1
        )

    elif cls_method == 'logit':
        cls = LogisticRegression(
            C=2 ** args['C'],
            random_state=random_state,
            max_iter=1000
        )

    elif cls_method == 'gbm':
        cls = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=args['learning_rate'],
            random_state=random_state
        )

    else:
        raise ValueError(f'Classifier không được hỗ trợ: {cls_method}')

    # Giữ nguyên cách đánh giá của tác giả: cross-validation 5 folds,
    # tối ưu accuracy mặc định của cross_val_score.
    score = cross_val_score(
        cls,
        X,
        y,
        cv=5,
        n_jobs=-1
    ).mean()

    for param_name, param_value in args.items():
        fout_all.write(
            '%s;%s;%s;%s;%s;%s\n'
            % (
                trial_nr,
                file_name,
                cls_method,
                param_name,
                param_value,
                score
            )
        )

    fout_all.flush()
    return {'loss': -score, 'status': STATUS_OK}


# Không gian tìm kiếm giữ nguyên như bản gốc
space_args = {
    'xgboost': {
        'learning_rate': hp.uniform('learning_rate', 0, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_weight': scope.int(
            hp.quniform('min_child_weight', 1, 6, 1)
        )
    },
    'logit': {
        'C': hp.loguniform('C', -5, 5)
    },
    'rf': {
        'max_features': hp.uniform('max_features', 0, 1)
    },
    'gbm': {
        'learning_rate': hp.uniform('learning_rate', 0, 1)
    }
}


for ds in datasets:
    matching_rows = info_df.loc[info_df['Dataset_name'] == ds]

    if matching_rows.empty:
        raise ValueError(
            f"Dataset '{ds}' không có trong {info_path}. "
            "Kiểm tra Dataset_name trong datasets_info.csv."
        )

    file_name = matching_rows['files_names'].iloc[0]
    target_name = matching_rows['targets_names'].iloc[0]

    preprocessed_file = os.path.join(
        logs_dir,
        f'preprocessed_not_discretized_dataset_{file_name}.csv'
    )

    if not os.path.exists(preprocessed_file):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu đã tiền xử lý: {preprocessed_file}. "
            "Hãy kiểm tra definitions.py."
        )

    df = retrieve_artefact(
        logs_dir,
        f'preprocessed_not_discretized_dataset_{file_name}',
        '.csv',
        ','
    )

    if target_name not in df.columns:
        raise ValueError(
            f"Không tìm thấy cột target '{target_name}' trong "
            f"{preprocessed_file}. Các cột hiện có: {df.columns.tolist()}"
        )

    y = df[target_name]
    X_original = df.drop(columns=[target_name])

    for cls_method in ['logit', 'xgboost', 'rf', 'gbm']:
        outfile = os.path.join(
            params_dir,
            f'optimal_params_{cls_method}_{file_name}.pickle'
        )

        # Giữ cơ chế bỏ qua nếu đã có kết quả tối ưu.
        if os.path.exists(outfile):
            print(f'SKIP: {cls_method} - {file_name} (đã có params)')
            continue

        print(f'RUN: {cls_method} - {file_name}')

        # Chỉ chuẩn hóa đầu vào của Logistic Regression.
        # Các model còn lại nhận dữ liệu gốc, không bị ảnh hưởng bởi Logit.
        if cls_method == 'logit':
            scaler = StandardScaler()
            X = scaler.fit_transform(X_original)
        else:
            X = X_original

        trial_nr = 0
        trials = Trials()
        space = space_args[cls_method]

        output_file = os.path.join(
            logs_dir,
            f'param_optim_all_trials_{cls_method}_{file_name}.csv'
        )

        with open(output_file, 'w', encoding='utf-8') as fout_all:
            fout_all.write('iter;dataset;cls;param;value;score\n')

            best = fmin(
                fn=create_and_evaluate_model,
                space=space,
                algo=tpe.suggest,
                max_evals=n_iter,
                trials=trials
            )

        best_params = hyperopt.space_eval(space, best)
        print('Best params:', best_params)

        with open(outfile, 'wb') as fout:
            pickle.dump(best_params, fout)

print('Hoàn thành tối ưu tham số.')
