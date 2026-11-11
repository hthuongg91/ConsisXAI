
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import hyperopt
from hyperopt import hp, tpe, Trials, STATUS_OK, fmin
from hyperopt.pyll.base import scope
from utils.retrieval import retrieve_artefact
import ast
import pickle
import os

datasets_folder = 'datasets_files'
logs_dir = 'uci_logs'
models_folder = os.path.join(logs_dir, 'models')
params_dir = 'optimal_params'
datasets_folder = logs_dir

info_df = pd.read_csv(
    os.path.join(logs_dir, 'datasets_info.csv'),
    sep=';'
)

datasets = [
    'Diabetic', 'Ionosphere', 'spect', 'Kidney', 'Credit',
    'Climate', 'Adult', 'TruckFailure', 'Spam'
]

params_dir = 'optimal_params'
if not os.path.exists(params_dir):
    os.makedirs(params_dir)

print(os.path.abspath(params_dir))


def create_and_evaluate_model(args):
    global trial_nr, file_name

    trial_nr += 1

    if cls_method == "xgboost":
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

    elif cls_method == "logit":
        cls = LogisticRegression(
            C=2**args['C'],
            random_state=random_state,
            max_iter=1000
        )

    elif cls_method == 'gbm':
        cls = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=args['learning_rate'],
            random_state=random_state
        )

    score = cross_val_score(
        cls,
        X,
        y,
        cv=5,
        n_jobs=-1
    ).mean()

    for k, v in args.items():
        fout_all.write(
            "%s;%s;%s;%s;%s;%s\n"
            % (trial_nr, file_name, cls_method, k, v, score)
        )

    fout_all.flush()

    return {'loss': -score, 'status': STATUS_OK}


######################################################################
n_iter = 50
random_state = 42

space_args = {
    'xgboost': {
        'learning_rate': hp.uniform("learning_rate", 0, 1),
        'subsample': hp.uniform("subsample", 0.5, 1),
        'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
        'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
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
        'learning_rate': hp.uniform("learning_rate", 0, 1)
    }
}


for ds in datasets:
    file_name = info_df.loc[
        info_df['Dataset_name'] == ds,
        'files_names'
    ].values[0]

    target_name = info_df.loc[
        info_df['Dataset_name'] == ds,
        'targets_names'
    ].values[0]

    df = retrieve_artefact(
        logs_dir,
        'preprocessed_not_discretized_dataset_%s' % file_name,
        '.csv',
        ','
    )

    y = df[target_name]
    X = df.drop([target_name], axis=1)

    for cls_method in ['logit', 'xgboost', 'rf', 'gbm']:
        outfile = os.path.join(
            params_dir,
            "optimal_params_%s_%s.pickle" % (cls_method, file_name)
        )

        if os.path.exists(outfile):
            print("SKIP:", cls_method, file_name)
            continue

        print("RUN:", cls_method, file_name)

        if cls_method == "logit":
            sscaler = StandardScaler()
            X_model = sscaler.fit_transform(X)
        else:
            X_model = X

        trial_nr = 0
        trials = Trials()
        space = space_args[cls_method]

        output_file = os.path.join(
            logs_dir,
            "param_optim_all_trials_%s_%s.csv"
            % (cls_method, file_name)
        )

        with open(output_file, "w", encoding="utf-8") as fout_all:
            fout_all.write(
                "%s;%s;%s;%s;%s;%s\n"
                % ("iter", "dataset", "cls", "param", "value", "score")
            )

            X_original = X
            X = X_model

            try:
                current_n_iter = 10 if ds == 'TruckFailure' else n_iter
                best = fmin(
                    create_and_evaluate_model,
                    space,
                    algo=tpe.suggest,
                    max_evals=n_iter,
                    trials=trials
                )
            finally:
                X = X_original

        best_params = hyperopt.space_eval(space, best)
        print('best params:', best_params)

        with open(outfile, "wb") as fout:
            pickle.dump(best_params, fout)