from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

logs_dir = 'uci_logs'


def shap_global(file_name, cls, X, ffeatures, cls_method):

    # Chuẩn hóa dữ liệu đầu vào cho SHAP
    if isinstance(X, pd.DataFrame):
        X_shap = X.apply(pd.to_numeric, errors='raise').astype(np.float64)
    else:
        X_shap = pd.DataFrame(
            np.asarray(X, dtype=np.float64),
            columns=ffeatures
        )

    print("\n========== KIỂM TRA DỮ LIỆU SHAP ==========")
    print("Kiểu dữ liệu X:", type(X_shap))
    print("Shape:", X_shap.shape)
    print("Dtype:", X_shap.dtypes.unique())

    predictions = cls.predict_proba(X_shap)
    print("Predict_proba shape:", predictions.shape)
    print("Predict_proba dtype:", predictions.dtype)
    print("================================================")

    # Tính SHAP
    if cls_method in ['xgboost', 'logit']:
        explainer = shap.Explainer(cls.predict_proba, X_shap)
        shap_result = explainer(X_shap)
        shap_values = shap_result.values

    else:
        explainer = shap.TreeExplainer(cls)
        shap_result = explainer(X_shap)
        shap_values = shap_result.values

    # Chọn SHAP values của lớp positive (class 1)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_vals = shap.Explanation(
        values=shap_values,
        data=X_shap.to_numpy(),
        feature_names=ffeatures
    )

    print('\n type of shap values: {}'.format(type(shap_values)))

    # Lưu explainer
    out1 = os.path.join(
        logs_dir,
        'shap_explainer_%s_%s.pickle' % (file_name, cls_method)
    )
    with open(out1, 'wb') as output:
        pickle.dump(explainer, output)

    # Lưu SHAP values dạng CSV
    shap_csv = os.path.join(
        logs_dir,
        'shap_values_%s_%s.csv' % (file_name, cls_method)
    )
    pd.DataFrame(
        shap_values,
        columns=ffeatures
    ).to_csv(shap_csv, sep=';', index=False)

    # Lưu SHAP values dạng pickle
    shap_data = os.path.join(
        logs_dir,
        'shap_values_%s_%s.pickle' % (file_name, cls_method)
    )
    with open(shap_data, 'wb') as fout:
        pickle.dump(shap_values, fout)

    # Summary plot - beeswarm
    print('summary plot of shap values - normal')

    shap.plots.beeswarm(
        shap_vals,
        max_display=10,
        show=False
    )

    plt.savefig(
        os.path.join(
            logs_dir,
            'Shap values_normal_%s_%s.png' % (file_name, cls_method)
        ),
        dpi=300,
        bbox_inches='tight'
    )
    plt.clf()
    plt.close()

    # Summary plot - bar
    print('-' * 100)
    print('summary plot of shap values - bar')

    shap.plots.bar(
        shap_vals,
        max_display=10,
        show=False
    )

    plt.savefig(
        os.path.join(
            logs_dir,
            'Shap values_bar_%s_%s.png' % (file_name, cls_method)
        ),
        dpi=300,
        bbox_inches='tight'
    )
    plt.clf()
    plt.close()

    del explainer
    del shap_values

##########################################################################################################################################
def Permutation_importance_analysis(file_name,cls, ffeatures, x, y, cls_method):
       
    permutation_file_name = 'permutation_importance_%s_%s' %(file_name,cls_method)
    training_result = permutation_importance(cls,x, y, n_repeats=10, random_state=42, n_jobs=-1)
    perm_sorted =  training_result.importances_mean.argsort()


    cols = ['Feature', 'importances(mean)', 'importances(std)', 'importances']
    df_res_train = pd.DataFrame(zip(ffeatures, training_result.importances_mean, \
                                    training_result.importances_std, training_result.importances), \
                                columns=cols)
    df_train_sorted = df_res_train.sort_values('importances(mean)', ascending=False)
    df_train_sorted.to_csv(os.path.join(logs_dir,'%s_training.csv' % (permutation_file_name)), sep=';', index=False)

    plt.figure(figsize=(18,8))
    plt.barh(np.arange(0, len(df_train_sorted['Feature'])), df_train_sorted.iloc[:, 1], align='center', alpha=0.5)
    plt.yticks(np.arange(0, len(df_train_sorted['Feature'])), df_train_sorted.iloc[:, 0])
    plt.xlabel('Importances')
    plt.title("Permutation Importances (train set)");
    plt.savefig(os.path.join(logs_dir,'%s_training2.png' % (permutation_file_name)), dpi=300,
                bbox_inches='tight');
    plt.close()

###############################################################################################################################################
def explain_predictions(xai_method, file_name, clf, ffeatures, X, y, clf_type):
    if xai_method == 'perm':
        Permutation_importance_analysis(
            file_name, clf, ffeatures, X, y, clf_type
        )

    elif xai_method == 'shap':
        shap_global(
            file_name, clf, X, ffeatures, clf_type
        )

    elif xai_method == 'ALE':
        print("ALE chưa được triển khai trong explain_predictions.")