# Import Libraries
import os
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

from utils.retrieval import retrieve_artefact, retrieve_vector, get_imp_features
from preprocessing.datasets_preprocessing import scale_update
from preprocessing.Indispensable_feats_gen import compute_features_importances
from modeling.Models_training import train_models
from evaluation.Consistency_measures import calculate_abic
from evaluation.Consistency_ratios import compute_reducts_core, ComputeRatio


out_dir = 'XAI_global_FeatsConsis'
logs_dir = 'uci_logs'
models_folder = os.path.join(logs_dir, 'models')
params_dir = 'optimal_params'
datasets_folder = logs_dir

datasets = [
    'Diabetic',
    'Ionosphere',
    'spect',
    'Kidney',
    'Credit',
    'Climate',
    'Spam'
]

os.makedirs(out_dir, exist_ok=True)
experiments_dir = os.path.join(out_dir, 'experimental_measurements')
os.makedirs(experiments_dir, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

info_df = retrieve_artefact(logs_dir, 'datasets_info', '.csv', ';')

# Huấn luyện đủ 4 mô hình. Các tham số tối ưu có sẵn sẽ được giữ nguyên;
# 6 file còn thiếu được xử lý bằng fallback trong Models_training.py.
train_models(
    models_folder=models_folder,
    datasets=datasets,
    info_df=info_df,
    params_dir=params_dir,
    datasets_folder=datasets_folder,
    out_dir=out_dir
)

# Các bảng tổng hợp tương ứng Table 4–7.
table4_rows, table5_rows, table6_rows, table7_rows = [], [], [], []

for ds in datasets:
    file_name = info_df.loc[
        info_df['Dataset_name'] == ds, 'files_names'
    ].values[0]
    target_name = info_df.loc[
        info_df['Dataset_name'] == ds, 'targets_names'
    ].values[0]

    df = retrieve_artefact(
        logs_dir,
        'preprocessed_not_discretized_dataset_%s' % file_name,
        '.csv',
        ','
    )

    df_discretized = pd.read_csv(
        os.path.join(logs_dir, 'discretized_dataset_%s.csv' % file_name),
        header=None,
        names=df.columns
    )

    model_logit = retrieve_artefact(
        models_folder, 'model_logit_%s' % file_name, '.pickle'
    )
    model_xgboost = retrieve_artefact(
        models_folder, 'model_xgboost_%s' % file_name, '.pickle'
    )
    model_rf = retrieve_artefact(
        models_folder, 'model_rf_%s' % file_name, '.pickle'
    )
    model_gbm = retrieve_artefact(
        models_folder, 'model_gbm_%s' % file_name, '.pickle'
    )

    y = df[target_name]
    X = df.drop([target_name], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ffeatures = X_train.columns.tolist()
    # Các thuộc tính đã được mã hóa số ở bước preprocessing.
    num_cols = ffeatures
    cat_cols = []

    compute_features_importances(
        ds, X_train, y_train, ffeatures, target_name,
        model_xgboost, model_logit, model_rf, model_gbm,
        df_discretized, cat_cols, num_cols
    )

    compute_reducts_core(file_name, ds)

    coreset = retrieve_vector(
        logs_dir,
        'core_%s_python.json' % file_name
    )
    if not coreset:
        print('WARNING: Không có core cho dataset %s; bỏ qua dataset này.' % ds)
        continue

    core_Feats_indices = coreset
    core_feats = [ffeatures[i] for i in coreset]

    all_reds = retrieve_vector(
        logs_dir,
        'reds_%s_python.json' % file_name
    )
    if not all_reds:
        print('WARNING: Không có reduct cho dataset %s; bỏ qua dataset này.' % ds)
        continue

    # Chọn reduct ngắn nhất; giữ đúng liên kết giữa reduct và độ dài.
    reduct_candidates = [
        list(red) for red in all_reds.values()
        if len(red) != len(ffeatures)
    ]
    if not reduct_candidates:
        print('WARNING: Không tìm thấy reduct hợp lệ cho %s.' % ds)
        continue

    selected_red = min(reduct_candidates, key=len)
    reduct_feats_indices = selected_red
    reduct_feats = [ffeatures[item] for item in selected_red]

    selected_by_xai_reduct = {}
    selected_by_xai_core = {}

    for cls_method in ['xgboost', 'logit', 'rf', 'gbm']:
        for xai_type in ['shap', 'perm']:
            key = '%s_%s_%s' % (ds, cls_method, xai_type)
            selected_by_xai_core[key] = get_imp_features(
                logs_dir, file_name, ffeatures, cls_method,
                len(core_Feats_indices), xai_type
            )

            selected_by_xai_reduct[key] = get_imp_features(
                logs_dir, file_name, ffeatures, cls_method,
                len(reduct_feats_indices), xai_type
            )

    scale_update(selected_by_xai_core)
    scale_update(selected_by_xai_reduct)

    sample_size = X_train.shape[0]

    reduct_ratio_dict = defaultdict(dict)
    core_ratio_dict = defaultdict(dict)
    AIC_reduct_dict = defaultdict(dict)
    AIC_core_dict = defaultdict(dict)
    BIC_reduct_dict = defaultdict(dict)
    BIC_core_dict = defaultdict(dict)
    interreduct_scores_sum = defaultdict(dict)
    reduct_intersection = defaultdict(dict)
    intercore_scores_sum = defaultdict(dict)
    core_intersection = defaultdict(dict)

    reduct_ratio_exp_dict = defaultdict(dict)
    core_ratio_exp_dict = defaultdict(dict)
    AIC_reduct_exp_dict = defaultdict(dict)
    AIC_core_exp_dict = defaultdict(dict)
    BIC_reduct_exp_dict = defaultdict(dict)
    BIC_core_exp_dict = defaultdict(dict)

    for key in selected_by_xai_reduct:
        reduct_ratio_dict[key], interreduct_scores_sum[key], reduct_intersection[key] = (
            ComputeRatio(selected_by_xai_reduct[key], reduct_feats, 'regular')
        )
        comp_intersect_reduct = len(reduct_feats_indices) - reduct_intersection[key]
        AIC_reduct_dict[key] = calculate_abic(
            sample_size, reduct_ratio_dict[key], comp_intersect_reduct, 'AIC'
        )
        BIC_reduct_dict[key] = calculate_abic(
            sample_size, reduct_ratio_dict[key], comp_intersect_reduct, 'BIC'
        )

        core_ratio_dict[key], intercore_scores_sum[key], core_intersection[key] = (
            ComputeRatio(selected_by_xai_core[key], core_feats, 'regular')
        )
        comp_intersect_core = len(core_feats) - core_intersection[key]
        AIC_core_dict[key] = calculate_abic(
            sample_size, core_ratio_dict[key], comp_intersect_core, 'AIC'
        )
        BIC_core_dict[key] = calculate_abic(
            sample_size, core_ratio_dict[key], comp_intersect_core, 'BIC'
        )

        reduct_ratio_exp_dict[key], _ = ComputeRatio(
            selected_by_xai_reduct[key], reduct_feats, 'experimental'
        )
        AIC_reduct_exp_dict[key] = calculate_abic(
            sample_size, reduct_ratio_exp_dict[key], comp_intersect_reduct, 'AIC'
        )
        BIC_reduct_exp_dict[key] = calculate_abic(
            sample_size, reduct_ratio_exp_dict[key], comp_intersect_reduct, 'BIC'
        )
        core_ratio_exp_dict[key], _ = ComputeRatio(
            selected_by_xai_core[key], core_feats, 'experimental'
        )
        if core_ratio_exp_dict[key] >= 1:
            print(
                'RATIO >= 1:',
                'Dataset =', ds,
                'Key =', key,
                'Ratio =', core_ratio_exp_dict[key],
                'Selected =', list(selected_by_xai_core[key].keys()),
                'Core =', core_feats
            )
        AIC_core_exp_dict[key] = calculate_abic(
            sample_size, core_ratio_exp_dict[key], comp_intersect_core, 'AIC'
        )
        BIC_core_exp_dict[key] = calculate_abic(
            sample_size, core_ratio_exp_dict[key], comp_intersect_core, 'BIC'
        )

    measurements = [
        'reduct_ratio', 'core_ratio', 'AIC_reduct', 'AIC_core',
        'BIC_reduct', 'BIC_core', 'intersect_reduct_scores_sum',
        '#feats_reduct_intersection', 'intersect_core_scores_sum',
        '#feats_core_intersection'
    ]
    results_df = pd.DataFrame.from_dict([
        reduct_ratio_dict, core_ratio_dict, AIC_reduct_dict, AIC_core_dict,
        BIC_reduct_dict, BIC_core_dict, interreduct_scores_sum,
        reduct_intersection, intercore_scores_sum, core_intersection
    ])
    results_df.insert(0, 'Measurement', measurements)
    results_df.to_csv(
        os.path.join(logs_dir, 'measurements_%s.csv' % file_name),
        sep=';', index=False
    )

    # Giữ lại danh sách feature được chọn để tiện kiểm tra.
    selected_feats_df = pd.concat([
        pd.DataFrame.from_dict([selected_by_xai_reduct]),
        pd.DataFrame.from_dict([selected_by_xai_core])
    ], ignore_index=True)
    selected_feats_df.insert(
        0, 'Selection_type',
        ['selected_by_xai_reduct', 'selected_by_xai_core']
    )
    selected_feats_df.to_csv(
        os.path.join(logs_dir, 'measurements_%s.csv' % file_name),
        sep=';', mode='a', index=False
    )

    # Dữ liệu tổng hợp Table 4–7: các giá trị dựa trên reduct.
    row4 = {
        'Dataset': ds,
        '#Features at Dataset Reduct': len(reduct_feats_indices)
    }
    row5 = {'Dataset': ds}
    row6 = {'Dataset': ds}
    row7 = {'Dataset': ds}

    for model in ['xgboost', 'logit', 'rf', 'gbm']:
        for method in ['shap', 'perm']:
            key = '%s_%s_%s' % (ds, model, method)
            col = '%s_%s' % (model, method)
            row4[col] = reduct_intersection[key]
            row5[col] = reduct_ratio_dict[key]
            row6[col] = AIC_reduct_dict[key]
            row7[col] = BIC_reduct_dict[key]

    table4_rows.append(row4)
    table5_rows.append(row5)
    table6_rows.append(row6)
    table7_rows.append(row7)

    # Lưu thêm file experimental như code gốc.
    exp_df = pd.DataFrame.from_dict([
        reduct_ratio_exp_dict, core_ratio_exp_dict,
        AIC_reduct_exp_dict, AIC_core_exp_dict,
        BIC_reduct_exp_dict, BIC_core_exp_dict
    ])
    exp_df.insert(
        0, 'Measurement',
        ['reduct_ratio', 'core_ratio', 'AIC_reduct',
         'AIC_core', 'BIC_reduct', 'BIC_core']
    )
    exp_df.to_csv(
        os.path.join(experiments_dir, 'measurements_experimental_%s.csv' % file_name),
        sep=';', index=False
    )

# Xuất bảng tổng hợp.
pd.DataFrame(table4_rows).to_csv(
    os.path.join(out_dir, 'Table4_intersection_sets.csv'),
    index=False, encoding='utf-8-sig'
)
pd.DataFrame(table5_rows).to_csv(
    os.path.join(out_dir, 'Table5_reduct_ratios.csv'),
    index=False, encoding='utf-8-sig'
)
pd.DataFrame(table6_rows).to_csv(
    os.path.join(out_dir, 'Table6_AIC.csv'),
    index=False, encoding='utf-8-sig'
)
pd.DataFrame(table7_rows).to_csv(
    os.path.join(out_dir, 'Table7_BIC.csv'),
    index=False, encoding='utf-8-sig'
)

print('Đã hoàn tất. Kiểm tra các file Table4–Table7 trong:', out_dir)
print('Table3_model_performance.csv được tạo trong thư mục:', out_dir)
