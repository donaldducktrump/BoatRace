# create_model_catboost.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import optuna
import joblib
# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合
import os
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, load_before_data, merge_before_data, load_before1min_data, preprocess_data1min
from get_data import load_fan_data, add_course_data
import numpy as np

def remove_common_columns(df_left, df_right, on_columns):
    common_cols = set(df_left.columns).intersection(set(df_right.columns)) - set(on_columns)
    if common_cols:
        print(f"共通列: {common_cols}。df_rightからこれらの列を削除します。")
        df_right = df_right.drop(columns=common_cols)
    return df_right

def save_dataframe(df, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_pickle(file_path)

def load_processed_odds_data():
    file_path = 'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/odds_dataframe/odds_data.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    
def load_dataframe(file_path):
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        return None
    
def create_model():
    month_folders = ['2311', '2312', '2401', '2402', '2403', '2404']
    date_files = {
        '2311': [f'2311{day:02d}' for day in range(1, 31)],
        '2312': [f'2312{day:02d}' for day in range(1, 32)],
        '2401': [f'2401{day:02d}' for day in range(1, 31)],
        '2402': [f'2402{day:02d}' for day in range(1, 29)],
        '2403': [f'2403{day:02d}' for day in range(1, 32)],
        '2404': [f'2404{day:02d}' for day in range(1, 31)],
    }

    data_list = []
    for month in month_folders:
        for date in date_files[month]:
            try:
                b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'
                k_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_data/{month}/K{date}.TXT'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'

                b_data = load_b_file(b_file)
                k_data, _ = load_k_file(k_file)
                before_data = load_before_data(before_file)

                if not b_data.empty and not k_data.empty and not before_data.empty:
                    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
                    before_data = remove_common_columns(data, before_data, on_columns=['選手登番', 'レースID', '艇番'])
                    data = merge_before_data(data, before_data)
                    data_list.append(data)
                else:
                    print(f"データが不足しています: {date}")

            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file}, {k_file}, または {before_file}")

    if not data_list:
        print("データが正しく読み込めませんでした。")
        return
    
    data = pd.concat(data_list, ignore_index=True)
    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')

    data = preprocess_data(data)
    max_position = 6
    data['target'] = max_position - data['着'] + 1

    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h','性別','前期能力指数', '今期能力指数', '平均スタートタイミング', '勝率', '複勝率', '優勝回数', '優出回数']
    
    X = data[features]
    y = data['target']
    categorical_features = ['会場', 'weather', 'wind_d', 'ESC', '性別', '級別', '支部', '艇番']
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]
    X_processed = pd.concat([X_numeric, X_categorical], axis=1)

    data_sorted = data.sort_values('レースID')
    X_processed_sorted = X_processed.loc[data_sorted.index]
    y_sorted = data_sorted['target']
    
    group_sizes = data_sorted.groupby('レースID').size().tolist()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X_processed_sorted, y_sorted, groups=data_sorted['レースID']))

    X_train, X_test = X_processed_sorted.iloc[train_idx], X_processed_sorted.iloc[test_idx]
    y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]
    group_train = data_sorted['レースID'].iloc[train_idx].value_counts().sort_index().tolist()
    group_test = data_sorted['レースID'].iloc[test_idx].value_counts().sort_index().tolist()

    train_pool = Pool(X_train, y_train, group_id=data_sorted['レースID'].iloc[train_idx])
    test_pool = Pool(X_test, y_test, group_id=data_sorted['レースID'].iloc[test_idx])

    ranker = CatBoostRanker(
        iterations=10000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=100,
        eval_metric='NDCG',
        task_type='CPU',
        # loss_function='YetiRank'  # PairLogit や QuerySoftmaxも試すことができます
        # loss_function='PairLogit',  # Pairwise
        loss_function='QuerySoftMax'  # Query-based loss
    )

    ranker.fit(train_pool, eval_set=test_pool, early_stopping_rounds=10)

    y_pred_scores = ranker.predict(X_test)
    test_data = data_sorted.iloc[test_idx].copy()
    test_data['pred_score'] = y_pred_scores

    predictions = test_data.groupby('レースID').apply(lambda x: x.nlargest(6, 'pred_score')).reset_index(drop=True)
    actual_top1 = test_data[test_data['着'] == 1].drop_duplicates(subset=['レースID']).set_index('レースID')['艇番']
    predicted_top1 = predictions.groupby('レースID').apply(lambda x: x.iloc[0]['艇番'])

    duplicates = test_data[test_data['着'] == 1]['レースID'].duplicated().sum()
    if duplicates > 0:
        print(f"There are {duplicates} duplicate 'レースID's with '着' == 1. They will be dropped.")
        actual_top1 = actual_top1[~actual_top1.index.duplicated(keep='first')]

    common_race_ids = predicted_top1.index.intersection(actual_top1.index)
    print(f"Number of Common レースID for Top-1 Accuracy: {len(common_race_ids)}")

    predicted_common = predicted_top1.loc[common_race_ids]
    actual_common = actual_top1.loc[common_race_ids]

    correct_top1 = (predicted_common == actual_common).sum()
    total_races = len(common_race_ids)
    accuracy_top1 = correct_top1 / total_races if total_races > 0 else 0
    print(f'Accuracy Top-1 on test data: {accuracy_top1:.4f}')

    importances = ranker.get_feature_importance(data=train_pool)
    feature_names = X_processed.columns.tolist()
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_catboost.png')
    plt.show()

    gkf = GroupKFold(n_splits=5)
    cv_scores = []

    for fold, (train_idx_cv, valid_idx_cv) in enumerate(gkf.split(X_processed_sorted, y_sorted, groups=data_sorted['レースID'])):
        X_train_cv, X_valid_cv = X_processed_sorted.iloc[train_idx_cv], X_processed_sorted.iloc[valid_idx_cv]
        y_train_cv, y_valid_cv = y_sorted.iloc[train_idx_cv], y_sorted.iloc[valid_idx_cv]
        
        group_train_cv = data_sorted['レースID'].iloc[train_idx_cv].value_counts().sort_index().tolist()
        group_valid_cv = data_sorted['レースID'].iloc[valid_idx_cv].value_counts().sort_index().tolist()
        
        train_pool_cv = Pool(X_train_cv, y_train_cv, group_id=data_sorted['レースID'].iloc[train_idx_cv])
        valid_pool_cv = Pool(X_valid_cv, y_valid_cv, group_id=data_sorted['レースID'].iloc[valid_idx_cv])

        ranker_cv = CatBoostRanker(
            iterations=10000,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=100,
            eval_metric='NDCG',
            task_type='CPU',
            loss_function='YetiRank'  # PairLogit や QuerySoftmaxも試すことができます
            # loss_function='PairLogit',
            # loss_function='QuerySoftmax'
        )

        ranker_cv.fit(train_pool_cv, eval_set=valid_pool_cv, early_stopping_rounds=10)

        y_pred_cv = ranker_cv.predict(X_valid_cv)
        test_data_cv = data_sorted.iloc[valid_idx_cv].copy()
        test_data_cv['pred_score'] = y_pred_cv

        predictions_cv = test_data_cv.groupby('レースID').apply(lambda x: x.nlargest(6, 'pred_score')).reset_index(drop=True)
        actual_top1_cv = test_data_cv[test_data_cv['着'] == 1].drop_duplicates(subset=['レースID']).set_index('レースID')['艇番']
        predicted_top1_cv = predictions_cv.groupby('レースID').apply(lambda x: x.iloc[0]['艇番'])

        duplicates_cv = test_data_cv[test_data_cv['着'] == 1]['レースID'].duplicated().sum()
        if duplicates_cv > 0:
            print(f"Fold {fold+1}: There are {duplicates_cv} duplicate 'レースID's with '着' == 1. They will be dropped.")
            actual_top1_cv = actual_top1_cv[~actual_top1_cv.index.duplicated(keep='first')]

        common_race_ids_cv = predicted_top1_cv.index.intersection(actual_top1_cv.index)
        predicted_common_cv = predicted_top1_cv.loc[common_race_ids_cv]
        actual_common_cv = actual_top1_cv.loc[common_race_ids_cv]

        correct_cv = (predicted_common_cv == actual_common_cv).sum()
        total_races_cv = len(common_race_ids_cv)
        accuracy_cv = correct_cv / total_races_cv if total_races_cv > 0 else 0
        cv_scores.append(accuracy_cv)
        print(f'Fold {fold+1} Accuracy Top-1: {accuracy_cv:.4f}')

    print(f'\nMean Cross-Validated Accuracy Top-1: {np.mean(cv_scores):.4f}')

    ranker.save_model('boatrace_ranker_catboost.cbm')
    X_processed.columns.to_series().to_csv('feature_names_ranker_catboost.csv', index=False)


if __name__ == '__main__':
    create_model()
