# create_model.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import optuna
import joblib
# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合
import os
import pandas as pd
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_optuna  # Optuna のインテグレーション
from optuna.samplers import TPESampler
from lightgbm import LGBMRanker
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, load_before_data, merge_before_data, load_before1min_data, preprocess_data1min
from get_data import load_fan_data, add_course_data
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV


def remove_common_columns(df_left, df_right, on_columns):
    """
    df_leftとdf_rightのマージ時に、on_columnsを除く共通の列をdf_rightから削除する。
    """
    common_cols = set(df_left.columns).intersection(set(df_right.columns)) - set(on_columns)
    if common_cols:
        print(f"共通列: {common_cols}。df_rightからこれらの列を削除します。")
        df_right = df_right.drop(columns=common_cols)
    return df_right

def save_dataframe(df, file_path):
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # データフレームをピクル形式で保存
    df.to_pickle(file_path)

def load_processed_odds_data():
    # Load the processed dataframe directly from saved CSV file
    file_path = 'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/odds_dataframe/odds_data.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    
def load_dataframe(file_path):
    # ファイルが存在する場合は読み込み
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        return None
    
def create_model():
    # 2403/240301～2407/240731 のデータを使用してモデルを学習
    # トレーニング期間のデータ日付リスト作成
    # month_folders = ['2210', '2211', '2212', '2301', '2302', '2303', '2304', '2305', '2306', '2307', '2308', '2309', '2310', '2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407' ]
    # month_folders = ['2310', '2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407']   
    month_folders = ['2311', '2312', '2401', '2402', '2403', '2404']
    # month_folders = ['2408', '2409', '2410']    

    date_files = {
        # '2210': [f'2210{day:02d}' for day in range(1, 32)],  # 221001 - 221031
        # '2211': [f'2211{day:02d}' for day in range(1, 31)],  # 221101 - 221130
        # '2212': [f'2212{day:02d}' for day in range(1, 32)],  # 221201 - 221231
        # '2301': [f'2301{day:02d}' for day in range(1, 31)],  # 230101 - 230131
        # '2302': [f'2302{day:02d}' for day in range(1, 29)],  # 230201 - 230228
        # '2303': [f'2303{day:02d}' for day in range(1, 32)],  # 230301 - 230331
        # '2304': [f'2304{day:02d}' for day in range(1, 31)],  # 230401 - 230430
        # '2305': [f'2305{day:02d}' for day in range(1, 32)],  # 230501 - 230531
        # '2306': [f'2306{day:02d}' for day in range(1, 31)],  # 230601 - 230630
        # '2307': [f'2307{day:02d}' for day in range(1, 32)],  # 230701 - 230731
        # '2308': [f'2308{day:02d}' for day in range(1, 32)],  # 230801 - 230831
        # '2309': [f'2309{day:02d}' for day in range(1, 31)],  # 230901 - 230930
        # '2310': [f'2310{day:02d}' for day in range(1, 32)],  # 231001 - 231031
        '2311': [f'2311{day:02d}' for day in range(1, 31)],  # 231101 - 231130
        '2312': [f'2312{day:02d}' for day in range(1, 32)],  # 231201 - 231231
        '2401': [f'2401{day:02d}' for day in range(1, 31)],  # 240101 - 240131
        '2402': [f'2402{day:02d}' for day in range(1, 29)],  # 240201 - 240228
        '2403': [f'2403{day:02d}' for day in range(1, 32)],  # 240301 - 240331
        '2404': [f'2404{day:02d}' for day in range(1, 31)],  # 240401 - 240430
        # '2405': [f'2405{day:02d}' for day in range(1, 32)],  # 240501 - 240531
        # '2406': [f'2406{day:02d}' for day in range(1, 31)],  # 240601 - 240630
        # '2407': [f'2407{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        # '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        # '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        # '2410': [f'2410{day:02d}' for day in range(1, 16)],  # 241001 - 241014
    }

    # データを結合
    data_list = []
    b_data_list = []
    k_data_list = []
    before_data_list = []

    for month in month_folders:
        for date in date_files[month]:
            try:
                b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'
                k_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_data/{month}/K{date}.TXT'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'
                # before1min_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data_1min/{month}/beforeinfo_{date}.txt'

                b_data = load_b_file(b_file)
                k_data, _ = load_k_file(k_file)
                before_data = load_before_data(before_file)
                # before1min_data = load_before1min_data(before1min_file)
                # print(f"before_data: {before_data}")

                if not b_data.empty and not k_data.empty and not before_data.empty:
                    # データのマージ
                    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
                    before_data = remove_common_columns(data, before_data, on_columns=['選手登番', 'レースID', '艇番'])
                    data = merge_before_data(data, before_data)
                    # print(data)
                    # before1min_data = remove_common_columns(data, before1min_data, on_columns=['選手登番', 'レースID', '艇番'])
                    # data = merge_before_data(data, before1min_data)
                    data_list.append(data)
                else:
                    print(f"データが不足しています: {date}")

            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file}, {k_file}, または {before_file}")

    if not data_list:
        print("データが正しく読み込めませんでした。")
        return
    
    data_odds = load_processed_odds_data()

    # 全データを結合
    data = pd.concat(data_list, ignore_index=True)

    # 前処理

    print("data columns:", data.columns.tolist())
    print(data)

    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    # 必要な列を指定してマージ
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')

    # 各行に対してコース別データを追加
    # data = data.apply(lambda row: add_course_data(row, df_fan), axis=1)
    data = preprocess_data(data)
    # 結果を表示
    print("data columns:", data.columns.tolist())
    print(data.head())

    # 目的変数の作成([0,1],[2,3],[4,5]に分類)
    # def target_label(x):
    #     x = int(x)
    #     if x in [1, 2]:
    #         return 0
    #     elif x in [3, 4]:
    #         return 1
    #     else:
    #         return 2
        
    # def target_label(x):
    #     x = int(x)
    #     if x in [0, 1, 2]:
    #         return 0
    #     else:
    #         return 1

    def target_label(x):
        x = int(x)
        if x == 1 :
            return 0
        else:
            return 1

    #1,2,3着,それ以外に分類
    # def target_label(x):
    #     x = int(x)
    #     if x == 1:
    #         return 0
    #     elif x == 2:
    #         return 1
    #     elif x == 3:
    #         return 2
    #     else:
    #         return 3

    #trio
    # def target_label(x):
    #     x = int(x)
    #     if x in [1,2,3]:
    #         return 0
    #     else:
    #         return 1


    # 特徴量と目的変数
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h','性別','前期能力指数', '今期能力指数', '平均スタートタイミング', '勝率', '複勝率', '優勝回数', '優出回数']
    
    # Calculate the maximum position (e.g., 3 for 1st, 2nd, 3rd)
    max_position = 6

    # Define target: higher values for better ranks
    data['target'] = max_position - data['着'] + 1

    X = data[features]
    y = data['target']

    # Categorical features
    categorical_features = ['会場', 'weather', 'wind_d', 'ESC', '性別', '級別', '支部', '艇番']

    # One-Hot Encoding
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # Numerical features
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]

    # Combine numerical and categorical features
    X_processed = pd.concat([X_numeric, X_categorical], axis=1)

    print("X_processed columns:", X_processed.columns.tolist())
    print(X_processed.head())

    # Sort data by 'レースID' to ensure group consistency
    data_sorted = data.sort_values('レースID')

    # Align X_processed_sorted with data_sorted
    X_processed_sorted = X_processed.loc[data_sorted.index]

    # Extract y_sorted directly from data_sorted
    y_sorted = data_sorted['target']
    
    # Verify alignment
    assert len(data_sorted) == len(X_processed_sorted) == len(y_sorted), "Mismatch in DataFrame lengths."
    assert all(data_sorted.index == X_processed_sorted.index), "Indices of data_sorted and X_processed_sorted do not match."
    assert all(data_sorted.index == y_sorted.index), "Indices of data_sorted and y_sorted do not match."

    # Group sizes: number of boats per race
    group_sizes = data_sorted.groupby('レースID').size().tolist()

    # Initialize GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

    # Generate train and test indices
    train_idx, test_idx = next(gss.split(X_processed_sorted, y_sorted, groups=data_sorted['レースID']))

    # Split the data
    X_train, X_test = X_processed_sorted.iloc[train_idx], X_processed_sorted.iloc[test_idx]
    y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]

    # Corresponding group sizes
    group_train = data_sorted['レースID'].iloc[train_idx].value_counts().sort_index().tolist()
    group_test = data_sorted['レースID'].iloc[test_idx].value_counts().sort_index().tolist()

    # Initialize LGBMRanker
    ranker = LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=10000,
        random_state=42,
        verbose=-1
    )

    # Fit the model with group information
    ranker.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_test, y_test)],
        eval_group=[group_test],
        eval_at=[1, 3],  # Evaluate NDCG at 1 and 3
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True),
                    lgb.log_evaluation(period=10)]
    )

    # Predict scores for test data
    y_pred_scores = ranker.predict(X_test)

    # Attach predictions to the test set
    test_data = data_sorted.iloc[test_idx].copy()
    test_data['pred_score'] = y_pred_scores


    # Determine Top-1 Predictions
    predictions = test_data.groupby('レースID').apply(lambda x: x.nlargest(6, 'pred_score')).reset_index(drop=True)

    # Extract Actual Top-1 Boats
    # Ensure there's exactly one '着' == 1 per 'レースID'
    actual_top1 = test_data[test_data['着'] == 1].drop_duplicates(subset=['レースID']).set_index('レースID')['艇番']

    # Extract Predicted Top-1 Boats
    predicted_top1 = predictions.groupby('レースID').apply(lambda x: x.iloc[0]['艇番'])

    # Handle Duplicate 'レースID's in Actual Top-1
    duplicates = test_data[test_data['着'] == 1]['レースID'].duplicated().sum()
    if duplicates > 0:
        print(f"There are {duplicates} duplicate 'レースID's with '着' == 1. They will be dropped.")
        actual_top1 = actual_top1[~actual_top1.index.duplicated(keep='first')]

    # Identify Common 'レースID's
    common_race_ids = predicted_top1.index.intersection(actual_top1.index)
    print(f"Number of Common レースID for Top-1 Accuracy: {len(common_race_ids)}")

    # Align the Series based on common 'レースID's
    predicted_common = predicted_top1.loc[common_race_ids]
    actual_common = actual_top1.loc[common_race_ids]

    # Check for duplicates after alignment
    duplicates_predicted = predicted_common.index.duplicated().sum()
    if duplicates_predicted > 0:
        print(f"There are {duplicates_predicted} duplicate 'レースID's in predicted_top1. They will be dropped.")
        predicted_common = predicted_common[~predicted_common.index.duplicated(keep='first')]

    duplicates_actual = actual_common.index.duplicated().sum()
    if duplicates_actual > 0:
        print(f"There are {duplicates_actual} duplicate 'レースID's in actual_top1. They will be dropped.")
        actual_common = actual_common[~actual_common.index.duplicated(keep='first')]

    # Re-identify common race IDs after dropping duplicates
    common_race_ids = predicted_common.index.intersection(actual_common.index)
    print(f"Number of Common レースID after dropping duplicates: {len(common_race_ids)}")

    predicted_common = predicted_common.loc[common_race_ids]
    actual_common = actual_common.loc[common_race_ids]

    # Calculate Top-1 Accuracy
    correct_top1 = (predicted_common == actual_common).sum()
    total_races = len(common_race_ids)
    accuracy_top1 = correct_top1 / total_races if total_races > 0 else 0
    print(f'Accuracy Top-1 on test data: {accuracy_top1:.4f}')


    # Get feature importances

    importances = ranker.feature_importances_
    feature_names = X_processed.columns.tolist()
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_ranker.png')
    plt.show()

    print("\nCross-validation with early stopping:")

    gkf = GroupKFold(n_splits=5)
    cv_scores = []

    for fold, (train_idx_cv, valid_idx_cv) in enumerate(gkf.split(X_processed_sorted, y_sorted, groups=data_sorted['レースID'])):
        X_train_cv, X_valid_cv = X_processed_sorted.iloc[train_idx_cv], X_processed_sorted.iloc[valid_idx_cv]
        y_train_cv, y_valid_cv = y_sorted.iloc[train_idx_cv], y_sorted.iloc[valid_idx_cv]
        
        # Group sizes
        group_train_cv = data_sorted['レースID'].iloc[train_idx_cv].value_counts().sort_index().tolist()
        group_valid_cv = data_sorted['レースID'].iloc[valid_idx_cv].value_counts().sort_index().tolist()
        
        # Initialize a new Ranker for each fold
        ranker_cv = LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=10000,
            random_state=42,
            verbose=-1
        )
        
        # Fit
        ranker_cv.fit(
            X_train_cv,
            y_train_cv,
            group=group_train_cv,
            eval_set=[(X_valid_cv, y_valid_cv)],
            eval_group=[group_valid_cv],
            eval_at=[1, 3],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True),
                        lgb.log_evaluation(period=10)]
        )
        
        # Predict
        y_pred_cv = ranker_cv.predict(X_valid_cv)
        test_data_cv = data_sorted.iloc[valid_idx_cv].copy()
        test_data_cv['pred_score'] = y_pred_cv
        

        # Determine Top-1 Predictions
        predictions_cv = test_data_cv.groupby('レースID').apply(lambda x: x.nlargest(6, 'pred_score')).reset_index(drop=True)

        # Extract Actual Top-1 Boats
        actual_top1_cv = test_data_cv[test_data_cv['着'] == 1].drop_duplicates(subset=['レースID']).set_index('レースID')['艇番']

        # Extract Predicted Top-1 Boats
        predicted_top1_cv = predictions_cv.groupby('レースID').apply(lambda x: x.iloc[0]['艇番'])

        # Handle Duplicate 'レースID's in Actual Top-1
        duplicates_cv = test_data_cv[test_data_cv['着'] == 1]['レースID'].duplicated().sum()
        if duplicates_cv > 0:
            print(f"Fold {fold+1}: There are {duplicates_cv} duplicate 'レースID's with '着' == 1. They will be dropped.")
            actual_top1_cv = actual_top1_cv[~actual_top1_cv.index.duplicated(keep='first')]

        # Identify Common 'レースID's
        common_race_ids_cv = predicted_top1_cv.index.intersection(actual_top1_cv.index)
        print(f'Fold {fold+1} - Number of Common レースID for Top-1 Accuracy: {len(common_race_ids_cv)}')

        # Align the Series based on common 'レースID's
        predicted_common_cv = predicted_top1_cv.loc[common_race_ids_cv]
        actual_common_cv = actual_top1_cv.loc[common_race_ids_cv]

        # Check for duplicates after alignment
        duplicates_predicted_cv = predicted_common_cv.index.duplicated().sum()
        if duplicates_predicted_cv > 0:
            print(f"Fold {fold+1}: There are {duplicates_predicted_cv} duplicate 'レースID's in predicted_top1_cv. They will be dropped.")
            predicted_common_cv = predicted_common_cv[~predicted_common_cv.index.duplicated(keep='first')]

        duplicates_actual_cv = actual_common_cv.index.duplicated().sum()
        if duplicates_actual_cv > 0:
            print(f"Fold {fold+1}: There are {duplicates_actual_cv} duplicate 'レースID's in actual_top1_cv. They will be dropped.")
            actual_common_cv = actual_common_cv[~actual_common_cv.index.duplicated(keep='first')]

        # Re-identify common race IDs after dropping duplicates
        common_race_ids_cv = predicted_common_cv.index.intersection(actual_common_cv.index)
        print(f"Fold {fold+1} - Number of Common レースID after dropping duplicates: {len(common_race_ids_cv)}")

        predicted_common_cv = predicted_common_cv.loc[common_race_ids_cv]
        actual_common_cv = actual_common_cv.loc[common_race_ids_cv]

        # Calculate Top-1 Accuracy
        correct_cv = (predicted_common_cv == actual_common_cv).sum()
        total_races_cv = len(common_race_ids_cv)
        accuracy_cv = correct_cv / total_races_cv if total_races_cv > 0 else 0
        cv_scores.append(accuracy_cv)
        print(f'Fold {fold+1} Accuracy Top-1: {accuracy_cv:.4f}')

    print(f'\nMean Cross-Validated Accuracy Top-1: {np.mean(cv_scores):.4f}')

    # Save the model
    ranker.booster_.save_model('boatrace_ranker_model.txt')

    # Save feature names
    X_processed.columns.to_series().to_csv('feature_names_ranker.csv', index=False)


if __name__ == '__main__':
    create_model()
