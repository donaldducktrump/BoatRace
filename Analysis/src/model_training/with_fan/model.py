# create_model.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import optuna

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合
import os
import pandas as pd
import optuna.integration.lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, load_before_data, merge_before_data, load_before1min_data, preprocess_data1min
from get_data import load_fan_data, add_course_data
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
    month_folders = ['2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410', '2411']
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
        '2405': [f'2405{day:02d}' for day in range(1, 32)],  # 240501 - 240531
        '2406': [f'2406{day:02d}' for day in range(1, 31)],  # 240601 - 240630
        '2407': [f'2407{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(1, 32)],  # 241001 - 241014
        '2411': [f'2411{day:02d}' for day in range(1, 5)],  # 241101 - 241130
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

    data['target'] = data['着'].apply(target_label)

    # 特徴量と目的変数
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h','性別','前期能力指数', '今期能力指数', '平均スタートタイミング', '勝率', '複勝率', '優勝回数', '優出回数']
    # '天候', '風向', '風量', '波'
    X = data[features]
    y = data['target']

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather','wind_d', 'ESC','性別','級別','支部','艇番']
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # 数値データの選択
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]

    # 特徴量の結合
    X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

    print("X_processed columns:", X_processed.columns.tolist())
    print(X_processed)
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.25, random_state=42)

    # データをDatasetクラスに格納
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test)

    # 使用するパラメータ
    params = {'objective': 'multiclass',
              'num_class': 2,
              'metric': 'multi_logloss',
              'random_state': 42,
              'boosting_type': 'gbdt',
              'verbose': -1}

    # params = {'objective': 'binary',
    #             'metric': 'binary_logloss',
    #             'random_state': 42,
    #             'boosting_type': 'gbdt',
    #             'verbose': -1}

    verbose_eval = 10  # 学習時のスコア推移を10ラウンドごとに表示

    # early_stoppingを指定してLightGBM学習
    gbm = lgb.train(params,
                    dtrain,
                    valid_sets=[dvalid],
                    num_boost_round=10000,
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True),
                               lgb.log_evaluation(verbose_eval)]
                    )

    # テストデータで予測
    y_pred = gbm.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # 精度の計算
    accuracy = accuracy_score(y_test, y_pred_labels)
    print(f'Accuracy on test data: {accuracy:.4f}')

    # 特徴量の重要度を取得してプロット
    importance = gbm.feature_importance(importance_type='gain')
    feature_name = gbm.feature_name()
    importance_df = pd.DataFrame({'feature': feature_name, 'importance': importance})
    importance_df = importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig('feature_importance.png')
    plt.show()

    # モデルの保存
    gbm.save_model('boatrace_model_first.txt')

    # 特徴量名の保存
    X_processed.columns.to_series().to_csv('feature_names_first.csv', index=False)


    # クロスバリデーションの実装
    print("\nCross-validation with early stopping:")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train)):
        X_train_cv, X_valid_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_train_cv, y_valid_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]

        dtrain_cv = lgb.Dataset(X_train_cv, label=y_train_cv)
        dvalid_cv = lgb.Dataset(X_valid_cv, label=y_valid_cv)

        gbm_cv = lgb.train(params,
                           dtrain_cv,
                           valid_sets=[dvalid_cv],
                           num_boost_round=10000,
                           callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                           )

        y_pred_cv = gbm_cv.predict(X_valid_cv)
        y_pred_labels_cv = np.argmax(y_pred_cv, axis=1)
        accuracy_cv = accuracy_score(y_valid_cv, y_pred_labels_cv)
        scores.append(accuracy_cv)
        print(f'Fold {i+1} Accuracy: {accuracy_cv:.4f}')

    print(f'\nMean Accuracy: {np.mean(scores):.4f}')


if __name__ == '__main__':
    create_model()
