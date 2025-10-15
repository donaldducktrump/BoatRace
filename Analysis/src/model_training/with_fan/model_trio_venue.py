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
    month_folders = ['2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410']

    date_files = {
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
    }

    # データを結合
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
                    # データのマージ
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
    
    data_odds = load_processed_odds_data()

    # 全データを結合
    data = pd.concat(data_list, ignore_index=True)

    # 前処理
    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    # 必要な列を指定してマージ
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')

    # データの前処理
    data = preprocess_data(data)

    # 目的変数の作成
    def target_label(x):
        x = int(x)
        if x in [1, 2, 3]:
            return 0
        else:
            return 1

    data['target'] = data['着'].apply(target_label)

    # 会場ごとにモデルをトレーニング
    venues = data['会場'].unique()
    print(f"トレーニング対象の会場数: {len(venues)}")

    # グローバルな特徴量名を保存または読み込み
    feature_names_file = 'feature_names_trio.csv'
    global_feature_names = None

    for venue in venues:
        print(f"\n会場: {venue} のモデルをトレーニング中...")
        venue_data = data[data['会場'] == venue]

        if venue_data.empty:
            print(f"会場 {venue} のデータが存在しません。スキップします。")
            continue

        # 特徴量と目的変数
        features = ['艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                    '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                    'ET', 'tilt', 'EST', 'ESC',
                    'weather', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                    '性別', '前期能力指数', '今期能力指数', '平均スタートタイミング', '勝率', '複勝率', '優勝回数', '優出回数']

        X_venue = venue_data[features]
        y_venue = venue_data['target']

        # カテゴリカル変数のOne-Hotエンコーディング
        categorical_features = ['weather', 'wind_d', 'ESC', '性別', '級別', '支部', '艇番']
        X_categorical = pd.get_dummies(X_venue[categorical_features], drop_first=True)

        # 数値データの選択
        numeric_features = [col for col in X_venue.columns if col not in categorical_features]
        X_numeric = X_venue[numeric_features]

        # 特徴量の結合
        X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

        # グローバルな特徴量名のリストを作成または読み込み
        if global_feature_names is None:
            if os.path.exists(feature_names_file):
                global_feature_names = pd.read_csv(feature_names_file).squeeze().tolist()
            else:
                global_feature_names = X_processed.columns.tolist()
                pd.Series(global_feature_names).to_csv(feature_names_file, index=False)

        # 欠けている列を補完（全て0で埋める）
        for feature in global_feature_names:
            if feature not in X_processed.columns:
                X_processed[feature] = 0

        # 列の順序を揃える
        X_processed = X_processed[global_feature_names]

        # データの分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_venue, test_size=0.25, random_state=42)

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
        print(f'Accuracy on test data for venue {venue}: {accuracy:.4f}')

        # 特徴量の重要度を取得してプロット
        importance = gbm.feature_importance(importance_type='gain')
        feature_name = gbm.feature_name()
        importance_df = pd.DataFrame({'feature': feature_name, 'importance': importance})
        importance_df = importance_df.sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance for Venue {venue}')
        plt.gca().invert_yaxis()
        # ディレクトリを作成
        os.makedirs('feature_importance', exist_ok=True)
        plt.savefig(f'feature_importance/feature_importance_{venue}.png')
        plt.close()

        # モデルの保存
        os.makedirs('models', exist_ok=True)
        model_filename = f'models/boatrace_model_trio_{venue}.txt'
        gbm.save_model(model_filename)
        print(f"モデルを保存しました: {model_filename}")

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

        print(f'\nMean Accuracy for venue {venue}: {np.mean(scores):.4f}')

if __name__ == '__main__':
    create_model()
