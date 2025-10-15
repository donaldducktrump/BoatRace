# create_model.py
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import optuna

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合
import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from ...utils.mymodule import (
    load_b_file,
    load_k_file,
    preprocess_data,
    load_before_data,
    merge_before_data,
    load_before1min_data,
    preprocess_data1min
)
from get_data import load_fan_data, add_course_data
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
import re  # 正規表現を使用するために追加
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

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

def train_and_save_model(date_str, cumulative_data, odds_results, feature_names, model_base_dir):
    try:
        # モデル作成対象の日付範囲 (240901～241105)
        target_start_date = datetime.strptime('240901', '%y%m%d')
        target_end_date = datetime.strptime('241105', '%y%m%d')
        current_date_dt = datetime.strptime(date_str, '%y%m%d')

        if not (target_start_date <= current_date_dt <= target_end_date):
            return  # モデル作成対象外の日付

        # 現在の日付までのデータでモデルを訓練
        if cumulative_data.empty:
            print(f"{date_str} の累積データが空です。モデルをスキップします。")
            return

        # '三連単結果' を odds_results からマージ
        data_with_result = pd.merge(
            cumulative_data, 
            odds_results[['レースID', '三連単結果']], 
            on='レースID', 
            how='left'
        )

        # ターゲット変数の定義
        def target_label(row):
            trifecta_result = row['三連単結果']
            if pd.isna(trifecta_result):
                return 0
            else:
                try:
                    trifecta_boats = [int(x) for x in trifecta_result.split('-')]
                    return 1 if row['艇番'] in trifecta_boats else 0
                except:
                    return 0

        data_with_result['target'] = data_with_result.apply(target_label, axis=1)

        # ターゲット変数がすべて0の場合、モデルを訓練できないのでスキップ
        if data_with_result['target'].sum() == 0:
            print(f"{date_str} のターゲット変数がすべて0です。モデルをスキップします。")
            return

        # 特徴量の指定
        features = [
            '会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
            '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
            'ET', 'tilt', 'EST','ESC',
            'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
            '性別','前期能力指数', '今期能力指数', '平均スタートタイミング',
            '勝率', '複勝率', '優勝回数', '優出回数'
        ]

        X = data_with_result[features]
        y = data_with_result['target']

        # カテゴリカル変数のOne-Hotエンコーディング
        categorical_features = ['会場', 'weather','wind_d', 'ESC','性別','級別','支部','艇番']
        X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

        # 数値データの選択
        numeric_features = [col for col in X.columns if col not in categorical_features]
        X_numeric = X[numeric_features]

        # 特徴量の結合
        X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

        # 特徴量の整列
        if feature_names:
            # 存在しない特徴量を0で埋める
            for col in feature_names:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            # 不要な特徴量を削除
            X_processed = X_processed[feature_names]
        else:
            # 特徴量名を保存
            feature_names = X_processed.columns.tolist()

        # データの分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.25, random_state=42
        )

        # データをDatasetクラスに格納
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_test, label=y_test)

        params = {
            'objective': 'multiclass',
            'num_class': 2,
            'metric': 'multi_logloss',
            'random_state': 42,
            'boosting_type': 'gbdt',
            'verbose': -1
        }

        verbose_eval = 10  # 学習時のスコア推移を10ラウンドごとに表示

        # early_stoppingを指定してLightGBM学習
        gbm = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, verbose=True),
                lgb.log_evaluation(verbose_eval)
            ]
        )

        # テストデータで予測
        y_pred = gbm.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)

        # 精度の計算
        accuracy = accuracy_score(y_test, y_pred_labels)
        print(f'{date_str} のテストデータ精度: {accuracy:.4f}')

        # 特徴量の重要度を取得してプロット
        importance = gbm.feature_importance(importance_type='gain')
        feature_name = gbm.feature_name()
        importance_df = pd.DataFrame({'feature': feature_name, 'importance': importance})
        importance_df = importance_df.sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance on {date_str}')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # モデルとプロットの保存
        model_folder = os.path.join(model_base_dir, date_str[:4])  # 'YYMM'に基づくフォルダ
        os.makedirs(model_folder, exist_ok=True)
        model_path = os.path.join(model_folder, f'model_first_test_{date_str}.txt')
        gbm.save_model(model_path)
        plot_path = os.path.join(model_folder, f'feature_importance_{date_str}.png')
        plt.savefig(plot_path)
        plt.close()

        print(f'Model saved to {model_path}')
        print(f'Feature importance plot saved to {plot_path}')

    except Exception as e:
        logging.error(f"{date_str} のモデル作成中にエラーが発生しました: {e}")

def create_model():
    # モデルを保存するベースディレクトリ
    model_base_dir = 'model_by_date'

    # 学習期間の開始日と終了日
    start_date = datetime.strptime('231101', '%y%m%d')
    end_date = datetime.strptime('241105', '%y%m%d')

    # 日付リストの作成 (231101から241105まで)
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%y%m%d')
        date_list.append(date_str)
        current_date += timedelta(days=1)

    # 累積データの初期化
    cumulative_data = pd.DataFrame()

    # 累積された三連単結果のデータフレーム
    odds_results = pd.DataFrame(columns=['レースID', '三連単結果'])

    # ファンデータの読み込み
    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    # odds_dataの読み込み
    odds_data = load_processed_odds_data()
    if odds_data.empty:
        print("odds_data が空です。")
        return

    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        filename='create_model.log',
        filemode='a',
        format='%(asctime)s %(levelname)s:%(message)s'
    )

    # 特徴量名の読み込みまたは初期化
    feature_names_path = 'feature_names_first_test.csv'
    if os.path.exists(feature_names_path):
        feature_names = pd.read_csv(feature_names_path).squeeze().tolist()
    else:
        feature_names = None  # 初回はNoneとしてモデル関数内で保存

    # ThreadPoolExecutorの初期化
    max_workers = 20  # 必要に応じて調整
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = []

    for date_str in tqdm(date_list, desc="Processing Dates"):
        month_str = date_str[:4]  # 'YYMM'
        # ファイルパスの定義
        b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month_str}/B{date_str}.TXT'
        k_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_data/{month_str}/K{date_str}.TXT'
        before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month_str}/beforeinfo_{date_str}.txt'

        # データの読み込み
        try:
            b_data = load_b_file(b_file)
            k_data, odds_list_part = load_k_file(k_file)
            before_data = load_before_data(before_file)

            if not b_data.empty and not before_data.empty:
                # データのマージ
                data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'], how='left')
                before_data = remove_common_columns(data, before_data, on_columns=['選手登番', 'レースID', '艇番'])
                data = merge_before_data(data, before_data)

                # ファンデータのマージ
                data = pd.merge(
                    data, 
                    df_fan[['選手登番'] + ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別',
                                          '勝率', '複勝率', '優勝回数', '優出回数']], 
                    on='選手登番', 
                    how='left'
                )

                # データ前処理
                data = preprocess_data(data)

                # 累積データに追加
                cumulative_data = pd.concat([cumulative_data, data], ignore_index=True)

                # odds_list_part を odds_results に追加（独立して扱う）
                if not odds_list_part.empty:
                    odds_results = pd.concat([odds_results, odds_list_part[['レースID', '三連単結果']]], ignore_index=True)
            else:
                print(f"データが不足しています: {date_str}")
                continue

        except FileNotFoundError as e:
            print(f"ファイルが見つかりません: {e}")
            continue
        except Exception as e:
            logging.error(f"データ処理中にエラーが発生しました: {e}")
            continue

        # モデル作成対象の日付範囲 (240901～241105)
        target_start_date = datetime.strptime('240901', '%y%m%d')
        target_end_date = datetime.strptime('241105', '%y%m%d')
        current_date_dt = datetime.strptime(date_str, '%y%m%d')

        if target_start_date <= current_date_dt <= target_end_date:
            # データをコピーしてスレッドに渡す
            cumulative_data_copy = copy.deepcopy(cumulative_data)
            odds_results_copy = copy.deepcopy(odds_results)
            futures.append(
                executor.submit(
                    train_and_save_model, 
                    date_str, 
                    cumulative_data_copy, 
                    odds_results_copy, 
                    feature_names, 
                    model_base_dir
                )
            )

    # 全てのタスクの完了を待つ
    for future in as_completed(futures):
        pass  # エラーはtrain_and_save_model内でログに記録される

    # Executorのシャットダウン
    executor.shutdown()

    # 特徴量名を保存（初回のみ）
    if feature_names is None:
        if not cumulative_data.empty:
            # 仮に最後のデータを使用
            data_with_result = pd.merge(
                cumulative_data, 
                odds_results[['レースID', '三連単結果']], 
                on='レースID', 
                how='left'
            )
            # ターゲット変数の定義（dummy）
            data_with_result['target'] = 0
            # 特徴量の指定（仮）
            features = [
                '会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                '性別','前期能力指数', '今期能力指数', '平均スタートタイミング',
                '勝率', '複勝率', '優勝回数', '優出回数'
            ]
            X = data_with_result[features]
            categorical_features = ['会場', 'weather','wind_d', 'ESC','性別','級別','支部','艇番']
            X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)
            numeric_features = [col for col in X.columns if col not in categorical_features]
            X_numeric = X[numeric_features]
            X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
            feature_names = X_processed.columns.tolist()
            pd.Series(feature_names).to_csv(feature_names_path, index=False)

if __name__ == '__main__':
    create_model()
