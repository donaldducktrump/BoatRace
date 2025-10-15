# create_model.py

import os
import pandas as pd
import lightgbm as lgb
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from ...utils.mymodule import (
    load_b_file, load_k_file, preprocess_data, load_before_data,
    merge_before_data, load_before1min_data, preprocess_data1min
)
from get_data import load_fan_data, add_course_data

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

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

def prepare_data():
    """
    データの読み込みと前処理を行い、訓練データとテストデータを返します。
    この関数はOptunaの各試行で繰り返さないように、一度だけ実行します。
    """
    # 使用する月フォルダーの指定
    month_folders = ['2311', '2312', '2401', '2402', '2403', '2404']
    
    date_files = {
        '2311': [f'2311{day:02d}' for day in range(1, 31)],  # 231101 - 231130
        '2312': [f'2312{day:02d}' for day in range(1, 32)],  # 231201 - 231231
        '2401': [f'2401{day:02d}' for day in range(1, 31)],  # 240101 - 240131
        '2402': [f'2402{day:02d}' for day in range(1, 29)],  # 240201 - 240228
        '2403': [f'2403{day:02d}' for day in range(1, 32)],  # 240301 - 240331
        '2404': [f'2404{day:02d}' for day in range(1, 31)],  # 240401 - 240430
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
        return None, None, None, None

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

    data = preprocess_data(data)
    print("data columns:", data.columns.tolist())
    print(data.head())

    # 目的変数の作成
    def target_label(x):
        x = int(x)
        if x == 1:
            return 0
        else:
            return 1

    data['target'] = data['着'].apply(target_label)

    # 特徴量と目的変数
    features = [
        '会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
        '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
        'ET', 'tilt', 'EST','ESC',
        'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
        '性別','前期能力指数', '今期能力指数', '平均スタートタイミング', 
        '勝率', '複勝率', '優勝回数', '優出回数'
    ]
    
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
        X_processed, y, test_size=0.25, random_state=42
    )

    return X_train, X_test, y_train, y_test

def objective(trial, X_train, X_valid, y_train, y_valid):
    """
    Optunaの目的関数。
    ハイパーパラメータをサンプリングし、LightGBMモデルを訓練して検証精度を返します。
    """
    param = {
        'objective': 'binary',  # 二値分類
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
    }

    verbose_eval = 10
    # データセットの作成
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    # モデルの学習
    gbm = lgb.train(
        param,
        lgb_train,
        valid_sets=[lgb_valid],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True),
                    lgb.log_evaluation(verbose_eval)]
    )

    # テストデータで予測
    y_pred = gbm.predict(X_valid)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # 精度の計算
    accuracy = accuracy_score(y_valid, y_pred_labels)

    return accuracy

def tune_hyperparameters(X_train, X_test, y_train, y_test):
    """
    Optunaを用いてハイパーパラメータのチューニングを行います。
    """
    # Optunaのスタディを作成
    study = optuna.create_study(direction='maximize', study_name='LightGBM Optimization')
    
    # スタディの最適化
    study.optimize(
        lambda trial: objective(
            trial, X_train, X_test, y_train, y_test
        ),
        n_trials=100,
        timeout=600  # 10分
    )

    print("最適なハイパーパラメータ: ", study.best_params)
    print("最高の精度: ", study.best_value)

    return study.best_params

def train_final_model(X_train, X_test, y_train, y_test, best_params):
    """
    最適なハイパーパラメータを用いて最終モデルを訓練し、評価します。
    """
    # パラメータに固定値を追加
    params = best_params.copy()
    params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42
    })

    # データセットの作成
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test)
    verbose_eval = 10 
    # モデルの学習
    gbm = lgb.train(
        params,
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
    gbm.save_model('boatrace_model_final.txt')

    # 特徴量名の保存
    X_train.columns.to_series().to_csv('feature_names_final.csv', index=False)

    # クロスバリデーションの実装
    print("\nCross-validation with early stopping:")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train)):
        X_train_cv, X_valid_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_train_cv, y_valid_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]

        dtrain_cv = lgb.Dataset(X_train_cv, label=y_train_cv)
        dvalid_cv = lgb.Dataset(X_valid_cv, label=y_valid_cv)

        gbm_cv = lgb.train(
            params,
            dtrain_cv,
            valid_sets=[dvalid_cv],
            num_boost_round=10000,
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )

        y_pred_cv = gbm_cv.predict(X_valid_cv, num_iteration=gbm_cv.best_iteration)
        y_pred_labels_cv = np.argmax(y_pred_cv, axis=1)
        accuracy_cv = accuracy_score(y_valid_cv, y_pred_labels_cv)
        scores.append(accuracy_cv)
        print(f'Fold {i+1} Accuracy: {accuracy_cv:.4f}')

    print(f'\nMean Accuracy: {np.mean(scores):.4f}')

def create_model():
    # データの準備
    X_train, X_test, y_train, y_test = prepare_data()
    if X_train is None:
        print("データの準備に失敗しました。処理を終了します。")
        return

    # ハイパーパラメータのチューニング
    best_params = tune_hyperparameters(X_train, X_test, y_train, y_test)

    # 最適なパラメータで最終モデルを訓練
    train_final_model(X_train, X_test, y_train, y_test, best_params)

if __name__ == '__main__':
    create_model()
