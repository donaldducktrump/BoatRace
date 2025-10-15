# create_model.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合
import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, load_before_data, merge_before_data
from get_data import load_fan_data, add_course_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib

def save_dataframe(df, file_path):
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # データフレームをピクル形式で保存
    df.to_pickle(file_path)

def load_dataframe(file_path):
    # ファイルが存在する場合は読み込み
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        return None

def create_model():
    # 使用する月のフォルダを指定
    month_folders = ['2311', '2312', '2401', '2402', '2403', '2404']
    
    date_files = {
        '2311': [f'2311{day:02d}' for day in range(1, 31)],  # 231101 - 231130
        '2312': [f'2312{day:02d}' for day in range(1, 32)],  # 231201 - 231231
        '2401': [f'2401{day:02d}' for day in range(1, 31)],  # 240101 - 240130
        '2402': [f'2402{day:02d}' for day in range(1, 29)],  # 240201 - 240228
        '2403': [f'2403{day:02d}' for day in range(1, 32)],  # 240301 - 240331
        '2404': [f'2404{day:02d}' for day in range(1, 31)],  # 240401 - 240430
    }

    # データを結合
    b_data_list = []
    k_data_list = []
    before_data_list = []

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
                    data = merge_before_data(data, before_data)
                    b_data_list.append(data)
                else:
                    print(f"データが不足しています: {date}")

            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file}, {k_file}, または {before_file}")

    if not b_data_list:
        print("データが正しく読み込めませんでした。")
        return
    
    # 全データを結合
    data = pd.concat(b_data_list, ignore_index=True)

    # 前処理
    print("data columns:", data.columns.tolist())
    print(data)

    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    # 必要な列を指定してマージ
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')

    # データの前処理
    data = preprocess_data(data)
    print("data columns after preprocessing:", data.columns.tolist())
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
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h','性別','前期能力指数', '今期能力指数', '平均スタートタイミング', '勝率', '複勝率', '優勝回数', '優出回数']
    
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
    print(X_processed.head())

    # データの分割（訓練、キャリブレーション、テスト）
    X_train_full, X_temp, y_train_full, y_temp = train_test_split(
        X_processed, y, test_size=0.25, random_state=42, stratify=y)

    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp)

    print(f'Training set size: {X_train_full.shape[0]}')
    print(f'Calibration set size: {X_calib.shape[0]}')
    print(f'Test set size: {X_test.shape[0]}')

    # データをDatasetクラスに格納
    dtrain = lgb.Dataset(X_train_full, label=y_train_full)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    # 使用するパラメータ（修正済み）
    params = {
        'objective': 'binary',  # 'multiclass' から 'binary' に変更
        'metric': 'binary_logloss',  # 'multi_logloss' から 'binary_logloss' に変更
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
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True),
                   lgb.log_evaluation(verbose_eval)]
    )

    # キャリブレーションセットに対する予測確率を取得
    y_calib_pred = gbm.predict(X_calib, num_iteration=gbm.best_iteration)

    # ロジスティック回帰によるキャリブレーション（プラットスケーリング）
    lr_calibrator = LogisticRegression()
    lr_calibrator.fit(y_calib_pred.reshape(-1, 1), y_calib)

    # アイソトニック回帰によるキャリブレーション
    iso_calibrator = IsotonicRegression(out_of_bounds='clip')
    iso_calibrator.fit(y_calib_pred, y_calib)

    # テストセットに対する予測確率を取得
    y_test_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # ロジスティック回帰によるキャリブレーション適用
    y_test_calibrated_lr = lr_calibrator.predict_proba(y_test_pred.reshape(-1, 1))[:, 1]

    # アイソトニック回帰によるキャリブレーション適用
    y_test_calibrated_iso = iso_calibrator.predict(y_test_pred)

    # 閾値を0.5に設定して予測ラベルを取得
    y_test_pred_labels = (y_test_pred >= 0.5).astype(int)
    y_test_calibrated_lr_labels = (y_test_calibrated_lr >= 0.5).astype(int)
    y_test_calibrated_iso_labels = (y_test_calibrated_iso >= 0.5).astype(int)

    # 精度の計算
    accuracy_original = accuracy_score(y_test, y_test_pred_labels)
    accuracy_lr = accuracy_score(y_test, y_test_calibrated_lr_labels)
    accuracy_iso = accuracy_score(y_test, y_test_calibrated_iso_labels)

    print(f'Original Accuracy on test data: {accuracy_original:.4f}')
    print(f'Logistic Regression Calibrated Accuracy on test data: {accuracy_lr:.4f}')
    print(f'Isotonic Regression Calibrated Accuracy on test data: {accuracy_iso:.4f}')

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
    plt.savefig('feature_importance_calibrated.png')
    plt.show()

    # モデルの保存
    gbm.save_model('boatrace_model_first_calibrated.txt')

    # 特徴量名の保存
    X_processed.columns.to_series().to_csv('feature_names_first_calibrated.csv', index=False)
    
    # キャリブレーションモデルの保存
    joblib.dump(lr_calibrator, 'lr_calibrator.pkl')
    joblib.dump(iso_calibrator, 'iso_calibrator.pkl')
    
    # クロスバリデーションの実装（オプション）
    print("\nCross-validation with early stopping:")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X_processed, y)):
        X_train_cv, X_valid_cv = X_processed.iloc[train_idx], X_processed.iloc[test_idx]
        y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[test_idx]

        dtrain_cv = lgb.Dataset(X_train_cv, label=y_train_cv)
        dvalid_cv = lgb.Dataset(X_valid_cv, label=y_valid_cv, reference=dtrain_cv)

        gbm_cv = lgb.train(
            params,
            dtrain_cv,
            valid_sets=[dvalid_cv],
            num_boost_round=10000,
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True),
                   lgb.log_evaluation(verbose_eval)]
        )

        y_pred_cv = gbm_cv.predict(X_valid_cv, num_iteration=gbm_cv.best_iteration)
        y_pred_labels_cv = (y_pred_cv >= 0.5).astype(int)
        accuracy_cv = accuracy_score(y_valid_cv, y_pred_labels_cv)
        scores.append(accuracy_cv)
        print(f'Fold {i+1} Accuracy: {accuracy_cv:.4f}')

    print(f'\nMean Accuracy: {np.mean(scores):.4f}')

if __name__ == '__main__':
    create_model()
