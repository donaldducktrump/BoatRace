# model_1min.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import (
    load_b_file, load_k_file, preprocess_data, preprocess_data1min,
    preprocess_data_old, load_before_data, load_before1min_data, merge_before_data
)
from get_data import load_fan_data
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.isotonic import IsotonicRegression
import seaborn as sns
import os
import joblib  # キャリブレーションモデルのロードと保存用

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

def create_model():
    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_first_calibrated.txt')  # キャリブレーション済みモデルをロード
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # キャリブレーションモデルの読み込み
    try:
        lr_calibrator = joblib.load('lr_calibrator.pkl')  # ロジスティック回帰キャリブレーションモデル
        iso_calibrator = joblib.load('iso_calibrator.pkl')  # アイソトニック回帰キャリブレーションモデル
    except Exception as e:
        print(f"Failed to load calibrators: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_first_calibrated.csv').squeeze().tolist()

    # LightGBM のパラメータを分類用に修正
    # params = {
    #     'objective': 'binary',  # 'regression' から 'binary' に変更
    #     'metric': 'binary_logloss',
    #     'random_state': 42,
    #     'boosting_type': 'gbdt',
    #     'verbose': -1,
    #     'learning_rate': 0.05,
    #     'num_leaves': 31,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    # }

    # 評価期間のデータ日付リスト作成
    month_folders = ['2410']
    date_files = {
        '2410': [f'2410{day:02d}' for day in range(15,23)],   # 241015 - 241022
    }

    # データを結合
    b_data_list = []
    k_data_list = []
    odds_list1 = []
    data_list = []

    for month in month_folders:
        for date in date_files[month]:
            try:
                b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'
                k_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_data/{month}/K{date}.TXT'
                before1min_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data_1min/{month}/beforeinfo1min_{date}.txt'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'

                b_data = load_b_file(b_file)
                k_data, odds_list_part = load_k_file(k_file)
                before_data = load_before_data(before_file)
                before1min_data = load_before1min_data(before1min_file)

                if not b_data.empty and not k_data.empty and not before_data.empty:
                    # データのマージ
                    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
                    before_data = remove_common_columns(data, before_data, on_columns=['選手登番', 'レースID', '艇番'])
                    data = merge_before_data(data, before_data)
                    before1min_data = remove_common_columns(data, before1min_data, on_columns=['選手登番', 'レースID', '艇番'])
                    data = merge_before_data(data, before1min_data)
                    data_list.append(data)
                    if not odds_list_part.empty:
                        odds_list1.append(odds_list_part)
                else:
                    print(f"データが不足しています: {date}")
            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file}, {k_file}, または {before_file}")

    if not data_list:
        print("データが正しく読み込めませんでした。")
        return

    # 過去のwin_oddsを取得
    month_folders_odds = ['2310', '2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410']
    date_files_odds = {
        '2310': [f'2310{day:02d}' for day in range(1, 32)],  # 231001 - 231031
        '2311': [f'2311{day:02d}' for day in range(1, 31)],  # 231101 - 231130
        '2312': [f'2312{day:02d}' for day in range(1, 32)],  # 231201 - 231231
        '2401': [f'2401{day:02d}' for day in range(1, 32)],  # 240101 - 240131
        '2402': [f'2402{day:02d}' for day in range(1, 29)],  # 240201 - 240228
        '2403': [f'2403{day:02d}' for day in range(1, 32)],  # 240301 - 240331
        '2404': [f'2404{day:02d}' for day in range(1, 31)],  # 240401 - 240430
        '2405': [f'2405{day:02d}' for day in range(1, 32)],  # 240501 - 240531
        '2406': [f'2406{day:02d}' for day in range(1, 31)],  # 240601 - 240630
        '2407': [f'2407{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(1,22)],   # 241001 - 241021
    }
    data_list_odds = []
    for month in month_folders_odds:
        for date in date_files_odds[month]:
            try:
                b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'
                bdata_odds = load_b_file(b_file)
                before_data_odds = load_before_data(before_file)

                if not bdata_odds.empty and not before_data_odds.empty:
                    # データのマージ
                    before_data_odds = remove_common_columns(bdata_odds, before_data_odds, on_columns=['選手登番', 'レースID', '艇番'])
                    bdata_odds = merge_before_data(bdata_odds, before_data_odds)
                    data_list_odds.append(bdata_odds)
                else:
                    print(f"データが不足しています: {date}")
            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file}, または {before_file}")
    data_odds = pd.concat(data_list_odds, ignore_index=True)
    data_odds = preprocess_data_old(data_odds)
    data_odds['win_odds_mean'] = data_odds.groupby(['選手登番','艇番'])['win_odds'].transform(lambda x: x.mean())
    print(data_odds[["win_odds_mean","選手登番","艇番","win_odds"]])
    print(data_odds[data_odds['選手登番']=='5107'][["win_odds_mean","選手登番","艇番","win_odds"]])

    # 全データを結合
    data = pd.concat(data_list, ignore_index=True)
    odds_list = pd.concat(odds_list1, ignore_index=True)

    # 前処理
    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    # 必要な列を指定してマージ
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')

    data = preprocess_data1min(data)

    # 特徴量の指定
    features = [
        '会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
        '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
        'ET', 'tilt', 'EST','ESC',
        'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
        '前期能力指数', '今期能力指数','平均スタートタイミング', '性別', 
        '勝率', '複勝率', '優勝回数', '優出回数'
    ]
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather', 'wind_d','ESC', '艇番', '性別', '支部', '級別']
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # 数値データの選択
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]

    # 特徴量の結合
    X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

    # 必要な特徴量が存在しない場合に追加
    for col in feature_names:
        if col not in X_processed.columns:
            X_processed[col] = 0

    X_processed = X_processed[feature_names]  # 学習時の特徴量と順序を合わせる

    # 予測
    y_pred = gbm.predict(X_processed)

    # LightGBMのbinary分類では、y_predはクラス1の確率
    # prob_1 = y_pred
    # prob_0 = 1 - y_pred

    # キャリブレーション適用
    # ロジスティック回帰によるキャリブレーション
    y_pred_calibrated_lr = lr_calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
    data['prob_0'] = 1 - y_pred_calibrated_lr  # キャリブレーション済みprob_0

    # prob_1 はクラス1の確率
    data['prob_1'] = y_pred  # クラス1の確率

    # デバッグ: キャリブレーション済み確率の確認
    print("Calibrated probabilities (Logistic Regression):", data['prob_0'].head())

    # 目的変数の作成: 確定後のオッズ
    # 各艇ごとにデータを整形
    boats = [1, 2, 3, 4, 5, 6]
    data_list_boats = []
    for boat in boats:
        boat_data = data.copy()
        boat_data = boat_data[boat_data['艇番'] == boat].copy()
        boat_data['final_odds'] = boat_data['win_odds']  # 確定後のオッズをターゲット
        boat_data['before1min_odds'] = boat_data['win_odds1min']  # 1分前のオッズを特徴量
        boat_data['boat_number'] = boat  # 舟番号を特徴量として追加
        data_list_boats.append(boat_data)

    # 各艇のデータを結合
    data = pd.concat(data_list_boats, ignore_index=True)
    print(data)

    # '選手登番' と '艇番' ごとに 'win_odds_mean' を集約（例: 平均を取る）
    data_odds_grouped = data_odds.groupby(['選手登番', '艇番'], as_index=False)['win_odds_mean'].mean()

    # '選手登番' と '艇番' をキーにして、'win_odds_mean' を data にマージ
    data = data.merge(data_odds_grouped, on=['選手登番', '艇番'], how='left')
    print(data)

    # 不要な列の削除（ターゲット以外に含まれる最終オッズなど）
    # 必要に応じて調整
    # 例: data = data.drop(['before_odds'], axis=1)

    # 特徴量と目的変数
    features_model = [
        '会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
        '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
        'ET', 'tilt', 'EST','ESC',
        'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
        # 'win_odds1min', 'prob_0','place_odds1min','win_odds_mean', 
        'win_odds1min', 'prob_0','place_odds1min','win_odds_mean', 
        '性別', '勝率', '複勝率', '優勝回数', '優出回数',
        '前期能力指数', '今期能力指数','平均スタートタイミング'
    ]

    target = 'win_odds'
    data = data.dropna()
    X = data[features_model]
    y = data[target]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features_model = ['会場', 'weather','wind_d', 'ESC', '艇番', '性別', '支部', '級別']
    X_categorical_model = pd.get_dummies(X[categorical_features_model], drop_first=True)

    # 数値データの選択
    numeric_features_model = [col for col in X.columns if col not in categorical_features_model]
    X_numeric_model = X[numeric_features_model]

    # 特徴量の結合
    X_processed_model = pd.concat([X_numeric_model.reset_index(drop=True), X_categorical_model.reset_index(drop=True)], axis=1)

    # # 必要な特徴量が存在しない場合に追加
    # for col in feature_names:
    #     if col not in X_processed_model.columns:
    #         X_processed_model[col] = 0

    # X_processed_model = X_processed_model[feature_names]  # 学習時の特徴量と順序を合わせる

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed_model, y, test_size=0.25, random_state=42
    )

    # LightGBMのデータセット作成
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    verbose_eval = 100  # 学習時のスコア推移を100ラウンドごとに表示
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'random_state': 42,
        'boosting_type': 'gbdt',
        'verbose': -1,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
    }
    # LightGBM学習
    gbm = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(verbose_eval)
        ]
    )

    # テストデータで予測
    y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # 精度の計算 (MAE)
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f'Mean Absolute Error on test data: {mae:.4f}')

    # 特徴量の重要度を取得してプロット
    importance = gbm.feature_importance(importance_type='gain')
    feature_name = gbm.feature_name()
    importance_df = pd.DataFrame({'feature': feature_name, 'importance': importance})
    importance_df = importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 12))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_odds_calibrated.png')
    plt.show()

    # 予測結果をデータフレームに追加
    odds_pred = gbm.predict(X_processed_model)
    data['odds_pred'] = odds_pred

    # # Reshape data to fit the linear model
    # X_lr = data['odds_pred'].values.reshape(-1, 1)  # Reshape to (n_samples, 1)
    # y_lr = data['win_odds'].values

    # # Fit a linear regression model
    # reg = LinearRegression().fit(X_lr, y_lr)

    # # Obtain the scaling factor and intercept from the regression model
    # scaling_factor = reg.coef_[0]
    # intercept = reg.intercept_

    # # Adjust the predicted odds based on the scaling factor and intercept
    # data['odds_pred_linear'] = scaling_factor * data['odds_pred'] + intercept

    # # Fit isotonic regression
    # iso_reg = IsotonicRegression().fit(data['odds_pred_linear'], data['win_odds'])

    # # Calibrate the predictions
    # data['odds_pred_adjusted'] = iso_reg.transform(data['odds_pred_linear'])

    # モデルの保存
    gbm.save_model('boatrace_odds_model_calibrated.txt')

    # 特徴量名の保存
    X_processed_model.columns.to_series().to_csv('feature_names_odds_calibrated.csv', index=False)

    # # ロジスティック回帰モデルの保存
    # joblib.dump(reg, 'lr_reg.pkl')
    # print("Logistic Regressionモデルが 'lr_reg.pkl' として保存されました。")
    # # Isotonic Regressionモデルの保存
    # joblib.dump(iso_reg, 'iso_reg.pkl')
    # print("Isotonic Regressionモデルが 'iso_reg.pkl' として保存されました。")

if __name__ == '__main__':
    create_model()
