# inspection.py

import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, load_before_data, merge_before_data
from get_data import load_fan_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # 新しく追加

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

def inspect_model():
    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_first_calibrated.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # キャリブレーションモデルの読み込み
    try:
        lr_calibrator = joblib.load('lr_calibrator.pkl')
        iso_calibrator = joblib.load('iso_calibrator.pkl')
    except Exception as e:
        print(f"Failed to load calibrators: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_first_calibrated.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2408','2409','2410']
    date_files = {
        '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(1, 16)],   # 241001 - 241015
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
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'

                b_data = load_b_file(b_file)
                k_data, odds_list_part = load_k_file(k_file)
                before_data = load_before_data(before_file)

                if not b_data.empty and not k_data.empty and not before_data.empty:
                    # データのマージ
                    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
                    data = merge_before_data(data, before_data)
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

    # 全データを結合
    data = pd.concat(data_list, ignore_index=True)
    odds_list = pd.concat(odds_list1, ignore_index=True)

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
    print(data)

    nan_counts = data.isna().sum()
    print("各列のNaNの個数:")
    print(nan_counts)

    # 特徴量の指定
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                 'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h','前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather', 'wind_d','ESC','性別', '支部', '級別', '艇番']
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # 数値データの選択
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]

    # 特徴量の結合
    X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
    # 天候_雪 カラムが存在しない場合に追加する
    # if '天候_雪' not in X_processed.columns:
    #     X_processed['天候_雪'] = 0
    for col in feature_names:
        if col not in X_processed.columns:
            X_processed[col] = 0
    print("X_processed columns:", X_processed.columns.tolist())
    X_processed = X_processed[feature_names]  # 学習時の特徴量と順序を合わせる

    # 予測
    y_pred = gbm.predict(X_processed, num_iteration=gbm.best_iteration)

    # LightGBMのbinary分類では、y_predはクラス1の確率
    data['prob_1'] = y_pred  # クラス1の確率
    data['prob_0'] = 1 - y_pred  # クラス0の確率

    # キャリブレーション適用
    y_pred_calibrated_lr = lr_calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
    y_pred_calibrated_iso = iso_calibrator.predict(y_pred)

    # キャリブレーション済み確率を新しいカラムに追加
    data['prob_calibrated_lr'] = 1-y_pred_calibrated_lr
    data['prob_calibrated_iso'] = 1-y_pred_calibrated_iso


    # デバッグ: キャリブレーション済み確率の確認
    print("Calibrated probabilities (Logistic Regression):", data['prob_calibrated_lr'].head())
    print("Calibrated probabilities (Isotonic Regression):", data['prob_calibrated_iso'].head())

    correct_predictions = 0
    total_races = 0
    total_bet = 0
    total_return = 0
    payout_list = []

    # 3連複 (Trio) 用の変数
    correct_predictions_trio = 0
    total_races_trio = 0
    total_bet_trio = 0
    total_return_trio = 0
    payout_list_trio = []

    # レースごとに予測順位を決定
    race_ids = data['レースID'].unique()

    for race_id in race_ids:
        race_data = data[data['レースID'] == race_id]
        race_data = race_data.sort_values('prob_calibrated_iso', ascending=False)

        # レースデータに少なくとも1艇が含まれていることを確認
        if len(race_data) < 1:
            print(f"レースID {race_id} のデータが不足しています。スキップします。")
            continue

        # 予測された1着を決定
        predicted_first = race_data.iloc[0]['艇番']

        # 実際の1着を取得
        actual_first = race_data[race_data['着'] == 1]['艇番'].values[0]

        # 賭け金を追加 (1レースあたり100円)
        total_bet += 100

        # 予測が的中したか確認
        if predicted_first == actual_first:
            correct_predictions += 1

            # オッズ情報の取得
            odds_row = odds_list[odds_list['レースID'] == race_id]
            if not odds_row.empty:
                winning_odds = odds_row["単勝オッズ"].values[0]

                if pd.notna(winning_odds):
                    payout = 100 * winning_odds  # 当たった場合は100円×オッズ
                    total_return += payout
                    payout_list.append(payout)
                else:
                    print(f"オッズがNaNです: レースID {race_id}")
                    payout_list.append(0)
            else:
                print(f"オッズ情報が見つかりません: レースID {race_id}")
                payout_list.append(0)
        else:
            payout_list.append(0)

        total_races += 1

    # 正答率を表示
    accuracy = correct_predictions / total_races if total_races > 0 else 0
    print(f"\n単勝の正答率: {accuracy * 100:.2f}%")

    # 回収率の計算
    if total_bet > 0:
        return_rate = (total_return / total_bet) * 100
    else:
        return_rate = 0

    # 結果の表示
    print(f"\n単勝の総賭け金: {total_bet}円")
    print(f"単勝の総回収金額: {total_return}円")
    print(f"単勝の回収率: {return_rate:.2f}%")

    # 配当金の統計情報
    payout_series = pd.Series(payout_list)
    print(f"\n配当金の統計情報:")
    print(f"最大値: {payout_series.max()}円")
    print(f"最小値: {payout_series.min()}円")
    print(f"平均: {payout_series.mean()}円")
    print(f"標準偏差: {payout_series.std()}円")

    # 配当金のヒストグラムをプロット
    plt.figure(figsize=(10, 6))
    sns.histplot(payout_series, bins=20, kde=False, color='blue')
    plt.title("単勝の配当金ヒストグラム")
    plt.xlabel("配当金額 (円)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('odds_histogram_calibrated.png')
    plt.show()
    print("ヒストグラムが 'odds_histogram_calibrated.png' として保存されました。")

if __name__ == '__main__':
    inspect_model()
