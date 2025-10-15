import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

import pandas as pd
import numpy as np
import seaborn as sns
import os

# 追加インポート
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data

def inspect_model():
    # モデルと前処理器の読み込み
    model = keras.models.load_model('boatrace_dnn_model.h5')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')

    # 評価期間のデータ日付リスト作成
    month_folders = ['2408', '2409', '2410']
    date_files = {
        '2408': [f'2408{day:02d}' for day in range(1, 32)],
        '2409': [f'2409{day:02d}' for day in range(1, 31)],
        '2410': [f'2410{day:02d}' for day in range(1, 9)],
    }

    # データを結合
    b_data_list = []
    k_data_list = []
    odds_list1 = []

    for month in month_folders:
        for date in date_files[month]:
            try:
                b_file = f'b_data/{month}/B{date}.TXT'
                k_file = f'k_data/{month}/K{date}.TXT'
                b_data = load_b_file(b_file)
                k_data, odds_list_part = load_k_file(k_file)

                if not b_data.empty and not k_data.empty:
                    b_data_list.append(b_data)
                    k_data_list.append(k_data)
                if not odds_list_part.empty:
                    odds_list1.append(odds_list_part)

            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file} または {k_file}")

    # b_dataとk_dataの結合
    b_data = pd.concat(b_data_list, ignore_index=True) if b_data_list else pd.DataFrame()
    k_data = pd.concat(k_data_list, ignore_index=True) if k_data_list else pd.DataFrame()
    odds_list = pd.concat(odds_list1, ignore_index=True) if odds_list1 else pd.DataFrame()

    if b_data.empty or k_data.empty:
        print("データが正しく読み込めませんでした。")
        return
    if odds_list.empty:
        print("オッズデータが正しく読み込めませんでした。")
        return

    # データのマージ（選手登番とレースIDで結合）
    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
    if data.empty:
        print("データのマージ後に結果が空です。")
        return

    # カラム名の整合性を確認・修正
    columns_to_check = [col for col in data.columns if '_x' in col]

    for col_x in columns_to_check:
        col_y = col_x.replace('_x', '_y')
        
        if col_y in data.columns:
            if data[col_x].equals(data[col_y]):
                data[col_x.replace('_x', '')] = data[col_x]
                data.drop([col_x, col_y], axis=1, inplace=True)
            # else:
                # print(f"Warning: {col_x}と{col_y}の内容が一致しません。手動で確認してください。")

    # 選手名の統一
    for idx, row in data.iterrows():
        if row['選手名_x'] != row['選手名_y']:
            if row['選手名_x'][:4] == row['選手名_y'][:4]:
                data.at[idx, '選手名'] = row['選手名_y']
            # else:
                # print(f"Warning: 選手名_x と 選手名_y の最初の4文字が一致しません。手動で確認してください。\n{row[['選手名_x', '選手名_y']]}")
        else:
            data.at[idx, '選手名'] = row['選手名_x']

    # 重複列を削除
    data.drop(['選手名_x', '選手名_y'], axis=1, inplace=True)

    # 結果のデバッグ表示
    print("選手名の統一が完了しました。")
    print(data.head())

    # データの前処理
    data = preprocess_data(data)
    if data.empty:
        print("前処理後のデータが空です。")
        return

    # 特徴量の指定
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                '展示タイム', '天候', '風向', '風量', '波']

    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', '艇番', '級別', '支部', '天候', '風向']
    X_categorical = encoder.transform(X[categorical_features])

    # 数値データの標準化
    numeric_features = ['年齢', '体重', '全国勝率', '全国2連率', '当地勝率', '当地2連率',
                        'モーター2連率', 'ボート2連率', '展示タイム', '風量', '波']
    X_numeric = X[numeric_features]
    X_numeric[numeric_features] = scaler.transform(X_numeric[numeric_features])

    # 特徴量の結合
    X_processed = np.hstack([X_categorical, X_numeric.values])

    # 列名の設定（必要に応じて）
    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    # 予測の実行
    y_pred = model.predict(X_processed)

    # 確率を取得
    probabilities = y_pred

    # 確率をデータフレームに追加
    data['prob_0'] = probabilities[:, 0]  # クラス0の確率（1着）
    data['prob_1'] = probabilities[:, 1]  # クラス1の確率（2着）
    data['prob_2'] = probabilities[:, 2]  # クラス2の確率（3着）

    # 以下、元のコードの処理を続けます
    # 回収率計算のための変数
    total_bet = 0
    total_return = 0
    payout_list = []
    betting_races = []
    venue_bet_return = {}
    correct_predictions = 0
    total_races = 0
    race_ids = data['レースID'].unique()

    for race_id in race_ids:
        race_data = data[data['レースID'] == race_id]
        race_data = race_data.sort_values('prob_0', ascending=False)

        # レースデータに少なくとも3艇が含まれていることを確認
        if len(race_data) < 3:
            print(f"レースID {race_id} のデータが不足しています。スキップします。")
            continue

        # 予測された1着、2着、3着を決定
        predicted_first = race_data.iloc[0]['艇番']
        predicted_second = race_data.iloc[1]['艇番']
        predicted_third = race_data.iloc[2]['艇番']

        

        # 実際の1着、2着、3着の選手（'着'カラムがそれぞれ1, 2, 3の選手を取得）
        actual_first = race_data[race_data['着'] == 1]['艇番'].values[0]
        actual_second = race_data[race_data['着'] == 2]['艇番'].values[0]
        actual_third = race_data[race_data['着'] == 3]['艇番'].values[0]

        # 賭け金を追加 (1レースあたり100円)
        total_bet += 100

        # #3連単の場合
        # # 正答率の計算（予測が的中したかを確認）
        # if (predicted_first == actual_first and
        #     predicted_second == actual_second and
        #     predicted_third == actual_third):
        #     correct_predictions += 1

        #     # 三連単オッズ情報の取得
        #     odds_row = odds_list[odds_list['レースID'] == race_id]

        #     # デバッグ: オッズ情報が存在するか確認
        #     if odds_row.empty:
        #         print(f"オッズ情報が見つかりません: レースID {race_id}")
        #         payout_list.append(0)
        #         continue

        #     # 予測が的中した場合、オッズに基づいて回収金額を計算
        #     winning_odds = odds_row["三連単オッズ"].values[0]

        #     # デバッグ: オッズが正常に取得されているか確認
        #     if pd.isna(winning_odds):
        #         print(f"オッズがNaNです: レースID {race_id}")
        #         payout_list.append(0)
        #         continue

        #     payout = 100 * winning_odds  # 当たった場合は100円×オッズ
        #     total_return += payout
        #     payout_list.append(payout)  # 配当金をリストに追加

        # else:
        #     payout_list.append(0)  # 外れた場合は0円

        # total_races += 1

        # # 三連単予測結果と実際の結果を表示
        # print(f"レースID: {race_id}")
        # print(f"予測 - 1着: {predicted_first}, 2着: {predicted_second}, 3着: {predicted_third}")
        # print(f"実際 - 1着: {actual_first}, 2着: {actual_second}, 3着: {actual_third}")


        # #複勝の場合
        # if predicted_first == actual_first:
        #     correct_predictions+=1
            
        #     #複勝オッズ情報の取得
        #     odds_row = odds_list[odds_list['レースID'] == race_id]
            
        #     # デバッグ: オッズ情報が存在するか確認
        #     if odds_row.empty:
        #         print(f"オッズ情報が見つかりません: レースID {race_id}")
        #         payout_list.append(0)
        #         continue
            
        #     #予測が的中した場合、オッズに基づいて回収金額を計算
        #     winning_odds1 = odds_row["複勝オッズ1"].values[0]

        #     #デバッグ:　オッズ情報が存在するか確認
        #     if pd.isna(winning_odds1):
        #         print(f"オッズがNaNです: レースID {race_id}")
        #         payout_list.append(100)
        #         continue

        #     payout = 100 * winning_odds1  # 当たった場合は100円×オッズ
        #     total_return += payout
        #     payout_list.append(payout)  # 配当金をリストに追加

        # elif predicted_first == actual_second:
        #     correct_predictions+=1
            
        #     #複勝オッズ情報の取得
        #     odds_row = odds_list[odds_list['レースID'] == race_id]
            
        #     # デバッグ: オッズ情報が存在するか確認
        #     if odds_row.empty:
        #         print(f"オッズ情報が見つかりません: レースID {race_id}")
        #         payout_list.append(0)
        #         continue
            
        #     #予測が的中した場合、オッズに基づいて回収金額を計算
        #     winning_odds2 = odds_row["複勝オッズ2"].values[0]

        #     #デバッグ:　オッズ情報が存在するか確認
        #     if pd.isna(winning_odds2):
        #         print(f"オッズがNaNです: レースID {race_id}")
        #         payout_list.append(100)
        #         continue

        #     payout = 100 * winning_odds2  # 当たった場合は100円×オッズ
        #     total_return += payout
        #     payout_list.append(payout)  # 配当金をリストに追加

        # else:
        #     payout_list.append(0)  # 外れた場合は0円

        # total_races += 1
        
        # # 複勝予測結果と実際の結果を表示
        # print(f"レースID: {race_id}")
        # print(f"予測 - 1着: {predicted_first}, 2着: {predicted_second}, 3着: {predicted_third}")
        # print(f"実際 - 1着: {actual_first}, 2着: {actual_second}, 3着: {actual_third}")

        #単勝の場合
        # 実際の1着の選手（'着'カラムが1の選手を取得）
        actual_first = race_data[race_data['着'] == 1]['艇番'].values[0]
        
        # 賭け金を追加 (1レースあたり100円)
        total_bet += 100

        # 正答率の計算
        if predicted_first == actual_first:
            correct_predictions += 1
            # オッズ情報の取得
            odds_row = odds_list[odds_list['レースID'] == race_id]
            # 予測が的中した場合、オッズに基づいて回収金額を計算
            winning_odds =  odds_row["単勝オッズ"].values[0]
            payout = 100 * winning_odds  # 当たった場合は100円×オッズ
            total_return += payout
            payout_list.append(payout)  # 配当金をリストに追加
        else:
            payout_list.append(0)  # 外れた場合は0円



    # 正答率を表示
    accuracy = correct_predictions / total_races if total_races > 0 else 0
    print(f"\n三連単の正答率: {accuracy * 100:.2f}%")

    # 的中率の計算


    # 回収率の計算
    if total_bet > 0:
        return_rate = (total_return / total_bet) * 100
    else:
        return_rate = 0
    
    # 結果の表示
    print(f"\n総賭け金: {total_bet}円")
    print(f"総回収金額: {total_return}円")
    print(f"回収率: {return_rate:.2f}%")

    # 配当金の統計情報
    payout_series = pd.Series(payout_list)
    print(f"\n配当金の統計情報:")
    print(f"最大値: {payout_series.max()}円")
    print(f"最小値: {payout_series.min()}円")
    print(f"平均: {payout_series.mean()}円")
    print(f"標準偏差: {payout_series.std()}円")

    # 配当金のヒストグラムをプロット
    plt.figure(figsize=(10, 6))
    sns.histplot(payout_series, bins=20, kde=False)
    plt.title("配当金のヒストグラム")
    plt.xlabel("配当金額 (円)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('odds_histogram.png')
    plt.show()
    print("ヒストグラムが 'odds_histogram.png' として保存されました。")

if __name__ == '__main__':
    inspect_model()
    # create_confidence_model()
    # create_confidence_model()

    # inspect_with_confidence_model()
