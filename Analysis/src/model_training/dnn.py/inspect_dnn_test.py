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

    # 使用する特徴量の指定
    features = ['艇番', '年齢', '体重', '級別', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーターNo', 'モーター2連率',
                'ボートNo', 'ボート2連率']
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['艇番', '級別', 'モーターNo', 'ボートNo']
    X_categorical = encoder.transform(X[categorical_features])

    # 数値データの抽出と標準化
    numeric_features = ['年齢', '体重', '全国勝率', '全国2連率',
                        '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率']
    X_numeric = X[numeric_features]
    X_numeric_scaled = scaler.transform(X_numeric)

    # 特徴量の結合
    X_processed = np.hstack([X_categorical, X_numeric_scaled])

    # 特徴量名の設定
    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    # 予測の実行
    y_pred = model.predict(X_processed)

    # 予測確率をデータフレームに追加
    data['win_prob'] = y_pred.flatten()

    # レースごとに上位の選手を選択
    race_ids = data['レースID'].unique()
    total_bet = 0
    total_return = 0
    payout_list = []
    betting_races = []
    venue_bet_return = {}

    for race_id in race_ids:
        race_data = data[data['レースID'] == race_id]
        if race_data.empty:
            continue

        # 会場名を取得
        venue = race_data['会場'].iloc[0]

        # 賭け金を追加 (1レースあたり100円)
        total_bet += 100
        betting_races.append(race_id)

        # 会場ごとの賭け金を更新
        if venue not in venue_bet_return:
            venue_bet_return[venue] = {'bet': 0, 'return': 0}
        venue_bet_return[venue]['bet'] += 100

        # 勝率が最も高い選手を選択
        top_player = race_data.loc[race_data['win_prob'].idxmax()]
        predicted_boat = top_player['艇番']
        predicted_prob = top_player['win_prob']

        # 単勝オッズを取得
        odds_row = odds_list[(odds_list['レースID'] == race_id)]
        if odds_row.empty or pd.isna(odds_row['単勝オッズ'].values[0]):
            print(f"レースID {race_id}: 単勝オッズが取得できません。ベットをスキップします。")
            payout_list.append(0)
            continue

        odds_win = odds_row['単勝オッズ'].values[0]

        # 期待値を計算
        ev_win = predicted_prob * 100 * odds_win
        if ev_win <= 0:
            print(f"レースID {race_id}: 期待値が0以下のため、ベットをスキップします。")
            payout_list.append(0)
            continue

        print(f"レースID {race_id}: ボート {predicted_boat} にベットします。オッズ: {odds_win}, 期待値: {ev_win:.2f}")

        # 実際の結果を取得
        actual_first = race_data[race_data['着'] == 1]['艇番'].values
        if len(actual_first) == 0:
            print(f"レースID {race_id} に実際の1着情報がありません。スキップします。")
            payout_list.append(0)
            continue
        actual_first = actual_first[0]

        if predicted_boat == actual_first:
            # 的中
            payout = 100 * odds_win
            total_return += payout
            payout_list.append(payout)
            print(f"単勝が的中しました。配当金: {payout}円")
            # 会場ごとの回収金を更新
            venue_bet_return[venue]['return'] += payout
        else:
            # 外れ
            payout_list.append(0)
            print(f"単勝が外れました。")

    # 回収率の計算
    return_rate = (total_return / total_bet) * 100 if total_bet > 0 else 0
    print(f"\n総ベット額: {total_bet}円, 総リターン: {total_return}円, 回収率: {return_rate:.2f}%")

    # 会場ごとの回収率を表示
    print("\n会場ごとの回収率:")
    for venue, br in venue_bet_return.items():
        venue_return_rate = (br['return'] / br['bet']) * 100 if br['bet'] > 0 else 0
        print(f"{venue}: ベット額 = {br['bet']}円, リターン = {br['return']}円, 回収率 = {venue_return_rate:.2f}%")

if __name__ == '__main__':
    inspect_model()
