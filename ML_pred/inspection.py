# inspection.py
import pandas as pd
import lightgbm as lgb
from mymodule import load_b_file, load_k_file, preprocess_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合


def inspect_model():
    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model.txt')
        # gbm = lgb.LGBMClassifier(model_file='boatrace_model.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    # 241001以降のデータで検証
    date_range = ['241001', '241002', '241003', '241004', '241005', 
                  '241006', '241007', '241008']
#================================================================================================
    # # データの読み込み
    # b_file = 'b_data/2409/B240901.TXT'
    # k_file = 'k_data/2409/K240901.TXT'

    # try:
    #     b_data = load_b_file(b_file)
    #     k_data, odds_data = load_k_file(k_file)
    # except FileNotFoundError:
    #     print("ファイルが見つかりません。指定されたディレクトリにファイルが存在するか確認してください。")
    #     return

    # if b_data.empty or k_data.empty:
    #     print("データが正しく読み込めませんでした。")
    #     return
    
    # if b_data.empty or k_data.empty:
    #     print("データが正しく読み込めませんでした。")
    #     return
#================================================================================================

    b_data_list = []
    k_data_list = []
    odds_list1 = []

    for date in date_range:
        try:
            b_file = f'b_data/2410/B{date}.TXT'
            k_file = f'k_data/2410/K{date}.TXT'
            b_data = load_b_file(b_file)
            k_data, odds_list_part = load_k_file(k_file)
            
            if not b_data.empty and not k_data.empty:
                b_data_list.append(b_data)
                k_data_list.append(k_data)
            if not odds_list_part.empty:
                odds_list1.append(odds_list_part)

                
        except FileNotFoundError:
            print(f"ファイルが見つかりません: {date}")
    
    # b_dataとk_dataの結合
    b_data = pd.concat(b_data_list, ignore_index=True)
    k_data = pd.concat(k_data_list, ignore_index=True)
    odds_list = pd.concat(odds_list1, ignore_index=True)

    if b_data.empty or k_data.empty:
        print("データが正しく読み込めませんでした。")
        return
    if odds_list.empty:
        print("データが正しく読み込めませんでした。")

    # データのマージ
    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
    if data.empty:
        print("Merged data is empty after merging B and K data.")
        return
    # _xと_yがついているカラムを探す
    columns_to_check = [col for col in data.columns if '_x' in col]

    for col_x in columns_to_check:
        col_y = col_x.replace('_x', '_y')
        
        # _yカラムが存在するか確認
        if col_y in data.columns:
            # 内容が一致するか確認
            if data[col_x].equals(data[col_y]):
                # 内容が一致している場合、_xの名前に統一して_yを削除
                data[col_x.replace('_x', '')] = data[col_x]  # 統一した名前に
                data.drop([col_x, col_y], axis=1, inplace=True)  # _x, _yを削除
            else:
                # 一致しない場合の処理（エラーメッセージ表示など）
                print(f"Warning: {col_x}と{col_y}の内容が一致しません。手動で確認してください。")

    # 選手名_x と 選手名_y の内容を比較して一致しない場合の処理
    for idx, row in data.iterrows():
        if row['選手名_x'] != row['選手名_y']:
            # 最初の4文字が一致するかを確認
            if row['選手名_x'][:4] == row['選手名_y'][:4]:
                # 選手名_y にそろえて選手名列を作成
                data.at[idx, '選手名'] = row['選手名_y']
            else:
                print(f"Warning: 選手名_x と 選手名_y の最初の4文字が一致しません。手動で確認してください。\n{row[['選手名_x', '選手名_y']]}")
        else:
            # 一致している場合は、どちらかを選んで選手名列に
            data.at[idx, '選手名'] = row['選手名_x']

    # 重複列を削除
    data.drop(['選手名_x', '選手名_y'], axis=1, inplace=True)

    # 結果のデバッグ表示
    print("選手名の統一が完了しました。")
    # data = pd.merge(data, odds_list, on='レースID')  # オッズデータも結合
    
    # print(data.columns)
    print(data)
    print(odds_list)
    print(odds_list.isnull().sum())
    odds_list.info()

    # # データのマージ
    # data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
    # if data.empty:
    #     print("Merged data is empty after merging B and K data.")
    #     return

    # データの前処理
    data = preprocess_data(data)
    if data.empty:
        print("Data is empty after preprocessing.")
        return

    # 特徴量の指定
    features = ['艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                '展示タイム', '天候', '風向', '風量', '波']
    X = data[features]

    # 予測
    y_pred = gbm.predict(X)
    data['prob_0'] = y_pred[:, 0]  # 1着or2着の確率
    data['prob_1'] = y_pred[:, 1]
    data['prob_2'] = y_pred[:, 2]
    
    # y_pred = gbm.fit(X)
    # probs = gbm.predict_proba(X)
    correct_predictions = 0
    total_races = 0
    # 各日にちごとの正答率を格納する辞書

    daily_accuracy = {}
    # 回収金額と賭け金を計算するための変数
    total_bet = 0  # 全体の賭け金（1レースごとに100円かける）
    total_return = 0  # 全体の回収金額
    payout_list = []  # 配当金のリスト（ヒストグラム用）

    # レースごとに予測順位を決定
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

        # 正答率の計算（予測が的中したかを確認）
        if (predicted_first == actual_first and
            predicted_second == actual_second and
            predicted_third == actual_third):
            correct_predictions += 1

            # 三連単オッズ情報の取得
            odds_row = odds_list[odds_list['レースID'] == race_id]

            # デバッグ: オッズ情報が存在するか確認
            if odds_row.empty:
                print(f"オッズ情報が見つかりません: レースID {race_id}")
                payout_list.append(0)
                continue

            # 予測が的中した場合、オッズに基づいて回収金額を計算
            winning_odds = odds_row["三連単オッズ"].values[0]

            # デバッグ: オッズが正常に取得されているか確認
            if pd.isna(winning_odds):
                print(f"オッズがNaNです: レースID {race_id}")
                payout_list.append(0)
                continue
            
            payout = 100 * winning_odds  # 当たった場合は100円×オッズ
            total_return += payout
            payout_list.append(payout)  # 配当金をリストに追加
        else:
            payout_list.append(0)  # 外れた場合は0円

        total_races += 1

        # 三連単予測結果と実際の結果を表示
        print(f"レースID: {race_id}")
        print(f"予測 - 1着: {predicted_first}, 2着: {predicted_second}, 3着: {predicted_third}")
        print(f"実際 - 1着: {actual_first}, 2着: {actual_second}, 3着: {actual_third}")

    # 正答率を表示
    accuracy = correct_predictions / total_races if total_races > 0 else 0
    print(f"\n三連単の正答率: {accuracy * 100:.2f}%")

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
