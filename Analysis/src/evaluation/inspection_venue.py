import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from ..utils.mymodule import load_b_file, load_k_file, preprocess_data
import numpy as np
import seaborn as sns
import os

def inspect_model():
    # モデルが保存されているディレクトリ
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print(f"モデル保存ディレクトリ '{model_dir}' が存在しません。モデルを作成してください。")
        return

    # トレーニング済みのモデルのリストを取得
    model_files = [f for f in os.listdir(model_dir) if f.startswith('first_model_') and f.endswith('.txt')]
    if not model_files:
        print(f"モデル保存ディレクトリ '{model_dir}' にモデルファイルが存在しません。モデルを作成してください。")
        return

    # venue_models のキーを整数に変換
    venue_models = {}
    for f in model_files:
        try:
            # ファイル名から会場名を抽出し、整数に変換
            venue_str = f.split('_')[2].split('.')[0]
            venue_int = int(venue_str)
            model_path = os.path.join(model_dir, f)
            venue_models[venue_int] = lgb.Booster(model_file=model_path)
            print(f"モデルロード成功: 会場 {venue_int} -> {model_path}")
        except (IndexError, ValueError) as e:
            print(f"モデルファイル名の解析に失敗しました: {f}. エラー: {e}")

    if not venue_models:
        print("有効な会場ごとのモデルがロードされていません。モデルファイル名を確認してください。")
        return

    # 評価期間のデータ日付リスト作成
    month_folders = ['2408','2409', '2410']
    date_files = {
        '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(1, 9)],   # 241001 - 241008
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
                print(f"ファイルが見つかりません: {date}")

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

    # カラム名の整合性を確認・修正（ユーザー提供のコードに置き換え）
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

    # 回収率計算のための変数
    total_bet = 0  # 全体の賭け金（1レースごとに100円かける）
    total_return = 0  # 全体の回収金額
    payout_list = []  # 配当金のリスト（ヒストグラム用）
    betting_races = []  # 投資対象となったレースのIDリスト

    # 会場ごとの賭け金と回収金を追跡する辞書
    venue_bet_return = {}  # {venue: {'bet': int, 'return': float}}

    # デバッグ用: data['会場'] のユニークな値を表示
    print("デバッグ: データ内の会場のユニークな値:")
    print(data['会場'].unique())
    print("デバッグ: data['会場'] のデータ型:", data['会場'].dtype)
    print("デバッグ: venue_models.keys():", venue_models.keys())

    # data['会場'] の型を venue_models のキーと合わせる
    # venue_models のキーは int 型なので、data['会場'] を int 型に変換
    data['会場'] = data['会場'].astype(int)
    correct_predictions = 0
    total_races = 0
    for venue in venue_models.keys():
        print(f"\n会場: {venue} のモデルを適用中...")
        venue_data = data[data['会場'] == venue]

        # デバッグ用: venue_data が空の場合の対処
        if venue_data.empty:
            print(f"会場 {venue} のデータが存在しません。スキップします。")
            continue

        # デバッグ用: venue_data 内のレースの日付を表示
        race_dates = pd.to_datetime(venue_data['レースID'].astype(str).str[:8], format='%Y%m%d').unique()
        race_dates = [date.strftime('%Y-%m-%d') for date in race_dates]
        print(f"会場 {venue} で分析しているレースの日付: {race_dates}")

        model = venue_models[venue]

        # 特徴量とターゲットの準備
        X_venue = venue_data[features]

        y_pred = model.predict(X_venue)

        # SettingWithCopyWarningを避けるためにコピーを作成
        venue_data = venue_data.copy()
        venue_data['prob_0'] = y_pred[:, 0]  # 1着or2着の確率
        venue_data['prob_1'] = y_pred[:, 1]
        venue_data['prob_2'] = y_pred[:, 2]

        # レースごとに予測順位を決定
        race_ids = venue_data['レースID'].unique()

        #単賭け
        # for race_id in race_ids:
        #     race_data = venue_data[venue_data['レースID'] == race_id]
        #     race_data = race_data.sort_values('prob_0', ascending=False)

        #     # レースデータに少なくとも3艇が含まれていることを確認
        #     if len(race_data) < 3:
        #         print(f"レースID {race_id} のデータが不足しています。スキップします。")
        #         continue

        #     # 予測された1着、2着、3着を決定
        #     predicted_first = race_data.iloc[0]['艇番']
        #     predicted_second = race_data.iloc[1]['艇番']
        #     predicted_third = race_data.iloc[2]['艇番']

        #     # 実際の1着、2着、3着の選手（'着'カラムがそれぞれ1, 2, 3の選手を取得）
        #     actual_first = race_data[race_data['着'] == 1]['艇番'].values[0]
        #     actual_second = race_data[race_data['着'] == 2]['艇番'].values[0]
        #     actual_third = race_data[race_data['着'] == 3]['艇番'].values[0]

        #     # if len(actual_third) == 0:
        #     #     print(f"レースID {race_id} に実際の1着情報がありません。スキップします。")
        #     #     payout_list.append(0)
        #     #     continue
        #     # actual_first = actual_first[0]

        #     # 賭け金を追加 (1レースあたり100円)
        #     total_bet += 100
        #     betting_races.append(race_id)

        #     # 会場ごとの賭け金を更新
        #     if venue not in venue_bet_return:
        #         venue_bet_return[venue] = {'bet': 0, 'return': 0}
        #     venue_bet_return[venue]['bet'] += 100

        #     # #3連単の場合
        #     # 正答率の計算（予測が的中したかを確認）
        #     if (predicted_first == actual_first and
        #         predicted_second == actual_second and
        #         predicted_third == actual_third):
        #         correct_predictions += 1

        #         # 三連単オッズ情報の取得
        #         odds_row = odds_list[odds_list['レースID'] == race_id]

        #         # デバッグ: オッズ情報が存在するか確認
        #         if odds_row.empty:
        #             print(f"オッズ情報が見つかりません: レースID {race_id}")
        #             payout_list.append(0)
        #             continue

        #         # 予測が的中した場合、オッズに基づいて回収金額を計算
        #         winning_odds = odds_row["三連単オッズ"].values[0]

        #         # デバッグ: オッズが正常に取得されているか確認
        #         if pd.isna(winning_odds):
        #             print(f"オッズがNaNです: レースID {race_id}")
        #             payout_list.append(0)
        #             continue

        #         payout = 100 * winning_odds  # 当たった場合は100円×オッズ
        #         total_return += payout
        #         payout_list.append(payout)  # 配当金をリストに追加
            
        #         # 会場ごとの回収金を更新
        #         venue_bet_return[venue]['return'] += payout

        #     else:
        #         payout_list.append(0)  # 外れた場合は0円

        #     total_races += 1


        #     # #単勝の場合
        #     # # 予測が的中したかを確認
        #     # if predicted_first == actual_first and race_data.iloc[0]['prob_0'] > 0.3:
        #     #     # オッズ情報の取得
        #     #     odds_row = odds_list[odds_list['レースID'] == race_id]
        #     #     if odds_row.empty:
        #     #         print(f"オッズ情報が見つかりません: レースID {race_id}")
        #     #         payout_list.append(0)
        #     #         continue

        #     #     if '単勝オッズ' not in odds_row.columns:
        #     #         print(f"オッズデータに '単勝オッズ' カラムが存在しません: レースID {race_id}")
        #     #         payout_list.append(0)
        #     #         continue
                
        #     #     winning_odds = odds_row["単勝オッズ"].values[0]

        #     #     # オッズがNaNでないか確認
        #     #     if pd.isna(winning_odds):
        #     #         print(f"単勝オッズがNaNです: レースID {race_id}")
        #     #         payout_list.append(0)
        #     #         continue

        #     #     # 回収金額の計算
        #     #     payout = 100 * winning_odds
        #     #     total_return += payout
        #     #     payout_list.append(payout)

        #     #     # 会場ごとの回収金を更新
        #     #     venue_bet_return[venue]['return'] += payout
        #     # else:
        #     #     payout_list.append(0)  # 外れた場合は0円


        #期待値検討
        for race_id in race_ids:
            race_data = venue_data[venue_data['レースID'] == race_id]

            #期待値検証
            race_data = venue_data[venue_data['レースID'] == race_id]
            # 各艇の確率を取得
            boats = race_data['艇番'].values
            prob_0 = race_data['prob_0'].values  # 1着の確率
            prob_1 = race_data['prob_1'].values  # 2着の確率
            prob_2 = race_data['prob_2'].values  # 3着の確率

            # ベットするタイプとその期待値を格納する辞書
            ev_dict = {}

            # 賭け金を追加 (1レースあたり100円)
            total_bet += 100
            betting_races.append(race_id)

            # 三連単 (Exact Triple)
            top3_indices = np.argsort(prob_0)[-3:][::-1]  # prob_0が高い順にトップ3
            predicted_triple = boats[top3_indices]
            if len(top3_indices) >= 3:
                p_triple = prob_0[top3_indices[0]] * prob_1[top3_indices[1]] * prob_2[top3_indices[2]]
            else:
                p_triple = 0
            # 三連単オッズの取得
            odds_row = odds_list[odds_list['レースID'] == race_id]
            odds_triple = np.nan
            if not odds_row.empty and '三連単オッズ' in odds_row.columns:
                odds_triple = odds_row["三連単オッズ"].values[0]
                if pd.isna(odds_triple):
                    print(f"レースID {race_id}: 三連単オッズが NaN です。")
                    odds_triple = np.nan
            if not pd.isna(odds_triple):
                ev_triple = p_triple * 100 * odds_triple
                ev_dict['三連単'] = {'EV': ev_triple, 'prediction': predicted_triple, 'odds': odds_triple}
            else:
                ev_dict['三連単'] = {'EV': 0, 'prediction': None, 'odds': None}

            # 会場ごとの賭け金を更新
            if venue not in venue_bet_return:
                venue_bet_return[venue] = {'bet': 0, 'return': 0}
            venue_bet_return[venue]['bet'] += 100

            # 三連複 (Trio)
            p_trio = 0
            from itertools import permutations
            if len(boats) >= 3:
                for perm in permutations(predicted_triple):
                    idx_first = np.where(boats == perm[0])[0][0]
                    idx_second = np.where(boats == perm[1])[0][0]
                    idx_third = np.where(boats == perm[2])[0][0]
                    p = prob_0[idx_first] * prob_1[idx_second] * prob_2[idx_third]
                    p_trio += p
            else:
                p_trio = 0
            # 三連複オッズの取得
            odds_trio = np.nan
            if not odds_row.empty and '三連複オッズ' in odds_row.columns:
                odds_trio = odds_row["三連複オッズ"].values[0]
                if pd.isna(odds_trio):
                    print(f"レースID {race_id}: 三連複オッズが NaN です。")
                    odds_trio = np.nan
            if not pd.isna(odds_trio):
                ev_trio = p_trio * 100 * odds_trio
                ev_dict['三連複'] = {'EV': ev_trio, 'prediction': predicted_triple, 'odds': odds_trio}
            else:
                ev_dict['三連複'] = {'EV': 0, 'prediction': None, 'odds': None}

            # 二連単 (Exact Double)
            top2_indices = np.argsort(prob_0)[-2:][::-1]
            predicted_double = boats[top2_indices]
            p_double = prob_0[top2_indices[0]] * prob_1[top2_indices[1]]
            # 二連単オッズの取得
            odds_double = np.nan
            if not odds_row.empty and '二連単オッズ' in odds_row.columns:
                odds_double = odds_row["二連単オッズ"].values[0]
                if pd.isna(odds_double):
                    print(f"レースID {race_id}: 二連単オッズが NaN です。")
                    odds_double = np.nan
            if not pd.isna(odds_double):
                ev_double = p_double * 100 * odds_double
                ev_dict['二連単'] = {'EV': ev_double, 'prediction': predicted_double, 'odds': odds_double}
            else:
                ev_dict['二連単'] = {'EV': 0, 'prediction': None, 'odds': None}

            # 二連複 (Double)
            p_double_combo = prob_0[top2_indices[0]] * prob_1[top2_indices[1]] + prob_0[top2_indices[1]] * prob_1[top2_indices[0]]
            # 二連複オッズの取得
            odds_double_combo = np.nan
            if not odds_row.empty and '二連複オッズ' in odds_row.columns:
                odds_double_combo = odds_row["二連複オッズ"].values[0]
                if pd.isna(odds_double_combo):
                    print(f"レースID {race_id}: 二連複オッズが NaN です。")
                    odds_double_combo = np.nan
            if not pd.isna(odds_double_combo):
                ev_double_combo = p_double_combo * 100 * odds_double_combo
                ev_dict['二連複'] = {'EV': ev_double_combo, 'prediction': predicted_double, 'odds': odds_double_combo}
            else:
                ev_dict['二連複'] = {'EV': 0, 'prediction': None, 'odds': None}

            # 単勝 (Win)
            odds_win = np.nan
            if not odds_row.empty and '単勝オッズ' in odds_row.columns:
                odds_win = odds_row["単勝オッズ"].values[0]
                if pd.isna(odds_win):
                    print(f"レースID {race_id}: 単勝オッズが NaN です。")
                    odds_win = np.nan
            if not pd.isna(odds_win):
                win_idx = np.argmax(prob_0)
                predicted_win = boats[win_idx]
                p_win = prob_0[win_idx]
                ev_win = p_win * 100 * odds_win
                ev_dict['単勝'] = {'EV': ev_win, 'prediction': predicted_win, 'odds': odds_win}
            else:
                ev_dict['単勝'] = {'EV': 0, 'prediction': None, 'odds': None}

            # 各投資タイプの期待値を表示（デバッグ）
            print(f"レースID {race_id} の期待値:")
            for bet_type, info in ev_dict.items():
                print(f"  {bet_type}: EV = {info['EV']:.2f}")

            # 期待値が最も高い投資タイプを選択
            best_bet_type = max(ev_dict.items(), key=lambda x: x[1]['EV'])[0]
            best_bet_info = ev_dict[best_bet_type]

            if best_bet_info['EV'] <= 0 or best_bet_info['prediction'] is None:
                # 期待値が0以下、または予測が存在しない場合はベットしない
                print(f"レースID {race_id} ではベットを行いません。")
                payout_list.append(0)
                continue

            # ベットタイプと予測を表示（デバッグ）
            print(f"レースID {race_id} では {best_bet_type} にベットします。予測: {best_bet_info['prediction']} オッズ: {best_bet_info['odds']}")

            # オッズが NaN の場合はベットをスキップ
            if pd.isna(best_bet_info['odds']):
                print(f"レースID {race_id}: {best_bet_type} のオッズが NaN です。ベットをスキップします。")
                payout_list.append(0)
                continue

            # 実際の結果と照合してベットが的中したか確認
            if best_bet_type == '三連単':
                # 三連単の的中確認
                actual_triple = race_data[race_data['着'] <=3].sort_values('着')
                if len(actual_triple) <3:
                    print(f"レースID {race_id} の実際の上位3着データが不足しています。スキップします。")
                    payout_list.append(0)
                    continue
                actual_triple_boats = actual_triple['艇番'].values
                if list(best_bet_info['prediction']) == list(actual_triple_boats):
                    # 的中
                    payout = 100 * best_bet_info['odds']
                    total_return += payout
                    payout_list.append(payout)
                    print(f"三連単が的中しました。配当金: {payout}円")
                    # 会場ごとの回収金を更新
                    venue_bet_return[venue]['return'] += payout
                else:
                    # 外れ
                    payout_list.append(0)
                    print(f"三連単が外れました。")

            elif best_bet_type == '三連複':
                # 三連複の的中確認
                actual_triple = race_data[race_data['着'] <=3].sort_values('着')
                if len(actual_triple) <3:
                    print(f"レースID {race_id} の実際の上位3着データが不足しています。スキップします。")
                    payout_list.append(0)
                    continue
                actual_triple_boats = actual_triple['艇番'].values
                # 比較は順不同
                if set(best_bet_info['prediction']) == set(actual_triple_boats):
                    # 的中
                    payout = 100 * best_bet_info['odds']
                    total_return += payout
                    payout_list.append(payout)
                    print(f"三連複が的中しました。配当金: {payout}円")
                    # 会場ごとの回収金を更新
                    venue_bet_return[venue]['return'] += payout
                else:
                    # 外れ
                    payout_list.append(0)
                    print(f"三連複が外れました。")

            elif best_bet_type == '二連単':
                # 二連単の的中確認
                actual_first = race_data[race_data['着'] == 1]['艇番'].values
                actual_second = race_data[race_data['着'] == 2]['艇番'].values
                if len(actual_first) == 0 or len(actual_second) == 0:
                    print(f"レースID {race_id} の実際の1着または2着データが不足しています。スキップします。")
                    payout_list.append(0)
                    continue
                actual_first = actual_first[0]
                actual_second = actual_second[0]
                if (best_bet_info['prediction'][0] == actual_first and
                    best_bet_info['prediction'][1] == actual_second):
                    # 的中
                    payout = 100 * best_bet_info['odds']
                    total_return += payout
                    payout_list.append(payout)
                    print(f"二連単が的中しました。配当金: {payout}円")
                    # 会場ごとの回収金を更新
                    venue_bet_return[venue]['return'] += payout
                else:
                    # 外れ
                    payout_list.append(0)
                    print(f"二連単が外れました。")

            elif best_bet_type == '二連複':
                # 二連複の的中確認
                actual_first = race_data[race_data['着'] == 1]['艇番'].values
                actual_second = race_data[race_data['着'] == 2]['艇番'].values
                if len(actual_first) == 0 or len(actual_second) == 0:
                    print(f"レースID {race_id} の実際の1着または2着データが不足しています。スキップします。")
                    payout_list.append(0)
                    continue
                actual_first = actual_first[0]
                actual_second = actual_second[0]
                if ((best_bet_info['prediction'][0] == actual_first and best_bet_info['prediction'][1] == actual_second) or
                    (best_bet_info['prediction'][0] == actual_second and best_bet_info['prediction'][1] == actual_first)):
                    # 的中
                    payout = 100 * best_bet_info['odds']
                    total_return += payout
                    payout_list.append(payout)
                    print(f"二連複が的中しました。配当金: {payout}円")
                    # 会場ごとの回収金を更新
                    venue_bet_return[venue]['return'] += payout
                else:
                    # 外れ
                    payout_list.append(0)
                    print(f"二連複が外れました。")

            elif best_bet_type == '単勝':
                # 単勝の的中確認
                actual_first = race_data[race_data['着'] == 1]['艇番'].values
                if len(actual_first) == 0:
                    print(f"レースID {race_id} に実際の1着情報がありません。スキップします。")
                    payout_list.append(0)
                    continue
                actual_first = actual_first[0]
                if best_bet_info['prediction'] == actual_first:
                    # 的中
                    payout = 100 * best_bet_info['odds']
                    total_return += payout
                    payout_list.append(payout)
                    print(f"単勝が的中しました。配当金: {payout}円")
                    # 会場ごとの回収金を更新
                    venue_bet_return[venue]['return'] += payout
                else:
                    # 外れ
                    payout_list.append(0)
                    print(f"単勝が外れました。")


    # 的中率の計算
    correct_predictions = 0
    total_bets = len(payout_list)
    correct_predictions = sum(payout > 0 for payout in payout_list)
    hit_rate = (correct_predictions / total_bets) * 100 if total_bets > 0 else 0
    print(f"\n的中率: {hit_rate:.2f}%")

    # 回収率の計算
    return_rate = (total_return / total_bet) * 100 if total_bet > 0 else 0
    print(f"投資戦略による総賭け金: {total_bet}円")
    print(f"投資戦略による総回収金額: {total_return}円")
    print(f"投資戦略による回収率: {return_rate:.2f}%")

    # 会場ごとの回収率の計算と出力
    print("\n会場ごとの回収率:")
    for venue, stats in venue_bet_return.items():
        bet = stats['bet']
        ret = stats['return']
        rate = (ret / bet) * 100 if bet > 0 else 0
        print(f"会場 {venue}: 賭け金 = {bet}円, 回収金 = {ret:.2f}円, 回収率 = {rate:.2f}%")

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
