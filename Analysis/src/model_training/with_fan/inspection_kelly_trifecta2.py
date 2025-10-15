# model_1min.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, load_before_data, merge_before_data
from get_data import load_fan_data
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import itertools
import logging
import seaborn as sns
import re  # 正規表現を使用するために追加

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

def load_processed_data():
    # Load the processed dataframe directly from saved CSV file
    file_path = 'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/odds_dataframe/odds_data.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

def inspect_model():
    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_trio.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_trio.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2410', '2411']
    date_files = {
        '2410': [f'2410{day:02d}' for day in range(1, 32)],
        '2411': [f'2411{day:02d}' for day in range(1, 6)],
    }

    # データを結合
    data_list = []
    odds_list1 = []

    for month in month_folders:
        for date in date_files[month]:
            try:
                b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'
                k_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_data/{month}/K{date}.TXT'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'

                b_data = load_b_file(b_file)
                k_data, odds_list_part = load_k_file(k_file)
                before_data = load_before_data(before_file)

                if not b_data.empty and not before_data.empty:
                    # データのマージ
                    data = b_data
                    before_data = remove_common_columns(
                        data, before_data, on_columns=['選手登番', 'レースID', '艇番']
                    )
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
    odds_data = pd.concat(odds_list1, ignore_index=True)

    # ファンデータの読み込みとマージ
    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    # 必要な列を指定してマージ
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別',
                      '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(
        data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left'
    )

    data = preprocess_data(data)
    print(data[data['レースID'] == '202411051002'])
    # 特徴量の指定
    features = [
        '会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
        '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
        'ET', 'tilt', 'EST', 'ESC',
        'weather', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
        '前期能力指数', '今期能力指数', '平均スタートタイミング', '性別',
        '勝率', '複勝率', '優勝回数', '優出回数'
    ]
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = [
        '会場', 'weather', 'wind_d', 'ESC', '艇番', '性別', '支部', '級別'
    ]
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # 数値データの選択
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]

    # 特徴量の結合（インデックスをリセットしない）
    X_processed = pd.concat([X_numeric, X_categorical], axis=1)

    # 学習時の特徴量と順序を合わせる
    for col in feature_names:
        if col not in X_processed.columns:
            X_processed[col] = 0

    X_processed = X_processed[feature_names]

    # 予測
    y_pred = gbm.predict(X_processed)

    data['prob_0'] = y_pred[:, 0]  # 1着の確率

    # 'prob_0' の分布をプロット
    plt.figure(figsize=(8, 6))
    plt.hist(
        data['prob_0'], bins=50, color='blue', edgecolor='black', alpha=0.7
    )
    plt.title("予測確率 'prob_0' の分布")
    plt.xlabel("予測確率 (prob_0)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('prob_0_distribution_trifecta_single_test.png')
    plt.show()
    print("プロットが 'prob_0_distribution_trifecta_single_test.png' として保存されました。")

    # オッズデータの読み込み
    base_dir_trifecta = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\trifecta_odds'
    base_dir_trio = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\trio_odds'  # 三連複オッズデータのディレクトリ
    start_date = datetime.strptime('2024-10-01', '%Y-%m-%d')
    end_date = datetime.strptime('2024-11-05', '%Y-%m-%d')
    total_capital = 100000  # 総資金（円）
    unit = 100  # 賭け金の単位（円）
    prob_threshold = 0  # 賭ける確率の閾値

    # オッズデータの読み込み
    trifecta_odds_data_df = load_odds_data(base_dir_trifecta, start_date, end_date, bet_type='trifecta')
    trio_odds_data_df = load_odds_data(base_dir_trio, start_date, end_date, bet_type='trio')  # 三連複オッズデータの読み込み

    if trifecta_odds_data_df.empty or trio_odds_data_df.empty:
        print("No odds data available. Exiting.")
        return

    # 各艇の確率データ
    data_boats = data[['レースID', '艇番', 'prob_0', '会場']].copy()

    # EV計算の閾値のリスト
    ev_threshold_values = [10,15,20,40,60,100]

    # 結果を保存するリスト
    cumulative_returns = {}
    summary_results = []

    for ev_threshold in ev_threshold_values:
        print(f"\n=== EV計算の閾値: {ev_threshold} ===")
        # 賭け金と回収金額の計算
        bets, total_investment, total_payout, return_rate = calculate_bets(
            trifecta_odds_data_df,
            trio_odds_data_df,
            data_boats,
            odds_data,  # odds_data を追加
            total_capital=total_capital,
            unit=unit,
            prob_threshold=prob_threshold,
            ev_threshold=ev_threshold  # EV計算の閾値を追加
        )

        # 資金の推移を計算
        capital_evolution = calculate_capital_evolution(bets)

        # 結果を保存
        cumulative_returns[ev_threshold] = capital_evolution
        summary_results.append({
            'EV_Threshold': ev_threshold,
            'Total Investment (yen)': total_investment,
            'Total Payout (yen)': total_payout,
            'Return Rate (%)': return_rate
        })

    # 結果の可視化
    plot_cumulative_returns(cumulative_returns)

    # 結果の比較表を作成
    summary_df = pd.DataFrame(summary_results)
    print("\n結果の比較:")
    print(summary_df)

    # 結果をCSVに保存
    summary_df.to_csv('bet_summary_comparison_trifecta2_allvenue_test.csv', index=False)
    print("結果の比較を 'bet_summary_comparison_trifecta2_allvenue_test.csv' として保存しました。")

    # ------------- 追加部分開始 -------------
    # trifecta_bets を bets からフィルタリングして定義
    trifecta_bets = bets[bets['Bet_Type'] == 'Trifecta'].copy()

    # '会場' 情報を追加するために 'data' とマージ
    trifecta_bets = trifecta_bets.merge(data[['レースID', '会場']], on='レースID', how='left')

    # 日付ごとの収益を計算
    revenue_per_date = trifecta_bets.groupby('Date').apply(
        lambda df: df['Payout'].sum() - df['Bet_Amount'].sum()
    ).reset_index(name='Net_Revenue')

    # 日付ごとの収益をプロット（点なしの折れ線グラフ）
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Date', y='Net_Revenue', data=revenue_per_date, color='green')
    plt.title('日付ごとの収益')
    plt.xlabel('日付')
    plt.ylabel('収益 (円)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('revenue_per_date_trifecta2_allvenue_test.png')
    plt.show()
    print("日付ごとの収益を 'revenue_per_date_trifecta2_allvenue_test.png' として保存しました。")

    # 1日当たりの投資金額を計算
    investment_per_date = trifecta_bets.groupby('Date')['Bet_Amount'].sum().reset_index()

    # 統計量の計算
    min_investment = investment_per_date['Bet_Amount'].min()
    max_investment = investment_per_date['Bet_Amount'].max()
    average_investment = investment_per_date['Bet_Amount'].mean()
    median_investment = investment_per_date['Bet_Amount'].median()

    # 統計結果の表示
    print("\n1日当たりの投資金額の統計量:")
    print(f"最小値: {min_investment} 円")
    print(f"最大値: {max_investment} 円")
    print(f"平均値: {average_investment:.2f} 円")
    print(f"中央値: {median_investment} 円")

    # 1日当たりの投資金額のヒストグラムをプロット
    plt.figure(figsize=(10, 6))
    sns.histplot(investment_per_date['Bet_Amount'], bins=30, kde=True, color='purple')
    plt.title('1日当たりの投資金額のヒストグラム')
    plt.xlabel('投資金額 (円)')
    plt.ylabel('頻度')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('investment_per_date_histogram_test.png')
    plt.show()
    print("1日当たりの投資金額のヒストグラムを 'investment_per_date_histogram_test.png' として保存しました。")

    # 会場ごとの回収率を計算
    venue_summary = trifecta_bets.groupby('会場').agg(
        Total_Payout=('Payout', 'sum'),
        Total_Bet=('Bet_Amount', 'sum')
    ).reset_index()
    venue_summary['Return_Rate (%)'] = (venue_summary['Total_Payout'] / venue_summary['Total_Bet']) * 100
    venue_summary = venue_summary[['会場', 'Return_Rate (%)']]

    # 会場ごとの回収率をCSVに保存
    venue_summary.to_csv('return_rate_per_venue_trifecta2_allvenue_test.csv', index=False)
    print("会場ごとの回収率を 'return_rate_per_venue_trifecta2_allvenue_test.csv' として保存しました。")
    # ------------- 追加部分終了 -------------

def process_race(race_id, race_trifecta_odds, race_trio_odds, data_boats, odds_data, total_capital=100000, unit=100, prob_threshold=0.0, ev_threshold=6):
    """
    各レースごとに三連単と三連複のどちらに投票するかを判定し、回収金額を計算します。
    """
    # 該当レースの艇の確率データを取得
    boats_data = data_boats[data_boats['レースID'] == race_id]
    if boats_data.empty:
        # データがない場合
        columns = ['レースID', 'Bet_Type', 'Combination', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout']
        return pd.DataFrame(columns=columns)

    boats_probs = boats_data.set_index('艇番')['prob_0'].to_dict()

    # 確率が高い順に上位4艇を選択
    top_boats = boats_data.sort_values(by='prob_0', ascending=False)['艇番'].head(4).tolist()

    if len(top_boats) < 3:
        # 上位4艇が3艇未満の場合
        columns = ['レースID', 'Bet_Type', 'Combination', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout']
        return pd.DataFrame(columns=columns)

    # 3艇の組み合わせ (4C3 = 4 combinations)
    combinations_3boats = list(itertools.combinations(top_boats, 3))

    # 該当レースの実際の三連単結果を取得
    race_odds_result = odds_data[odds_data['レースID'] == race_id]
    if not race_odds_result.empty and '三連単結果' in race_odds_result.columns:
        trifecta_result = race_odds_result.iloc[0]['三連単結果']
        if isinstance(trifecta_result, str) and re.match(r'^\d-\d-\d$', trifecta_result):
            actual_combo_trifecta = tuple(map(int, trifecta_result.split('-')))
            actual_combo_trio = tuple(sorted(actual_combo_trifecta))
        else:
            actual_combo_trifecta = ()
            actual_combo_trio = ()
    else:
        actual_combo_trifecta = ()
        actual_combo_trio = ()

    bets_list = []

    for combo in combinations_3boats:
        # 6通りの三連単 (3! permutations)
        trifecta_permutations = list(itertools.permutations(combo, 3))

        # DataFrame化
        trifecta_df = pd.DataFrame(trifecta_permutations, columns=['Boat1', 'Boat2', 'Boat3'])

        # Merge with race_trifecta_odds to get odds
        merged_trifecta = pd.merge(trifecta_df, race_trifecta_odds, on=['Boat1', 'Boat2', 'Boat3'], how='left')
        merged_trifecta = merged_trifecta.dropna(subset=['Odds'])

        if merged_trifecta.empty:
            continue  # オッズがない場合はスキップ

        # Calculate probability for each trifecta
        merged_trifecta['Probability'] = merged_trifecta.apply(lambda row: calculate_trifecta_probability(
            boats_probs.get(row['Boat1'], 0),
            boats_probs.get(row['Boat2'], 0),
            boats_probs.get(row['Boat3'], 0)
        ), axis=1)

        # Calculate EV for each trifecta
        merged_trifecta['EV'] = merged_trifecta['Probability'] * merged_trifecta['Odds']

        # Check if all EVs exceed the threshold
        if (merged_trifecta['EV'] >= ev_threshold).all():
            # 合成オッズを計算
            inverted_odds = 1 / merged_trifecta['Odds']
            composite_trifecta_odds = 1 / inverted_odds.sum()

            # Get trio odds for this combination
            trio_combo = sorted(combo)
            trio_odds_row = race_trio_odds[
                ((race_trio_odds['Boat1'] == trio_combo[0]) & (race_trio_odds['Boat2'] == trio_combo[1]) & (race_trio_odds['Boat3'] == trio_combo[2])) |
                ((race_trio_odds['Boat1'] == trio_combo[0]) & (race_trio_odds['Boat2'] == trio_combo[2]) & (race_trio_odds['Boat3'] == trio_combo[1])) |
                ((race_trio_odds['Boat1'] == trio_combo[1]) & (race_trio_odds['Boat2'] == trio_combo[0]) & (race_trio_odds['Boat3'] == trio_combo[2])) |
                ((race_trio_odds['Boat1'] == trio_combo[1]) & (race_trio_odds['Boat2'] == trio_combo[2]) & (race_trio_odds['Boat3'] == trio_combo[0])) |
                ((race_trio_odds['Boat1'] == trio_combo[2]) & (race_trio_odds['Boat2'] == trio_combo[0]) & (race_trio_odds['Boat3'] == trio_combo[1])) |
                ((race_trio_odds['Boat1'] == trio_combo[2]) & (race_trio_odds['Boat2'] == trio_combo[1]) & (race_trio_odds['Boat3'] == trio_combo[0]))
            ]

            if not trio_odds_row.empty:
                trio_odds = trio_odds_row.iloc[0]['Odds']
            else:
                # Trio odds not available, skip this combination
                continue

            # Decide which bet to place based on higher odds
            if composite_trifecta_odds > trio_odds:
                # Bet on trifecta permutations
                merged_trifecta['Bet_Type'] = 'Trifecta'
                merged_trifecta['Combination'] = merged_trifecta.apply(lambda row: (row['Boat1'], row['Boat2'], row['Boat3']), axis=1)
                merged_trifecta['Bet_Amount'] = unit
                # Calculate payout
                merged_trifecta['Payout'] = merged_trifecta.apply(
                    lambda row: row['Bet_Amount'] * row['Odds'] if (row['Boat1'], row['Boat2'], row['Boat3']) == actual_combo_trifecta else 0,
                    axis=1
                )

                bets_list.append(merged_trifecta[['Bet_Type', 'Combination', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout']])
            else:
                # Bet on trio
                bet = {
                    'Bet_Type': 'Trio',
                    'Combination': tuple(trio_combo),
                    'Probability': merged_trifecta['Probability'].sum(),  # Sum of probabilities
                    'Odds': trio_odds,
                    'EV': trio_odds * merged_trifecta['Probability'].sum(),
                    'Bet_Amount': unit * 6,  # Since trifecta bets would have been 6 units
                    'Payout': 0  # To be calculated
                }
                # Determine if the actual result matches the trio combination
                if set(actual_combo_trio) == set(trio_combo):
                    bet['Payout'] = bet['Bet_Amount'] * bet['Odds'] / (unit * 6)  # Adjust payout

                bets_list.append(pd.DataFrame([bet]))

    if bets_list:
        bets_df = pd.concat(bets_list, ignore_index=True)
        bets_df['レースID'] = race_id
        return bets_df[['レースID', 'Bet_Type', 'Combination', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout']]
    else:
        columns = ['レースID', 'Bet_Type', 'Combination', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout']
        return pd.DataFrame(columns=columns)

def calculate_bets(trifecta_odds_df, trio_odds_df, data_boats, odds_data, total_capital=100000, unit=100, prob_threshold=0.0, ev_threshold=6):
    """
    全レースに対して賭け金額を決定し、回収率や資金の推移を計算します。
    """
    race_ids = trifecta_odds_df['レースID'].unique()
    all_bets = []

    # オッズデータを 'レースID' でグループ化
    grouped_trifecta_odds = trifecta_odds_df.groupby('レースID')
    grouped_trio_odds = trio_odds_df.groupby('レースID')

    # race_odds_list をリストとしてまとめる
    race_odds_list = []
    total_groups = len(grouped_trifecta_odds)
    with tqdm(total=total_groups, desc="Grouping Races") as pbar_group:
        for race_id, race_trifecta_odds in grouped_trifecta_odds:
            race_trio_odds = grouped_trio_odds.get_group(race_id) if race_id in grouped_trio_odds.groups else pd.DataFrame()
            race_odds_list.append((race_id, race_trifecta_odds, race_trio_odds))
            pbar_group.update(1)

    # ThreadPoolExecutor を使用して並列処理
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 並列で処理を実行
        futures = {
            executor.submit(
                process_race, race_id, race_trifecta_odds, race_trio_odds, data_boats, odds_data, total_capital, unit, prob_threshold, ev_threshold
            ): race_id for race_id, race_trifecta_odds, race_trio_odds in race_odds_list
        }

        # プログレスバーの設定
        with tqdm(total=len(futures), desc="Processing Races") as pbar_races:
            for future in as_completed(futures):
                race_id = futures[future]
                try:
                    bet = future.result()
                    if not bet.empty:
                        all_bets.append(bet)
                except Exception as e:
                    logging.error(f"Error processing race_id {race_id}: {e}")
                finally:
                    pbar_races.update(1)

    if all_bets:
        bets = pd.concat(all_bets, ignore_index=True)
    else:
        bets = pd.DataFrame(columns=['レースID', 'Bet_Type', 'Combination', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout'])

    bets['Date'] = pd.to_datetime(bets['レースID'].str[:8], format='%Y%m%d')

    total_investment = bets['Bet_Amount'].sum()
    total_payout = bets['Payout'].sum()
    return_rate = (total_payout / total_investment) * 100 if total_investment > 0 else 0

    return bets, total_investment, total_payout, return_rate

def calculate_capital_evolution(bets, initial_capital=100000):
    """
    資金の推移を計算します。
    """
    if bets.empty:
        return pd.DataFrame({
            'Cumulative_Investment': [0],
            'Cumulative_Payout': [0],
            'Net': [0],
            'Capital': [initial_capital]
        })

    bets['date'] = pd.to_datetime(bets['レースID'].str[:8], format='%Y%m%d')
    bets_sorted = bets.sort_values(by=['date']).reset_index(drop=True)
    bets_sorted['cumulative_bet'] = bets_sorted['Bet_Amount'].cumsum()
    bets_sorted['cumulative_payout'] = bets_sorted['Payout'].cumsum()
    bets_sorted['Net'] = bets_sorted['cumulative_payout'] - bets_sorted['cumulative_bet']
    bets_sorted['Capital'] = initial_capital + bets_sorted['Net']

    return bets_sorted

def plot_cumulative_returns(cumulative_returns):
    """
    各EV閾値ごとの資金の推移を一つのグラフにプロットします。
    """
    plt.figure(figsize=(12, 6))

    for ev_threshold, capital_evolution in cumulative_returns.items():
        if capital_evolution.empty:
            continue
        plt.plot(capital_evolution['Capital'].values, label=f'EV閾値={ev_threshold}')

    plt.title('資金の推移（EV閾値ごとの比較）')
    plt.xlabel('ベット回数')
    plt.ylabel('資金 (円)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_return_comparison_trifecta2_allvenue_test.png')
    plt.show()
    print("資金の推移を 'cumulative_return_comparison_trifecta2_allvenue_test.png' として保存しました。")

def load_odds_data(base_dir, start_date, end_date, bet_type='trifecta'):
    """
    指定されたディレクトリからオッズデータを読み込み、1つのDataFrameに統合します。
    """
    all_data = []
    current_date = start_date
    total_days = (end_date - start_date).days + 1
    pbar = tqdm(total=total_days, desc=f"Loading {bet_type.capitalize()} Odds")

    while current_date <= end_date:
        date_str = current_date.strftime('%y%m%d')
        yymm = current_date.strftime('%y%m')
        file_name = f'{bet_type}_{date_str}.csv'
        file_path = os.path.join(base_dir, yymm, file_name)

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['レースID'] = df.apply(
                    lambda row: f"{current_date.strftime('%Y%m%d')}{int(row['JCD']):02d}{int(row['Race']):02d}",
                    axis=1
                )
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            pass  # ファイルが存在しない場合はスキップ

        current_date += timedelta(days=1)
        pbar.update(1)

    pbar.close()

    if all_data:
        odds_data_df = pd.concat(all_data, ignore_index=True)
        return odds_data_df
    else:
        print(f"No {bet_type} odds data loaded.")
        return pd.DataFrame()

def calculate_trifecta_probability(p1, p2, p3):
    """
    三連単の確率を計算します。
    """
    return p1 * p2 * p3  # 簡略化した計算

if __name__ == '__main__':
    inspect_model()
