# model_1min.py
import matplotlib.dates as mdates  # 追加
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

def calculate_sharpe_ratio(capital_evolution, risk_free_rate=0.0):
    """
    シャープレシオを計算します。
    Args:
        capital_evolution (DataFrame): 資金の推移を含むDataFrame
        risk_free_rate (float): 無リスク金利（デフォルトは0）
    Returns:
        float: シャープレシオ
    """
    # 日次リターンを計算
    capital = capital_evolution['Capital']
    returns = capital.pct_change().dropna()
    
    # シャープレシオを計算
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() if returns.std() != 0 else 0.0
    return sharpe_ratio

def calculate_max_drawdown(capital_evolution):
    """
    最大ドローダウンを計算します。
    Args:
        capital_evolution (DataFrame): 資金の推移を含むDataFrame
    Returns:
        float: 最大ドローダウン（パーセンテージ）
    """
    capital = capital_evolution['Capital']
    rolling_max = capital.cummax()
    drawdowns = (capital - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100  # パーセンテージ表示
    return max_drawdown


def inspect_model():
    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_first.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_first.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    # month_folders = [
    #     '2211', '2212', '2301', '2302', '2303', '2304', '2305', '2306',
    #     '2307', '2308', '2309', '2310', '2311', '2312', '2401', '2402',
    #     '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410'
    # ]

    # month_folders = ['2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410']
    # month_folders = ['2212', '2301', '2302', '2303', '2304', '2305', '2306', '2407', '2408', '2409', '2410', '2411']
    month_folders = ['2409','2410','2411']
    # month_folders = ['2410']

    date_files = {
        # '2211': [f'2211{day:02d}' for day in range(1, 31)],
        # '2212': [f'2212{day:02d}' for day in range(1, 32)],
        # '2301': [f'2301{day:02d}' for day in range(1, 32)],
        # '2302': [f'2302{day:02d}' for day in range(1, 29)],
        # '2303': [f'2303{day:02d}' for day in range(1, 32)],
        # '2304': [f'2304{day:02d}' for day in range(1, 31)],
        # '2305': [f'2305{day:02d}' for day in range(1, 32)],
        # '2306': [f'2306{day:02d}' for day in range(1, 31)],
        # '2307': [f'2307{day:02d}' for day in range(1, 32)],
        # '2308': [f'2308{day:02d}' for day in range(1, 32)],
        # '2309': [f'2309{day:02d}' for day in range(1, 31)],
        # '2310': [f'2310{day:02d}' for day in range(1, 31)],
        # '2311': [f'2311{day:02d}' for day in range(1, 31)],
        # '2312': [f'2312{day:02d}' for day in range(1, 32)],
        # '2401': [f'2401{day:02d}' for day in range(1, 32)],
        # '2402': [f'2402{day:02d}' for day in range(1, 29)],
        # '2403': [f'2403{day:02d}' for day in range(1, 32)],
        # '2404': [f'2404{day:02d}' for day in range(1, 31)],
        # '2405': [f'2405{day:02d}' for day in range(1, 32)],
        # '2406': [f'2406{day:02d}' for day in range(1, 31)],
        # '2407': [f'2407{day:02d}' for day in range(1, 32)],
        # '2408': [f'2408{day:02d}' for day in range(1, 32)],
        '2409': [f'2409{day:02d}' for day in range(1, 31)],
        '2410': [f'2410{day:02d}' for day in range(1, 32)],
        '2411': [f'2411{day:02d}' for day in range(1, 4)],
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

                if not b_data.empty and not k_data.empty and not before_data.empty:
                    # データのマージ
                    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
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

    data_odds = load_processed_data()

    # 全データを結合
    data = pd.concat(data_list, ignore_index=True)
    odds_list = pd.concat(odds_list1, ignore_index=True)

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

    # 天候_雪 カラムが存在しない場合に追加する
    for col in feature_names:
        if col not in X_processed.columns:
            X_processed[col] = 0

    # 学習時の特徴量と順序を合わせる
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

    # 三連単オッズデータの読み込み
    base_dir = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\trifecta_odds'
    start_date = datetime.strptime('2024-09-01', '%Y-%m-%d')
    end_date = datetime.strptime('2024-11-11', '%Y-%m-%d')
    total_capital = 100000  # 総資金（円）
    unit = 100  # 賭け金の単位（円）
    prob_threshold = 0  # 賭ける確率の閾値

    # 三連単オッズデータの読み込み
    trifecta_odds_data_df = load_trifecta_odds(base_dir, start_date, end_date)

    if trifecta_odds_data_df.empty:
        print("No trifecta odds data available. Exiting.")
        return

    # 各艇の確率データ
    data_boats = data[['レースID', '艇番', 'prob_0', '着', '会場']].copy()

    # EV計算のための減算値のリスト
    ev_subtract_values = [1,4,5,6,10]

    # 結果を保存するリスト
    cumulative_returns = {}
    summary_results = []

    for ev_value in ev_subtract_values:
        print(f"\n=== EV計算の減算値: {ev_value} ===")
        # 三連単賭け金と回収金額の計算
        trifecta_bets, total_investment, total_payout, return_rate = calculate_trifecta_bets(
            trifecta_odds_data_df,
            data_boats,
            total_capital=total_capital,
            unit=unit,
            prob_threshold=prob_threshold,
            ev_subtract=ev_value  # EV計算の減算値を追加
        )

        # 資金の推移を計算
        capital_evolution = calculate_capital_evolution(trifecta_bets)


        # シャープレシオと最大ドローダウンを計算
        sharpe_ratio = calculate_sharpe_ratio(capital_evolution, risk_free_rate=0.0)
        max_drawdown = calculate_max_drawdown(capital_evolution)

        # 結果を保存
        cumulative_returns[ev_value] = capital_evolution
        summary_results.append({
            'EV_Subtract': ev_value,
            'Total Investment (yen)': total_investment,
            'Total Payout (yen)': total_payout,
            'Return Rate (%)': return_rate,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown
        })

    # 結果の可視化
    plot_cumulative_returns(cumulative_returns)

    # 結果の比較表を作成
    summary_df = pd.DataFrame(summary_results)
    print("\n結果の比較:")
    print(summary_df)

    # 結果をCSVに保存
    summary_df.to_csv('trifecta_summary_comparison_single_test.csv', index=False)
    print("結果の比較を 'trifecta_summary_comparison_single_test.csv' として保存しました。")

 
    # ------------- 追加部分開始 -------------
    for ev_value in ev_subtract_values:


        print(f"\n=== 日付ごとの収益および投資金額の計算 (EV_Subtract={ev_value}) ===")
        trifecta_bets = cumulative_returns[ev_value]

        # trifecta_bets に '会場' 情報を追加するために 'data' とマージ
        data_unique = data[['レースID', '会場']].drop_duplicates(subset='レースID')
        trifecta_bets = trifecta_bets.merge(data_unique, on='レースID', how='left')
        print(trifecta_bets[trifecta_bets['Payout'] > 0])
        # pd.set_option('display.max_rows', 100)
        # print(trifecta_bets[trifecta_bets['レースID']=='202411032412'].sort_values(by='レースID'))

        # 日付ごとの収益を計算
        revenue_per_date = trifecta_bets.groupby('Date').apply(
            lambda df: df['Payout'].sum() - df['Bet_Amount'].sum()
        ).reset_index(name='Net_Revenue')

        # 日付ごとの投資金額を計算
        investment_per_date = trifecta_bets.groupby('Date')['Bet_Amount'].sum().reset_index()

        # 日付ごとの収益をプロット（棒グラフ）
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Date', y='Net_Revenue', data=revenue_per_date, color='green')
        plt.title(f'日付ごとの収益 (EV_Subtract={ev_value})')
        plt.xlabel('日付')
        plt.ylabel('収益 (円)')
        plt.gcf().autofmt_xdate()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'revenue_per_date_ev_{ev_value}_test.png')
        plt.show()
        print(f"日付ごとの収益を 'revenue_per_date_ev_{ev_value}_test.png' として保存しました。")

        # 1日当たりの投資金額を計算
        investment_per_date = trifecta_bets.groupby('Date')['Bet_Amount'].sum().reset_index()

        # 統計量の計算
        min_investment = investment_per_date['Bet_Amount'].min()
        max_investment = investment_per_date['Bet_Amount'].max()
        average_investment = investment_per_date['Bet_Amount'].mean()
        median_investment = investment_per_date['Bet_Amount'].median()

        # 統計結果の表示
        print("\n1日当たりの投資金額の統計量:")
        print(f"最小値: {min_investment:.2f} 円")
        print(f"最大値: {max_investment:.2f} 円")
        print(f"平均値: {average_investment:.2f} 円")
        print(f"中央値: {median_investment:.2f} 円")

        # 1日当たりの投資金額のヒストグラムをプロット
        plt.figure(figsize=(10, 6))
        sns.histplot(investment_per_date['Bet_Amount'], bins=30, kde=True, color='purple')
        plt.title('1日当たりの投資金額のヒストグラム')
        plt.xlabel('投資金額 (円)')
        plt.ylabel('頻度')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'investment_per_date_histogram_ev_{ev_value}_test.png')
        plt.show()
        print(f"1日当たりの投資金額のヒストグラムを 'investment_per_date_histogram_ev_{ev_value}_test.png' として保存しました。")

        # 会場ごとの回収率を計算
        venue_summary = trifecta_bets.groupby('会場').agg(
            Total_Payout=('Payout', 'sum'),
            Total_Bet=('Bet_Amount', 'sum')
        ).reset_index()
        venue_summary['Return_Rate (%)'] = (venue_summary['Total_Payout'] / venue_summary['Total_Bet']) * 100
        venue_summary = venue_summary[['会場', 'Return_Rate (%)']]

        # 会場ごとの回収率をCSVに保存
        venue_summary.to_csv(f'return_rate_per_venue_ev_{ev_value}_test.csv', index=False)
        print(f"会場ごとの回収率を 'return_rate_per_venue_ev_{ev_value}_test.csv' として保存しました。")
    # ------------- 追加部分終了 -------------
def process_race(race_id, race_odds, data_boats, total_capital=100000, unit=100, prob_threshold=0.0, ev_subtract=1.4):
    """
    各レースごとに三連単のボックス投票（上位4艇）を決定し、回収金額を計算します。
    Args:
        race_id (str): レースID
        race_odds (DataFrame): 該当レースの三連単オッズデータ
        data_boats (DataFrame): 各レースの各艇の確率データ
        total_capital (int): 総資金
        unit (int): 賭け金の単位（円）
        prob_threshold (float): この確率以下の組み合わせは賭けない
        ev_subtract (float): EV計算の減算値
    Returns:
        DataFrame: 各三連単の賭け金額と回収金額
    """
    # 該当レースの艇の確率データを取得
    boats_data = data_boats[data_boats['レースID'] == race_id]
    # if boats_data.empty or boats_data['会場'].values[0]=='05' or boats_data['会場'].values[0]=='06' or boats_data['会場'].values[0]=='07' or boats_data['会場'].values[0]=='08' or boats_data['会場'].values[0]=='09' or boats_data['会場'].values[0]=='11' or boats_data['会場'].values[0]=='12' or boats_data['会場'].values[0]=='18' or boats_data['会場'].values[0] == '22' or boats_data['会場'].values[0]=='23':
    if boats_data.empty:
        # データがない場合
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout']
        return pd.DataFrame(columns=columns)

    boats_probs = boats_data.set_index('艇番')['prob_0'].to_dict()

    # 確率が高い順に上位4艇を選択
    top_boats = boats_data.sort_values(by='prob_0', ascending=False)['艇番'].head(4).tolist()

    if len(top_boats) < 3:
        # 上位4艇が3艇未満の場合
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout']
        return pd.DataFrame(columns=columns)

    # 全ての順列（三連単）
    trifecta_combinations = list(itertools.permutations(top_boats, 3))  # 4P3 = 24 combinations

    # Create a DataFrame of these combinations
    trifecta_df = pd.DataFrame(trifecta_combinations, columns=['Boat1', 'Boat2', 'Boat3'])

    # Merge with race_odds to get odds
    merged_df = pd.merge(trifecta_df, race_odds, on=['Boat1', 'Boat2', 'Boat3'], how='left')

    # Drop combinations that do not have odds (if any)
    merged_df = merged_df.dropna(subset=['Odds'])

    # Calculate probability
    merged_df['Probability'] = merged_df.apply(lambda row: calculate_trifecta_probability(
        boats_probs.get(row['Boat1'], 0),
        boats_probs.get(row['Boat2'], 0),
        boats_probs.get(row['Boat3'], 0)
    ), axis=1)

    # Calculate EV
    merged_df['EV'] = (merged_df['Probability'] * merged_df['Odds']) - ev_subtract
    # print(merged_df[merged_df['レースID'] == '202411032412'])
    # Filter based on EV
    merged_df = merged_df[merged_df['EV'] > 0]

    if merged_df.empty:
        # 賭けがなかった場合
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout']
        return pd.DataFrame(columns=columns)

    # Assign bet amount (fixed unit per bet)
    merged_df['Bet_Amount'] = unit  # 固定ベット金額

    # Determine the actual combination
    boats_actual = boats_data.sort_values(by='着')['艇番'].tolist()
    if len(boats_actual) < 3:
        actual_combo = ()
    else:
        actual_combo = tuple(boats_actual[:3])

    # Calculate payout
    merged_df['Payout'] = merged_df.apply(
        lambda row: row['Bet_Amount'] * row['Odds'] if (row['Boat1'], row['Boat2'], row['Boat3']) == actual_combo else 0,
        axis=1
    )

    # Assign race_id
    merged_df['レースID'] = race_id

    # Select and reorder columns
    trifecta_bet = merged_df[['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout']]

    return trifecta_bet

def calculate_trifecta_bets(trifecta_odds_df, data_boats, total_capital=100000, unit=100, prob_threshold=0.0, ev_subtract=1.4):
    """
    全レースに対して三連単の賭け金額を決定し、回収率や資金の推移を計算します。
    """
    race_ids = trifecta_odds_df['レースID'].unique()
    all_trifecta_bets = []

    # trifecta_odds_df を 'レースID' でグループ化
    grouped_odds = trifecta_odds_df.groupby('レースID')

    # race_odds_list をリストとしてまとめる
    race_odds_list = []
    total_groups = len(grouped_odds)
    with tqdm(total=total_groups, desc="Grouping Races") as pbar_group:
        for race_id, race_odds in grouped_odds:
            race_odds_list.append((race_id, race_odds))
            pbar_group.update(1)

    # ThreadPoolExecutor を使用して並列処理
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 並列で処理を実行
        futures = {
            executor.submit(
                process_race, race_id, race_odds, data_boats, total_capital, unit, prob_threshold, ev_subtract
            ): race_id for race_id, race_odds in race_odds_list
        }

        # プログレスバーの設定
        with tqdm(total=len(futures), desc="Processing Races") as pbar_races:
            for future in as_completed(futures):
                race_id = futures[future]
                try:
                    trifecta_bet = future.result()
                    if not trifecta_bet.empty:
                        all_trifecta_bets.append(trifecta_bet)
                except Exception as e:
                    logging.error(f"Error processing race_id {race_id}: {e}")
                finally:
                    pbar_races.update(1)

    if all_trifecta_bets:
        trifecta_bets = pd.concat(all_trifecta_bets, ignore_index=True)
    else:
        trifecta_bets = pd.DataFrame(columns=['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout'])

    trifecta_bets['Date'] = pd.to_datetime(trifecta_bets['レースID'].str[:8], format='%Y%m%d')

    total_investment = trifecta_bets['Bet_Amount'].sum()
    total_payout = trifecta_bets['Payout'].sum()
    return_rate = (total_payout / total_investment) * 100 if total_investment > 0 else 0

    return trifecta_bets, total_investment, total_payout, return_rate

def calculate_capital_evolution(trifecta_bets, initial_capital=100000):
    """
    資金の推移を計算します。
    """
    if trifecta_bets.empty:
        return pd.DataFrame({
            'Cumulative_Investment': [0],
            'Cumulative_Payout': [0],
            'Net': [0],
            'Capital': [initial_capital]
        })

    trifecta_bets['date'] = pd.to_datetime(trifecta_bets['レースID'].str[:8], format='%Y%m%d')
    trifecta_bets_sorted = trifecta_bets.sort_values(by=['date']).reset_index(drop=True)
    trifecta_bets_sorted['cumulative_bet'] = trifecta_bets_sorted['Bet_Amount'].cumsum()
    trifecta_bets_sorted['cumulative_payout'] = trifecta_bets_sorted['Payout'].cumsum()
    trifecta_bets_sorted['Net'] = trifecta_bets_sorted['cumulative_payout'] - trifecta_bets_sorted['cumulative_bet']
    trifecta_bets_sorted['Capital'] = initial_capital + trifecta_bets_sorted['Net']

    return trifecta_bets_sorted

def plot_cumulative_returns(cumulative_returns):
    """
    各EV減算値ごとの資金の推移を一つのグラフにプロットします。
    Args:
        cumulative_returns (dict): EV減算値をキー、資金推移DataFrameを値とする辞書
    """
    plt.figure(figsize=(12, 6))

    for ev_value, capital_evolution in cumulative_returns.items():
        if capital_evolution.empty:
            continue
        plt.plot(capital_evolution['Capital'].values, label=f'EV減算値={ev_value}')

    plt.title('資金の推移（EV減算値ごとの比較）')
    plt.xlabel('ベット回数')
    plt.ylabel('資金 (円)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_return_comparison_single_test.png')
    plt.show()
    print("資金の推移を 'cumulative_return_comparison_single_test.png' として保存しました。")

def load_trifecta_odds(base_dir, start_date, end_date):
    """
    指定されたディレクトリから三連単オッズデータを読み込み、1つのDataFrameに統合します。
    """
    all_data = []
    current_date = start_date
    total_days = (end_date - start_date).days + 1
    pbar = tqdm(total=total_days, desc="Loading Trifecta Odds")

    while current_date <= end_date:
        date_str = current_date.strftime('%y%m%d')
        yymm = current_date.strftime('%y%m')
        file_name = f'trifecta_{date_str}.csv'
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
            print(f"File not found: {file_path}")

        current_date += timedelta(days=1)
        pbar.update(1)

    pbar.close()

    if all_data:
        trifecta_odds_data_df = pd.concat(all_data, ignore_index=True)
        return trifecta_odds_data_df
    else:
        print("No trifecta odds data loaded.")
        return pd.DataFrame()

def calculate_trifecta_probability(p1, p2, p3):
    """
    三連単の確率を計算します。
    """
    # denominator1 = 1 - p1
    # denominator2 = 1 - p1 - p2
    # denominator3 = 1 - p1 - p2 - p3
    # if denominator1 <= 0 or denominator2 <= 0:
    #     return 0
    # 各値の合計を計算
    total = p1 + p2 + p3
    p1 = p1 / total
    p2 = p2 / total
    p3 = p3 / total
    denominator1 = 1 - p1
    denominator2 = 1 - p1 * p2
    # if denominator1 <= 0 or denominator2 <= 0:
    #     return 0
    return (p1 * p2 * p3/denominator1/denominator2) 

if __name__ == '__main__':
    inspect_model()
