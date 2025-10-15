# model_1min.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, preprocess_data1min, preprocess_data_old, load_before_data, load_before1min_data ,merge_before_data, load_dataframe
from get_data import load_fan_data
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timedelta
from tqdm import tqdm
import itertools, logging

def load_fan_data_from_csv():
    file_path = 'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/fan_dataframe/fan_data.csv'
    if os.path.exists(file_path):
        df_fan = pd.read_csv(file_path)
        return df_fan
    else:
        print(f"ファイルが見つかりません: {file_path}")
        return pd.DataFrame()

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
        gbm_odds = lgb.Booster(model_file='boatrace_odds_model.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_trio.csv').squeeze().tolist()
    feature_names_odds = pd.read_csv('feature_names_odds.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    # month_folders = ['2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410']
    # month_folders = ['2211', '2212', '2301', '2302', '2303', '2304', '2305', '2306', '2307', '2308', '2309', '2310', '2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410']
    month_folders = ['2411']
    date_files = {
        # '2211': [f'2211{day:02d}' for day in range(1, 31)],  # 221101 - 221130
        # '2212': [f'2212{day:02d}' for day in range(1, 32)],  # 221201 - 221231
        # '2301': [f'2301{day:02d}' for day in range(1, 32)],  # 230101 - 230131
        # '2302': [f'2302{day:02d}' for day in range(1, 29)],  # 230201 - 230228
        # '2303': [f'2303{day:02d}' for day in range(1, 32)],  # 230301 - 230331
        # '2304': [f'2304{day:02d}' for day in range(1, 31)],  # 230401 - 230430
        # '2305': [f'2305{day:02d}' for day in range(1, 32)],  # 230501 - 230531
        # '2306': [f'2306{day:02d}' for day in range(1, 31)],  # 230601 - 230630
        # '2307': [f'2307{day:02d}' for day in range(1, 32)],  # 230701 - 230731
        # '2308': [f'2308{day:02d}' for day in range(1, 32)],  # 230801 - 230831
        # '2309': [f'2309{day:02d}' for day in range(1, 31)],  # 230901 - 230930
        # '2310': [f'2310{day:02d}' for day in range(1, 31)],  # 231001 - 231031
        # '2311': [f'2311{day:02d}' for day in range(1, 31)],  # 231101 - 231130
        # '2312': [f'2312{day:02d}' for day in range(1, 32)],  # 231201 - 231231
        # '2401': [f'2401{day:02d}' for day in range(1, 32)],  # 240101 - 240131
        # '2402': [f'2402{day:02d}' for day in range(1, 29)],  # 240201 - 240228
        # '2403': [f'2403{day:02d}' for day in range(1, 15)],  # 240301 - 240331
        # '2404': [f'2404{day:02d}' for day in range(1, 31)],  # 240401 - 240430
        # '2405': [f'2405{day:02d}' for day in range(1, 32)],  # 240501 - 240531
        # '2406': [f'2406{day:02d}' for day in range(1, 31)],  # 240601 - 240630
        # '2407': [f'2407{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        # '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        # '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(1,32)],   # 241001 - 241008
        '2411': [f'2411{day:02d}' for day in range(1, 5)],  # 241101 - 241130
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
                # before1min_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data_1min/{month}/beforeinfo1min_{date}.txt'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'


                b_data = load_b_file(b_file)
                k_data, odds_list_part = load_k_file(k_file)
                before_data = load_before_data(before_file)
                # before1min_data = load_before1min_data(before1min_file)

                if not b_data.empty and not k_data.empty and not before_data.empty:
                    # データのマージ
                    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
                    before_data = remove_common_columns(data, before_data, on_columns=['選手登番', 'レースID', '艇番'])
                    data = merge_before_data(data, before_data)
                    # before1min_data = remove_common_columns(data, before1min_data, on_columns=['選手登番', 'レースID', '艇番'])
                    # data = merge_before_data(data, before1min_data)
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
    print(data_odds[["win_odds_mean","選手登番","艇番","win_odds"]])
    print(data_odds[data_odds['選手登番']=='5107'][["win_odds_mean","選手登番","艇番","win_odds"]])
    # data_list['win_odds_mean']=data_list.groupby(['選手登番','艇番'])['win_odds'].transform(lambda x: x.mean())

    # データの確認
    # print("data_list:", data_list)

    # 全データを結合
    data = pd.concat(data_list, ignore_index=True)
    odds_list = pd.concat(odds_list1, ignore_index=True)
    # print("data columns:", data.columns.tolist())
    # print(data)
    # print(data[['win_odds1min', 'place_odds1min']].head())
    # print(data[['win_odds1min', 'place_odds1min']].isnull().sum())
    # 前処理
  
    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    # 必要な列を指定してマージ
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')
    # print("data columns:", data.columns.tolist())
    # print(data)

    data = preprocess_data(data)
    # print("data columns:", data.columns.tolist())
    # print(data)

    # 特徴量の指定
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h','前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather', 'wind_d','ESC','艇番','性別','支部','級別']
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # 数値データの選択
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]

    # 特徴量の結合
    X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
    # 天候_雪 カラムが存在しない場合に追加する
    for col in feature_names:
        if col not in X_processed.columns:
            X_processed[col] = 0
    # print("X_processed columns:", X_processed.columns.tolist())
    X_processed = X_processed[feature_names]  # 学習時の特徴量と順序を合わせる

    # 予測
    y_pred = gbm.predict(X_processed)

    # data['prob_0'] = y_pred[:, 0]  # 1,2着の確率
    # data['prob_1'] = y_pred[:, 1]
    # data['prob_2'] = y_pred[:, 2]

    data['prob_0'] = y_pred[:, 0]  # 1,着の確率
    print(data[['prob_0', '着']].head(50))

    # 'prob_0' の分布をプロット
    plt.figure(figsize=(8, 6))
    plt.hist(data['prob_0'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title("予測確率 'prob_0' の分布")
    plt.xlabel("予測確率 (prob_0)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('prob_0_distribution_trifecta_allvenue.png')
    plt.show()
    print("プロットが 'prob_0_distribution.png' として保存されました。")


    #　確定後オッズの予測
    # 特徴量の指定
    boats = [1, 2, 3, 4, 5, 6]
    data_list = []
    for boat in boats:
        boat_data = data.copy()
        boat_data = boat_data[boat_data['艇番'] == boat].copy()
        boat_data['final_odds'] = boat_data['win_odds']  # 確定後のオッズをターゲット
        # boat_data['before1min_odds'] = boat_data['win_odds1min']  # 1分前のオッズを特徴量
        boat_data['boat_number'] = boat  # 舟番号を特徴量として追加
        data_list.append(boat_data)

    # 各艇のデータを結合
    data = pd.concat(data_list, ignore_index=True)
    # '選手登番' と '艇番' ごとに 'win_odds_mean' を集約（例: 平均を取る）
    data_odds_grouped = data_odds.groupby(['選手登番', '艇番'], as_index=False)['win_odds_mean'].mean()
    # '選手登番'のデータ型を文字列に変換
    data['選手登番'] = data['選手登番'].astype(int)
    data_odds_grouped['選手登番'] = data_odds_grouped['選手登番'].astype(int)

    # '艇番'のデータ型も確認し、一致させる (同様にstrに変換)
    data['艇番'] = data['艇番'].astype(int)
    data_odds_grouped['艇番'] = data_odds_grouped['艇番'].astype(int)
    # '選手登番' と '艇番' をキーにして、'win_odds_mean' を data にマージ
    data = data.merge(data_odds_grouped, on=['選手登番', '艇番'], how='left')

   # パラメータ設定
    base_dir = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\trifecta_odds'
    start_date = datetime.strptime('2024-11-01', '%Y-%m-%d')
    end_date = datetime.strptime('2024-11-02', '%Y-%m-%d')  # 注意: end_dateがstart_dateより前になっています
    total_capital = 100000  # 総資金（円）
    unit = 100  # 賭け金の単位（円）
    prob_threshold = 0  # 賭ける確率の閾値
    
    # 三連単オッズデータの読み込み
    trifecta_odds_data_df = load_trifecta_odds(base_dir, start_date, end_date)
    
    if trifecta_odds_data_df.empty:
        print("No trifecta odds data available. Exiting.")
        return
    
    # 各艇の確率データの読み込み（実際のデータに置き換えてください）
    # data_boats = pd.read_csv('path_to_data_boats.csv')
    
    data_boats = data[['レースID', '艇番', 'prob_0', '着','会場']].copy()


    # 三連単賭け金と回収金額の計算
    trifecta_bets, total_investment, total_payout, return_rate = calculate_trifecta_bets(
        trifecta_odds_data_df, 
        data_boats, 
        total_capital=total_capital, 
        unit=unit, 
        prob_threshold=prob_threshold
    )
    
    # 資金の推移を計算
    capital_evolution = calculate_capital_evolution(trifecta_bets)
    
    # 結果の可視化
    plot_results(trifecta_bets, capital_evolution)
    
    # **追加部分開始** #
    # Calculate Sharpe Ratio
    if not capital_evolution.empty and len(capital_evolution) > 1:
        # Calculate daily returns as percentage change in Capital
        capital_evolution['Daily_Return'] = capital_evolution['Capital'].pct_change()
        # Drop NaN values resulting from pct_change
        daily_returns = capital_evolution['Daily_Return'].dropna()
        # Compute mean and standard deviation of daily returns
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        # Calculate Sharpe Ratio (assuming risk-free rate = 0)
        sharpe_ratio = mean_return / std_return if std_return != 0 else 0
    else:
        sharpe_ratio = 0
        print("Insufficient data to calculate Sharpe Ratio.")

    # Print Sharpe Ratio
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Create a summary of main results
    summary = {
        'Total Investment (yen)': [total_investment],
        'Total Payout (yen)': [total_payout],
        'Return Rate (%)': [return_rate],
        'Sharpe Ratio': [sharpe_ratio]
    }
    summary_df = pd.DataFrame(summary)

    # Save summary to a CSV file
    summary_file = 'trifecta_summary_allvenue.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary of main results saved to {summary_file}.")

    print(trifecta_bets[trifecta_bets['Payout'] > 0])


    # Save 'trifecta_bets' DataFrame to a CSV file
    trifecta_bets_file = 'trifecta_bets_allvenue.csv'
    trifecta_bets.to_csv(trifecta_bets_file, index=False)
    print(f"'trifecta_bets' DataFrame saved to {trifecta_bets_file}.")
    # **追加部分終了** #

    # 各会場ごとの回収率を計算・表示
    calculate_return_rate_by_venue(trifecta_bets)

    correlation_analysis(trifecta_bets, data)
    visualize_conditions(trifecta_bets, data)

    # 総回収率の表示
    print(f"\n総投資額: {total_investment:.2f}円")
    print(f"総回収額: {total_payout:.2f}円")
    print(f"回収率: {return_rate:.2f}%")

    # 最初の数レースの結果を表示
    if not trifecta_bets.empty:
        pd.set_option('display.max_rows', 500)
        print("\n最初の数レースの詳細:")
        print(trifecta_bets.head(50))
    else:
        # 列を持つ空のDataFrameを表示
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'Kelly_Fraction', 'Bet_Amount', 'Payout']
        empty_df = pd.DataFrame(columns=columns)
        print("\nNo trifecta bets were placed. Displaying an empty DataFrame structure:")
        print(empty_df)

import itertools, logging
# 賭け金額を決定し、回収金額を計算する関数（ボックス投票対応）
def process_race(race_id, race_odds, data_boats, total_capital=100000, unit=100, prob_threshold=0.0):
    """
    各レースごとに三連単のボックス投票（上位4艇）を決定し、回収金額を計算します。
    Args:
        race_id (str): レースID
        race_odds (DataFrame): 該当レースの三連単オッズデータ
        data_boats (DataFrame): 各レースの各艇の確率データ
        total_capital (int): 総資金
        unit (int): 賭け金の単位（円）
        prob_threshold (float): この確率以下の組み合わせは賭けない
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
    merged_df['EV'] = (merged_df['Probability'] * merged_df['Odds']) - 1
    print(merged_df)
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

def calculate_trifecta_bets(trifecta_odds_df, data_boats, total_capital=100000, unit=100, prob_threshold=0.0):
    """
    全レースに対して三連単の賭け金額を決定し、回収率や資金の推移を計算します。
    Args:
        trifecta_odds_df (DataFrame): 三連単オッズのデータ
        data_boats (DataFrame): 各レースの各艇の確率データ
        total_capital (int): 総資金
        unit (int): 賭け金の単位（円）
        prob_threshold (float): この確率以下の組み合わせは賭けない
    Returns:
        DataFrame: 全ての賭け金と回収金額
        float: 総投資額
        float: 総回収額
        float: 回収率
    """
    race_ids = trifecta_odds_df['レースID'].unique()
    all_trifecta_bets = []

    # trifecta_odds_df を 'レースID' でグループ化
    grouped_odds = trifecta_odds_df.groupby('レースID')

    # race_odds_list をリストとしてまとめる（進行状況を表示）
    race_odds_list = []
    total_groups = len(grouped_odds)
    with tqdm(total=total_groups, desc="Grouping Races") as pbar_group:
        for race_id, race_odds in grouped_odds:
            race_odds_list.append((race_id, race_odds))
            pbar_group.update(1)

    # ThreadPoolExecutor を使用して並列処理
    with ThreadPoolExecutor(max_workers=2000) as executor:
        # 並列で処理を実行
        futures = {
            executor.submit(
                process_race, race_id, race_odds, data_boats, total_capital, unit, prob_threshold
            ): race_id for race_id, race_odds in race_odds_list
        }

        # プログレスバーの設定（処理中の進行状況と残り時間を表示）
        with tqdm(total=len(futures), desc="Processing Races") as pbar_races:
            for future in as_completed(futures):
                race_id = futures[future]
                try:
                    trifecta_bet = future.result()
                    if not trifecta_bet.empty:
                        all_trifecta_bets.append(trifecta_bet)
                        if len(all_trifecta_bets) <= 3:
                            logging.info(f"\nレースID: {race_id} の賭け結果:")
                            logging.info(trifecta_bet)
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

# 資金の推移を計算する関数
def calculate_capital_evolution(trifecta_bets, initial_capital=100000):
    """
    資金の推移を計算します。
    Args:
        trifecta_bets (DataFrame): 全ての賭け金と回収金額
        initial_capital (int): 初期資金
    Returns:
        DataFrame: 資金の推移
    """
    if trifecta_bets.empty:
        # 賭けがなかった場合、初期資金のみのDataFrameを返す
        return pd.DataFrame({
            'Cumulative_Investment': [0],
            'Cumulative_Payout': [0],
            'Net': [0],
            'Capital': [initial_capital]
        })
    
    # 'レースID'から 'date' カラムを生成（YYYYMMDD形式）
    trifecta_bets['date'] = pd.to_datetime(trifecta_bets['レースID'].str[:8], format='%Y%m%d')
    
    # 日付順にソート
    trifecta_bets_sorted = trifecta_bets.sort_values(by=['date']).reset_index(drop=True)
    
    # 累積投資額と累積回収額を計算
    trifecta_bets_sorted['cumulative_bet'] = trifecta_bets_sorted['Bet_Amount'].cumsum()
    trifecta_bets_sorted['cumulative_payout'] = trifecta_bets_sorted['Payout'].cumsum()
    
    # 純利益と資本を計算
    trifecta_bets_sorted['Net'] = trifecta_bets_sorted['cumulative_payout'] - trifecta_bets_sorted['cumulative_bet']
    trifecta_bets_sorted['Capital'] = initial_capital + trifecta_bets_sorted['Net']
    
    return trifecta_bets_sorted

def plot_results(trifecta_bets, capital_evolution):
    """
    回収率や資金の推移をプロットします。
    Args:
        trifecta_bets (DataFrame): 全ての賭け金と回収金額
        capital_evolution (DataFrame): 資金の推移
    """
    # 回収率の表示
    total_investment = trifecta_bets['Bet_Amount'].sum() if not trifecta_bets.empty else 0
    total_payout = trifecta_bets['Payout'].sum() if not trifecta_bets.empty else 0
    return_rate = (total_payout / total_investment) * 100 if total_investment > 0 else 0
    print(f"\n総投資額: {total_investment:.2f}円")
    print(f"総回収額: {total_payout:.2f}円")
    print(f"回収率: {return_rate:.2f}%")
    
    # 最初の数レースの結果を表示
    if not trifecta_bets.empty:
        print("\n最初の数レースの詳細:")
        print(trifecta_bets.head(5))
    else:
        print("\n賭けがなかったため、レースの詳細はありません。")
    
    # 資金の推移プロット
    if capital_evolution.empty or (capital_evolution.shape[0] == 1 and capital_evolution['Capital'].iloc[0] == 100000):
        print("\n資金の推移をプロットできません。")
    else:
        # 日付ごとの総投資額と総回収額を計算
        daily_summary = trifecta_bets.groupby('date').agg(
            daily_investment=pd.NamedAgg(column='Bet_Amount', aggfunc='sum'),
            daily_payout=pd.NamedAgg(column='Payout', aggfunc='sum')
        ).reset_index()
    
        # 日付ごとの純利益を計算
        daily_summary['daily_return'] = daily_summary['daily_payout'] - daily_summary['daily_investment']
    
        # 累積リターンを計算
        daily_summary['cumulative_return'] = daily_summary['daily_return'].cumsum()
    
        # 累積投資額と累積回収額を計算
        daily_summary['cumulative_bet'] = daily_summary['daily_investment'].cumsum()
        daily_summary['cumulative_payout'] = daily_summary['daily_payout'].cumsum()
    
        # 日付を文字列形式に変換（プロットの見やすさのため）
        daily_summary['date_str'] = daily_summary['date'].dt.strftime('%Y-%m-%d')
    
        # 単純な累積投資と回収のプロット
        plt.figure(figsize=(10, 6))
        plt.plot(daily_summary['date'], daily_summary['cumulative_payout'], label='累積回収金額', color='green')
        plt.plot(daily_summary['date'], daily_summary['cumulative_bet'], label='累積投資金額', color='red')
        plt.title("三連単の累積回収金額 vs 累積投資金額")
        plt.xlabel("ベット回数")
        plt.ylabel("金額 (円)")
        plt.legend()
        plt.grid(True)
        plt.savefig('cumulative_return_trifecta_allvenue.png')
        plt.show()
        print("プロットが 'cumulative_return_trifecta_allvenue.png' として保存されました。")
    
        # 日付ごとの累積リターンのプロット
        plt.figure(figsize=(12, 6))
        plt.plot(daily_summary['date'], daily_summary['cumulative_return'], color='purple')
        plt.title("日付ごとの累積リターンの推移")
        plt.xlabel("日付")
        plt.ylabel("累積リターン (円)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('cumulative_return_by_date_trifecta_allvenue.png')
        plt.show()
        print("日付ごとの累積リターンの推移が 'cumulative_return_by_date_trifecta_allvenue.png' として保存されました。")
    
        # 回収金額のヒストグラム
        if not trifecta_bets.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(trifecta_bets['Payout'], bins=50, kde=True, color='green')
            plt.title("回収金額の分布")
            plt.xlabel("回収金額 (円)")
            plt.ylabel("度数")
            plt.grid(True)
            plt.savefig('payout_distribution_trifecta_allvenue.png')
            plt.show()
            print("回収金額の分布が 'payout_distribution_trifecta.png' として保存されました。")
    
            # 賭け金額のヒストグラム
            plt.figure(figsize=(10, 6))
            sns.histplot(trifecta_bets['Bet_Amount'], bins=50, kde=True, color='orange')
            plt.title("賭け金額の分布")
            plt.xlabel("賭け金額 (円)")
            plt.ylabel("度数")
            plt.grid(True)
            plt.savefig('bet_amount_distribution_trifecta_allvenue.png')
            plt.show()
            print("賭け金額の分布が 'bet_amount_distribution_trifecta.png' として保存されました。")
    
            # 投資金額と回収金額の比較プロット
            plt.figure(figsize=(12, 6))
            plt.bar(daily_summary['date'], daily_summary['cumulative_bet'], label='累積投資金額', alpha=0.6, color='red')
            plt.bar(daily_summary['date'], daily_summary['cumulative_payout'], label='累積回収金額', alpha=0.6, color='blue')
            plt.title("累積投資金額と累積回収金額の比較")
            plt.xlabel("日付")
            plt.ylabel("金額 (円)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('investment_vs_payout_trifecta_allvenue.png')
            plt.show()
            print("累積投資金額と累積回収金額の比較が 'investment_vs_payout_trifecta.png' として保存されました。")
    
            # 日付を横軸とした累積収益の推移プロット
            plt.figure(figsize=(12, 6))
            plt.plot(daily_summary['date'], daily_summary['cumulative_return'], marker='o', linestyle='-', color='purple')
            plt.title("日付を横軸とした累積収益の推移")
            plt.xlabel("日付")
            plt.ylabel("累積収益 (円)")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('cumulative_return_by_date_trifecta_allvenue.png')
            plt.show()
            print("日付を横軸とした累積収益の推移が 'cumulative_return_by_date_trifecta_allvenue.png' として保存されました。")
        else:
            print("Net列が存在しないため、累積収益の推移をプロットできません。")

def calculate_return_rate_by_venue(trifecta_bets):
    """
    各会場ごとの総投資額、総回収額、回収率を計算して表示します。
    Args:
        trifecta_bets (DataFrame): 全ての賭け金と回収金額を含むDataFrame
    """
    if trifecta_bets.empty:
        print("賭けデータが存在しません。")
        return
    
    # 'レースID'の後ろから4番目から3番目の2桁を会場コードとして抽出
    trifecta_bets['会場コード'] = trifecta_bets['レースID'].astype(str).str[-4:-2]
    
    # 各会場ごとに総投資額と総回収額を計算
    venue_summary = trifecta_bets.groupby('会場コード').agg(
        総投資額=('Bet_Amount', 'sum'),
        総回収額=('Payout', 'sum')
    ).reset_index()
    
    # 回収率を計算
    venue_summary['回収率(%)'] = (venue_summary['総回収額'] / venue_summary['総投資額']) * 100
    
    # 結果を表示
    print("\n各会場ごとの回収率:")
    for index, row in venue_summary.iterrows():
        print(f"会場コード: {row['会場コード']}, 総投資額: {row['総投資額']:.2f}円, 総回収額: {row['総回収額']:.2f}円, 回収率: {row['回収率(%)']:.2f}%")

def correlation_analysis(trifecta_bets, data):
    """
    特徴量と回収率の相関を分析します。
    Args:
        trifecta_bets (DataFrame): 全ての賭け金と回収金額を含むDataFrame
        data (DataFrame): 元データ（特徴量を含む）
    """
    # 'レースID'をキーにしてデータをマージ
    merged_data = pd.merge(trifecta_bets, data, on='レースID', how='left')
    
    # 回収率を計算
    merged_data['回収率'] = merged_data['Payout'] / merged_data['Bet_Amount']
    
    # 数値特徴量のみを対象とする
    numeric_features = merged_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # '回収率' を特徴量に追加
    if '回収率' not in numeric_features:
        numeric_features.append('回収率')
    
    # 相関行列を計算
    correlation_matrix = merged_data[numeric_features].corr()
    
    # 確認用に相関_matrixのカラムを表示
    print("Correlation Matrix Columns:", correlation_matrix.columns.tolist())
    
    # 回収率との相関を表示
    if '回収率' in correlation_matrix.columns:
        correlation_with_return = correlation_matrix['回収率'].drop('回収率')
        
        # もし Series でない場合（例えば複数列が存在する場合）には修正
        if isinstance(correlation_with_return, pd.DataFrame):
            # 複数列が存在する場合、最初の列を選択
            correlation_with_return = correlation_with_return.iloc[:, 0]
            print("警告: '回収率' 列が複数存在するため、最初の列のみを使用しています。")
        
        # Series であることを確認
        if isinstance(correlation_with_return, pd.Series):
            correlation_with_return = correlation_with_return.sort_values(ascending=False)
            print("\n特徴量と回収率の相関:")
            print(correlation_with_return)
        else:
            print("エラー: '回収率' 列の相関が正しく計算できませんでした。")
    else:
        print("'回収率' 列が相関行列に存在しません。")
    
    # 相関ヒートマップのプロット
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('特徴量と回収率の相関ヒートマップ')
    plt.tight_layout()
    plt.savefig('correlation_heatmap_allvenue.png')
    plt.show()
    print("相関ヒートマップを 'correlation_heatmap.png' として保存しました。")

def visualize_conditions(trifecta_bets, data):
    """
    条件ごとの回収率を可視化します。
    Args:
        trifecta_bets (DataFrame): 全ての賭け金と回収金額を含むDataFrame
        data (DataFrame): 元データ（特徴量を含む）
    """
    merged_data = pd.merge(trifecta_bets, data, on='レースID', how='left')
    
    # 回収率を計算
    merged_data['回収率'] = merged_data['Payout'] / merged_data['Bet_Amount']
    
    # 天候ごとの回収率の箱ひげ図
    if 'weather' in merged_data.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='weather', y='回収率', data=merged_data)
        plt.title('天候ごとの回収率の分布')
        plt.xlabel('天候')
        plt.ylabel('回収率')
        plt.tight_layout()
        plt.savefig('return_rate_by_weather_allvenue.png')
        plt.show()
        print("天候ごとの回収率の分布を 'return_rate_by_weather.png' として保存しました。")
    
    # 会場ごとの回収率の箱ひげ図
    if '会場' in merged_data.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='会場コード', y='回収率', data=merged_data)
        plt.title('会場ごとの回収率の分布')
        plt.xlabel('会場コード')
        plt.ylabel('回収率')
        plt.tight_layout()
        plt.savefig('return_rate_by_venue_allvenue.png')
        plt.show()
        print("会場ごとの回収率の分布を 'return_rate_by_venue.png' として保存しました。")

# 以下、省略...

def load_trifecta_odds(base_dir, start_date, end_date):
    """
    指定されたディレクトリから三連単オッズデータを読み込み、1つのDataFrameに統合します。
    Args:
        base_dir (str): 三連単オッズデータが保存されているベースディレクトリ
        start_date (datetime): 開始日
        end_date (datetime): 終了日
    Returns:
        DataFrame: 統合された三連単オッズデータ
    """
    all_data = []
    current_date = start_date
    total_days = (end_date - start_date).days + 1
    pbar = tqdm(total=total_days, desc="Loading Trifecta Odds")

    while current_date <= end_date:
        date_str = current_date.strftime('%y%m%d')  # 修正箇所
        yymm = current_date.strftime('%y%m')  # フォルダ名が 'yymm' の場合
        file_name = f'trifecta_{date_str}.csv'
        file_path = os.path.join(base_dir, yymm, file_name)
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # 生成されたDataFrameにレースIDを追加
                df['レースID'] = df.apply(
                    lambda row: f"{current_date.strftime('%Y%m%d')}{int(row['JCD']):02d}{int(row['Race']):02d}",
                    axis=1
                )
                all_data.append(df)
                print(f"Loaded {file_path} with {len(df)} records.")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
        
        current_date += timedelta(days=1)
        pbar.update(1)
    
    pbar.close()
    
    if all_data:
        trifecta_odds_data_df = pd.concat(all_data, ignore_index=True)
        print(f"Total trifecta odds records loaded: {len(trifecta_odds_data_df)}")
        return trifecta_odds_data_df
    else:
        print("No trifecta odds data loaded.")
        return pd.DataFrame()
    
# 三連単の確率を計算する関数
def calculate_trifecta_probability(p1, p2, p3):
    """
    三連単の確率を計算します。
    Args:
        p1 (float): 1着の確率
        p2 (float): 2着の確率
        p3 (float): 3着の確率
    Returns:
        float: 三連単の確率
    """
    denominator1 = 1 - p1
    denominator2 = 1 - p1 - p2
    denominator3 = 1 - p1 - p2 - p3
    if denominator1 <= 0 or denominator2 <= 0:
        return 0
    return (p1 * p2 * p3) 

# ケリー基準の計算関数を定義
def calculate_kelly_criterion(prob, odds):
    """
    ケリー基準に基づく最適な掛け金割合を計算します。
    prob: 予測確率（'prob_0'）
    odds: オッズ（デシマルオッズ）
    """
    if odds <= 1:
        return 0  # オッズが1以下の場合は掛け金割合を0とする
    else:
        f = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
    return max(0, f)  # 掛け金割合は0以上とする

def round_bet_amount(amount, unit=100):
    """
    bet_amount を指定された単位（デフォルトは100円）の倍数に丸めます。
    ただし、0より大きい場合は最低でも指定単位に設定します。
    """
    if amount <= 0:
        return 0
    else:
        return int(np.ceil(amount / unit)) * unit

if __name__ == '__main__':
    inspect_model()
