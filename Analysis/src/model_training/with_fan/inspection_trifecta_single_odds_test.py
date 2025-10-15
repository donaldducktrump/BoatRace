# model.py
import matplotlib.dates as mdates  # 追加
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, preprocess_data1min,load_before_data, merge_before_data, load_before1min_data
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

def load_processed_odds_data():
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
        gbm = lgb.Booster(model_file='boatrace_model_first_test.txt')
        gbm_odds = lgb.Booster(model_file='boatrace_odds_model_inv_test.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_first_test.csv').squeeze().tolist()
    feature_names_odds = pd.read_csv('feature_names_odds_inv_test.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2411']
    date_files = {
        # '2409': [f'2409{day:02d}' for day in range(1, 31)],
        # '2410': [f'2410{day:02d}' for day in range(16, 31)],
        '2411': [f'2411{day:02d}' for day in range(4,6)],
    }

    # データを結合
    data_list = []
    odds_list1 = []

    for month in month_folders:
        for date in date_files[month]:
            try:
                b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'
                # k_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_data/{month}/K{date}.TXT'
                # before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'
                before1min_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data_1min/{month}/beforeinfo1min_{date}.txt'

                b_data = load_b_file(b_file)
                # k_data, odds_list_part = load_k_file(k_file)
                # before_data = load_before_data(before_file)
                before1min_data = load_before_data(before1min_file)

                if not b_data.empty and not before1min_data.empty:
                    # データのマージ
                    # data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
                    data = b_data
                    # before_data = remove_common_columns(
                    #     data, before_data, on_columns=['選手登番', 'レースID', '艇番']
                    # )
                    # data = merge_before_data(data, before_data)
                    before1min_data = remove_common_columns(
                        data, before1min_data, on_columns=['選手登番', 'レースID', '艇番']
                    )
                    data = merge_before_data(data, before1min_data)
                    data_list.append(data)
                    # if not odds_list_part.empty:
                        # odds_list1.append(odds_list_part)
                else:
                    print(f"データが不足しています: {date}")
            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file},  または {before1min_file}")

    if not data_list:
        print("データが正しく読み込めませんでした。")
        return

    data_odds = load_processed_odds_data()

    # 全データを結合
    data = pd.concat(data_list, ignore_index=True)
    # odds_list = pd.concat(odds_list1, ignore_index=True)

    # ファンデータの読み込みとマージ
    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    # 必要な列を指定してマージ
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別',
                      '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(
        data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left'
    )
    pd.set_option('display.max_columns', 100)
    print(data[data['ET'].notna()])
    print(data[data['レースID']=='202411040203']['win_odds'])
    print(data[data['レースID']=='202411040905']['win_odds'])
    # data.rename(columns={'win_odds1min':'win_odds','place_odds1min':'place_odds'},inplace=True)

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
    plt.savefig('prob_0_distribution_trifecta_single_odds.png')
    plt.show()
    print("プロットが 'prob_0_distribution_trifecta_single_odds.png' として保存されました。")

    # オッズ予測のためのデータ準備
    # data_odds の読み込み
    data_odds = load_processed_odds_data()

    # 'win_odds_mean' を計算してマージ
    data_odds_grouped = data_odds.groupby(['選手登番', '艇番'], as_index=False)['win_odds_mean'].mean()
    data['選手登番'] = data['選手登番'].astype(int)
    data_odds_grouped['選手登番'] = data_odds_grouped['選手登番'].astype(int)
    data = data.merge(data_odds_grouped, on=['選手登番', '艇番'], how='left')

    # 必要な特徴量を選択
    features_odds = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                    '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                    'ET', 'tilt', 'EST','ESC',
                    'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                    'win_odds1min', 'prob_0','place_odds1min','win_odds_mean', '性別', '勝率', '複勝率', '優勝回数', '優出回数','前期能力指数', '今期能力指数','平均スタートタイミング']
    
    data.rename(columns={'win_odds':'win_odds1min','place_odds':'place_odds1min'},inplace=True)
    # data=data.dropna()
    X_odds = data[features_odds].copy()
    # 'win_odds1min', 'place_odds1min', 'win_odds_mean' 列を数値型に変換
    data['win_odds1min'] = pd.to_numeric(data['win_odds1min'], errors='coerce')
    data['place_odds1min'] = pd.to_numeric(data['place_odds1min'], errors='coerce')
    data['win_odds_mean'] = pd.to_numeric(data['win_odds_mean'], errors='coerce')


    # 欠損値やゼロを処理
    data['win_odds1min'] = np.where(data['win_odds1min']==0, data['win_odds_mean'], data['win_odds1min'])
    data['place_odds1min'] = np.where(data['place_odds1min']==0, data['win_odds_mean'], data['place_odds1min'])
    data['win_odds1min'] = 0.725 / data['win_odds1min']
    data['place_odds1min'] = 0.725 / data['place_odds1min']
    data['win_odds_mean'] = 0.725 / data['win_odds_mean']

    X_odds['win_odds1min'] = data['win_odds1min']
    X_odds['place_odds1min'] = data['place_odds1min']
    X_odds['win_odds_mean'] = data['win_odds_mean']

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features_odds = ['会場', 'weather','wind_d', 'ESC', '艇番', '性別', '支部', '級別']
    X_categorical_odds = pd.get_dummies(X_odds[categorical_features_odds], drop_first=True)

    # 数値データの選択
    numeric_features_odds = [col for col in X_odds.columns if col not in categorical_features_odds]
    X_numeric_odds = X_odds[numeric_features_odds]

    # 特徴量の結合
    X_processed_odds = pd.concat([X_numeric_odds.reset_index(drop=True), X_categorical_odds.reset_index(drop=True)], axis=1)

    # 特徴量の順序を合わせる
    for col in feature_names_odds:
        if col not in X_processed_odds.columns:
            X_processed_odds[col] = 0
    X_processed_odds = X_processed_odds[feature_names_odds]

    # オッズの予測
    odds_inv_pred = gbm_odds.predict(X_processed_odds)
    data['odds_inv_pred'] = odds_inv_pred

    # 三連単オッズデータの読み込み
    base_dir = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\trifecta_odds1min'
    start_date = datetime.strptime('2024-09-01', '%Y-%m-%d')
    end_date = datetime.strptime('2024-11-05', '%Y-%m-%d')
    total_capital = 100000  # 総資金（円）
    unit = 100  # 賭け金の単位（円）
    prob_threshold = 0  # 賭ける確率の閾値

    # 三連単オッズデータの読み込み
    date = '2024-11-04'
    date_str = '20241104'
    trifecta_odds_data_df = load_trifecta_odds(date,16,12)

    if trifecta_odds_data_df.empty:
        print("No trifecta odds data available. Exiting.")
        return

    # 各艇の確率データ
    data_boats = data[['レースID', '艇番', 'odds_inv_pred', '会場']].copy()
    # data_boats = data_boats[data_boats['レースID']=='202411040905']

    trifecta_bet = process_race('202411041612', trifecta_odds_data_df, data_boats, total_capital, unit, prob_threshold)
    
    print(f'trifecta_bet: {trifecta_bet}')
    # EV計算のための減算値のリスト
    ev_subtract_values = [4.5]

    # 結果を保存するリスト
    cumulative_returns = {}
    summary_results = []

    # for ev_value in ev_subtract_values:
    #     print(f"\n=== EV計算の減算値: {ev_value} ===")
    #     # 三連単賭け金と回収金額の計算
    #     trifecta_bets, total_investment, total_payout, return_rate = calculate_trifecta_bets(
    #         trifecta_odds_data_df,
    #         data_boats,
    #         total_capital=total_capital,
    #         unit=unit,
    #         prob_threshold=prob_threshold,
    #         # ev_subtract=ev_value  # EV計算の減算値を追加
    #     )
    #     print(trifecta_bets)
    #     # 資金の推移を計算
    #     capital_evolution = calculate_capital_evolution(trifecta_bets)


    #     # シャープレシオと最大ドローダウンを計算
    #     sharpe_ratio = calculate_sharpe_ratio(capital_evolution, risk_free_rate=0.0)
    #     max_drawdown = calculate_max_drawdown(capital_evolution)

    #     # 結果を保存
    #     cumulative_returns[ev_value] = capital_evolution
    #     summary_results.append({
    #         'EV_Subtract': ev_value,
    #         'Total Investment (yen)': total_investment,
    #         'Total Payout (yen)': total_payout,
    #         'Return Rate (%)': return_rate,
    #         'Sharpe Ratio': sharpe_ratio,
    #         'Max Drawdown (%)': max_drawdown
    #     })

    # # 結果の可視化
    # plot_cumulative_returns(cumulative_returns)

    # # 結果の比較表を作成
    # summary_df = pd.DataFrame(summary_results)
    # print("\n結果の比較:")
    # print(summary_df)

    # # 結果をCSVに保存
    # summary_df.to_csv('trifecta_summary_comparison_single_odds.csv', index=False)
    # print("結果の比較を 'trifecta_summary_comparison_single_odds.csv' として保存しました。")

    # # ------------- 追加部分開始 -------------
    # for ev_value in ev_subtract_values:

    #     print(f"\n=== 日付ごとの収益および投資金額の計算 (EV_Subtract={ev_value}) ===")
    #     trifecta_bets = cumulative_returns[ev_value]

    #     # trifecta_bets に '会場' 情報を追加するために 'data' とマージ
    #     data_unique = data[['レースID', '会場']].drop_duplicates(subset='レースID')
    #     trifecta_bets = trifecta_bets.merge(data_unique, on='レースID', how='left')
    #     # print(trifecta_bets[trifecta_bets['Payout'] > 0])
    #     pd.set_option('display.max_rows', 200)
    #     print(trifecta_bets.tail(100))
    #     # 日付ごとの収益を計算
    #     revenue_per_date = trifecta_bets.groupby('Date').apply(
    #         lambda df: df['Payout'].sum() - df['Bet_Amount'].sum()
    #     ).reset_index(name='Net_Revenue')

    #     # 日付ごとの投資金額を計算
    #     investment_per_date = trifecta_bets.groupby('Date')['Bet_Amount'].sum().reset_index()

    #     # 日付ごとの収益をプロット（棒グラフ）
    #     plt.figure(figsize=(12, 6))
    #     sns.barplot(x='Date', y='Net_Revenue', data=revenue_per_date, color='green')
    #     plt.title(f'日付ごとの収益 (EV_Subtract={ev_value})')
    #     plt.xlabel('日付')
    #     plt.ylabel('収益 (円)')
    #     plt.gcf().autofmt_xdate()
    #     plt.xticks(rotation=45)
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f'revenue_per_date_ev_{ev_value}_odds.png')
    #     plt.show()
    #     print(f"日付ごとの収益を 'revenue_per_date_ev_{ev_value}_odds.png' として保存しました。")

    #     # 1日当たりの投資金額を計算
    #     investment_per_date = trifecta_bets.groupby('Date')['Bet_Amount'].sum().reset_index()

    #     # 統計量の計算
    #     min_investment = investment_per_date['Bet_Amount'].min()
    #     max_investment = investment_per_date['Bet_Amount'].max()
    #     average_investment = investment_per_date['Bet_Amount'].mean()
    #     median_investment = investment_per_date['Bet_Amount'].median()

    #     # 統計結果の表示
    #     print("\n1日当たりの投資金額の統計量:")
    #     print(f"最小値: {min_investment:.2f} 円")
    #     print(f"最大値: {max_investment:.2f} 円")
    #     print(f"平均値: {average_investment:.2f} 円")
    #     print(f"中央値: {median_investment:.2f} 円")

    #     # 1日当たりの投資金額のヒストグラムをプロット
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(investment_per_date['Bet_Amount'], bins=30, kde=True, color='purple')
    #     plt.title('1日当たりの投資金額のヒストグラム')
    #     plt.xlabel('投資金額 (円)')
    #     plt.ylabel('頻度')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f'investment_per_date_histogram_ev_{ev_value}_odds.png')
    #     plt.show()
    #     print(f"1日当たりの投資金額のヒストグラムを 'investment_per_date_histogram_ev_{ev_value}_odds.png' として保存しました。")

    #     # 会場ごとの回収率を計算
    #     venue_summary = trifecta_bets.groupby('会場').agg(
    #         Total_Payout=('Payout', 'sum'),
    #         Total_Bet=('Bet_Amount', 'sum')
    #     ).reset_index()
    #     venue_summary['Return_Rate (%)'] = (venue_summary['Total_Payout'] / venue_summary['Total_Bet']) * 100
    #     venue_summary = venue_summary[['会場', 'Return_Rate (%)']]

    #     # 会場ごとの回収率をCSVに保存
    #     venue_summary.to_csv(f'return_rate_per_venue_ev_{ev_value}_odds.csv', index=False)
    #     print(f"会場ごとの回収率を 'return_rate_per_venue_ev_{ev_value}_odds.csv' として保存しました。")
    # # ------------- 追加部分終了 -------------

def process_race(race_id, race_odds, data_boats, total_capital=100000, unit=500, prob_threshold=0.0, ev_subtract=1.4):
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

    if boats_data.empty:
        # データがない場合
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'Bet_amount']
        return pd.DataFrame(columns=columns)

    boats_probs = boats_data.set_index('艇番')['odds_inv_pred'].to_dict()
    # 正規化
    total_odds_inv_pred = sum(boats_probs.values())
 
    if total_odds_inv_pred == 0:
        # 合計が0の場合、計算できないのでスキップ
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'Bet_amount']
        return pd.DataFrame(columns=columns)
    boats_probs = {k: v / total_odds_inv_pred for k, v in boats_probs.items()}
    
    print(f'boats_probs : {boats_probs}')

    # 確率が高い順に上位4艇を選択
    top_boats = boats_data.sort_values(by='odds_inv_pred', ascending=False)['艇番'].head(4).tolist()
    print(top_boats)

    if len(top_boats) < 3:
        # 上位4艇が3艇未満の場合
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'Bet_amount']
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
    merged_df['EV'] = (merged_df['Probability'] * merged_df['Odds']) - 4.5
    # print(merged_df[merged_df['レースID'] == '202411032412'])
    # Filter based on EV
    print(f'merged_df : {merged_df}')
    merged_df = merged_df[merged_df['EV'] > 0]

    if merged_df.empty:
        # 賭けがなかった場合
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'Bet_Amount']
        return pd.DataFrame(columns=columns)

    # Assign bet amount (fixed unit per bet)
    merged_df['Bet_Amount'] = unit  # 固定ベット金額

    # # Determine the actual combination
    # boats_actual = boats_data.sort_values(by='着')['艇番'].tolist()
    # if len(boats_actual) < 3:
    #     actual_combo = ()
    # else:
    #     actual_combo = tuple(boats_actual[:3])

    # # Calculate payout
    # merged_df['Payout'] = merged_df.apply(
    #     lambda row: row['Bet_Amount'] * row['Odds'] if (row['Boat1'], row['Boat2'], row['Boat3']) == actual_combo else 0,
    #     axis=1
    # )

    # Assign race_id
    merged_df['レースID'] = race_id

    # Select and reorder columns
    trifecta_bet = merged_df[['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'Bet_Amount']]

    return trifecta_bet

# メイン処理関数
def calculate_trifecta_bets(trifecta_odds_df, data_boats, total_capital=100000, unit=500, prob_threshold=0.0):
    """
    全レースに対して三連単のベット金額を決定し、回収率や資金の推移を計算します。
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
    # ユニークなレースIDを取得
    race_ids = trifecta_odds_df['レースID'].unique()
    
    all_trifecta_bets = []
    
    print("Processing races and calculating bets...")
    for race_id in tqdm(race_ids, desc="Processing Races"):
        race_odds = trifecta_odds_df[trifecta_odds_df['レースID'] == race_id]
        trifecta_bet = process_race(race_id, race_odds, data_boats, 
                                   total_capital=total_capital, 
                                   unit=unit, 
                                   prob_threshold=prob_threshold)
        if not trifecta_bet.empty:
            all_trifecta_bets.append(trifecta_bet)
            # 最初の数レースの結果を表示（ここでは最初の3レースに限定）
            if len(all_trifecta_bets) <= 3:
                print(f"\nレースID: {race_id} の賭け結果:")
                print(trifecta_bet)
    
    if all_trifecta_bets:
        trifecta_bets = pd.concat(all_trifecta_bets, ignore_index=True)
    else:
        # 全レースで賭けがなかった場合、必要な列を持つ空のDataFrameを作成
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'bet_amount']
        trifecta_bets = pd.DataFrame(columns=columns)
    
    # 日付情報の抽出
    if not trifecta_bets.empty:
        trifecta_bets['Date'] = trifecta_bets['レースID'].apply(lambda x: datetime.strptime(x[:8], '%Y%m%d'))
    else:
        trifecta_bets['Date'] = pd.to_datetime([])  # 空のDatetime列
    
    # 総投資額と総回収額を計算
    if not trifecta_bets.empty:
        total_investment = trifecta_bets['bet_amount'].sum()
        # total_payout = trifecta_bets['Payout'].sum()
        # return_rate = (total_payout / total_investment) * 100 if total_investment > 0 else 0
    else:
        total_investment = 0
        total_payout = 0
        return_rate = 0
    
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
    plt.savefig('cumulative_return_comparison_single_odds.png')
    plt.show()
    print("資金の推移を 'cumulative_return_comparison_single_odds.png' として保存しました。")

def load_trifecta_odds(date, place_cd, race_no):
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')  # 例: '2023-11-01' 形式の文字列を変換
    month = date.strftime('%y%m')
    date_str = date.strftime('%y%m%d')
    file_path = os.path.join(r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\trifecta_odds1min', month, f'trifecta1min_{date_str}.txt')
    
    # ファイルが存在しない場合はエラー
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    # 読み込みとフィルタリング
    trifecta_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        recording = False
        for line in f:
            if f"Place: {place_cd}, Race: {race_no}" in line:
                recording = True
            elif recording and line.strip() == "":
                break
            elif recording:
                parts = line.strip().split(',')
                boat1 = int(parts[0].split(': ')[1])
                boat2 = int(parts[1].split(': ')[1])
                boat3 = int(parts[2].split(': ')[1])
                odds = float(parts[3].split(': ')[1])
                trifecta_data.append({'Boat1': boat1, 'Boat2': boat2, 'Boat3': boat3, 'Odds': odds})
    
    trifecta_df = pd.DataFrame(trifecta_data)
    # trifecta_df['レースID'] = trifecta_df.apply(
    #     lambda row: f"{date_str}{place_cd:02d}{race_no:02d}",
    #     axis=1
    # )
    print(trifecta_df.head(120))
    print(f"Loaded {len(trifecta_df)} trifecta odds entries for Place {place_cd}, Race {race_no} on {date_str}")
    return trifecta_df

def calculate_trifecta_probability(p1, p2, p3):
    """
    三連単の確率を計算します。
    """
    # 各値の合計を計算
    # total = p1 + p2 + p3
    # if total == 0:
    #     return 0
    # p1 = p1 / total
    # p2 = p2 / total
    # p3 = p3 / total
    denominator1 = 1 - p1
    denominator2 = 1 - p1 - p2
    if denominator1 <= 0 or denominator2 <= 0:
        return 0
    return (p1 * p2 * p3/denominator1/denominator2) 

if __name__ == '__main__':
    inspect_model()
