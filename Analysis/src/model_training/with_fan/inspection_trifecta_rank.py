# model_1min.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
from catboost import CatBoostRanker, Pool

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
    
    
def calculate_rank_probabilities(scores_df, race_id_col='レースID', boat_col='艇番', score_col='pred_score'):
    """
    各レースにおける各艇の各順位になる確率を計算する関数。
    
    Parameters:
        scores_df (pd.DataFrame): 各艇のスコアが含まれるデータフレーム。
        race_id_col (str): レースIDを示すカラム名。
        boat_col (str): 各艇を識別するカラム名。
        score_col (str): 各艇のスコアを示すカラム名。
        
    Returns:
        pd.DataFrame: 各艇の各順位になる確率を含むデータフレーム。
    """
    # 結果を格納するリスト
    results = []

    # レースごとに処理
    for race_id, group in scores_df.groupby(race_id_col):
        boats = group[boat_col].values
        scores = group[score_col].values

        # ソフトマックス関数を使用して各艇の選択確率を計算
        exp_scores = np.exp(scores)
        probabilities = exp_scores / exp_scores.sum()

        # 各艇の順位1の確率を記録
        for boat, prob in zip(boats, probabilities):
            results.append({
                race_id_col: race_id,
                boat_col: boat,
                '順位': 1,
                '確率': prob
            })

        # 残りの順位2〜6を計算
        remaining_boats = list(boats)
        remaining_scores = list(scores)
        for rank in range(2, 7):  # 2位から6位まで
            if not remaining_boats:
                break  # 残りの艇がない場合

            # ソフトマックスで確率計算
            exp_scores = np.exp(remaining_scores)
            probabilities = exp_scores / np.sum(exp_scores)

            for boat, prob in zip(remaining_boats, probabilities):
                results.append({
                    race_id_col: race_id,
                    boat_col: boat,
                    '順位': rank,
                    '確率': prob
                })

            # 最も確率が高い艇を選択して残りから除外
            top_boat_idx = np.argmax(probabilities)
            selected_boat = remaining_boats.pop(top_boat_idx)
            remaining_scores.pop(top_boat_idx)

    # データフレームに変換
    probability_df = pd.DataFrame(results)

    # 必要に応じてピボットテーブルに変換
    probability_pivot = probability_df.pivot_table(index=[race_id_col, boat_col],
                                                   columns='順位',
                                                   values='確率',
                                                   fill_value=0).reset_index()

    # 列名を調整
    probability_pivot.columns.name = None
    probability_pivot = probability_pivot.rename(columns=lambda x: f'順位_{int(x)}_確率' if isinstance(x, int) else x)

    return probability_pivot


def inspect_model():
    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_trio.txt')
        gbm_odds = lgb.Booster(model_file='boatrace_odds_model.txt')
        ranker = CatBoostRanker()
        ranker.load_model('boatrace_ranker_catboost.cbm')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_ranker_catboost.csv', header=None).squeeze().tolist()
    feature_names_odds = pd.read_csv('feature_names_odds.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2405', '2406', '2407', '2408', '2409', '2410']
    date_files = {
        # '2311': [f'2311{day:02d}' for day in range(1, 31)],  # 231101 - 231130
        # '2312': [f'2312{day:02d}' for day in range(1, 32)],  # 231201 - 231231
        # '2401': [f'2401{day:02d}' for day in range(1, 32)],  # 240101 - 240131
        # '2402': [f'2402{day:02d}' for day in range(1, 29)],  # 240201 - 240228
        # '2403': [f'2403{day:02d}' for day in range(1, 15)],  # 240301 - 240331
        # '2404': [f'2404{day:02d}' for day in range(1, 31)],  # 240401 - 240430
        '2405': [f'2405{day:02d}' for day in range(1, 32)],  # 240501 - 240531
        '2406': [f'2406{day:02d}' for day in range(1, 31)],  # 240601 - 240630
        '2407': [f'2407{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(1,30)],   # 241001 - 241008
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



    # output_dir = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\trifecta_data'
    # data = pd.read_csv(output_dir + '/final_data.csv')

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #     print(f"出力ディレクトリを作成しました: {output_dir}")
    # final_output_file = os.path.join(output_dir, 'final_data.csv')
    # data.to_csv(final_output_file, index=False)
    # print(f"最終データを保存しました: {final_output_file}")



    # print(data.head(50))
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
    # if '天候_雪' not in X_processed.columns:
    #     X_processed['天候_雪'] = 0
    for col in feature_names:
        if col not in X_processed.columns:
            X_processed[col] = 0
    # print("X_processed columns:", X_processed.columns.tolist())
    X_processed = X_processed[feature_names]  # 学習時の特徴量と順序を合わせる

    y_pred = ranker.predict(X_processed)
    data['pred_score'] = y_pred
    
    probabilities = calculate_rank_probabilities(data)
    print(probabilities)
    data = data.merge(probabilities, on=['レースID', '艇番'], how='left')

    # data['pred_score'] = y_pred[:, 0]  # 1,2着の確率
    # data['prob_1'] = y_pred[:, 1]
    # data['prob_2'] = y_pred[:, 2]

    # data['pred_score'] = y_pred[:, 0]  # 1,着の確率
    # print(data[['pred_score', '着']].head(50))

    data['prob_0'] =data['順位_1_確率']
    data['prob_1'] =  data['順位_2_確率']
    data['prob_2'] =  data['順位_3_確率']

  # 2. 艇番ごとの実際のオッズと予測オッズの散布図
    boat_col = '艇番'
    score_col = 'prob_0'
    boats = data[boat_col].unique()
    palette = sns.color_palette("husl", len(boats))
    bins = 50
    alpha = 0.5
    for boat, color in zip(boats, palette):
        sns.histplot(
            data[data[boat_col] == boat][score_col],
            bins=bins,
            kde=False,
            stat="density",
            label=f'艇番 {boat}',
            color=color,
            alpha=alpha
        )
    
    plt.title('各艇のpred_score分布の重ね合わせヒストグラム')
    plt.xlabel('pred_score')
    plt.ylabel('密度')
    plt.legend(title='艇番')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('pred_score_hist_rank_cat.png')
    plt.show()


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
    start_date = datetime.strptime('2024-05-01', '%Y-%m-%d')
    end_date = datetime.strptime('2024-05-30', '%Y-%m-%d')
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
    
    data_boats = data[['レースID', '艇番', 'pred_score','prob_0','prob_1','prob_2', '着']].copy()
    
    # 三連単賭け金と回収金額の計算
    trifecta_bets, total_investment, total_payout, return_rate = calculate_trifecta_bets(
        trifecta_odds_data_df, 
        data_boats, 
        total_capital=total_capital, 
        unit=unit, 
        prob_threshold=prob_threshold
    )
    
    # if trifecta_bets.empty:
    #     print("No trifecta bets were placed.")
    #     return
    
    # 資金の推移を計算
    capital_evolution = calculate_capital_evolution(trifecta_bets)
    
    # 結果の可視化
    plot_results(trifecta_bets, capital_evolution)
    
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

import itertools
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
    # merged_df = merged_df[merged_df['EV'] > 0]
    
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

# メイン処理関数
def calculate_trifecta_bets(trifecta_odds_df, data_boats, total_capital=100000, unit=100, prob_threshold=0.0):
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
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'Bet_Amount', 'Payout']
        trifecta_bets = pd.DataFrame(columns=columns)
    
    # 日付情報の抽出
    if not trifecta_bets.empty:
        trifecta_bets['Date'] = trifecta_bets['レースID'].apply(lambda x: datetime.strptime(x[:8], '%Y%m%d'))
    else:
        trifecta_bets['Date'] = pd.to_datetime([])  # 空のDatetime列
    
    # 総投資額と総回収額を計算
    if not trifecta_bets.empty:
        total_investment = trifecta_bets['Bet_Amount'].sum()
        total_payout = trifecta_bets['Payout'].sum()
        return_rate = (total_payout / total_investment) * 100 if total_investment > 0 else 0
    else:
        total_investment = 0
        total_payout = 0
        return_rate = 0
    
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
    
    # 日付順にソート
    trifecta_bets_sorted = trifecta_bets.sort_values(by=['Date']).reset_index(drop=True)
    trifecta_bets_sorted['Cumulative_Investment'] = trifecta_bets_sorted['Bet_Amount'].cumsum()
    trifecta_bets_sorted['Cumulative_Payout'] = trifecta_bets_sorted['Payout'].cumsum()
    trifecta_bets_sorted['Net'] = trifecta_bets_sorted['Cumulative_Payout'] - trifecta_bets_sorted['Cumulative_Investment']
    trifecta_bets_sorted['Capital'] = initial_capital + trifecta_bets_sorted['Net']
    return trifecta_bets_sorted

# 結果の可視化関数
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
        plt.figure(figsize=(12, 6))
        plt.plot(capital_evolution['Date'], capital_evolution['Capital'], marker='o', linestyle='-', color='blue')
        plt.title("資金の推移")
        plt.xlabel("日付")
        plt.ylabel("資金 (円)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('capital_evolution_trifecta.png')
        plt.show()
        print("資金の推移が 'capital_evolution_trifecta.png' として保存されました。")
    
    # 回収金額のヒストグラム
    if not trifecta_bets.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(trifecta_bets['Payout'], bins=50, kde=True, color='green')
        plt.title("回収金額の分布")
        plt.xlabel("回収金額 (円)")
        plt.ylabel("度数")
        plt.grid(True)
        plt.savefig('payout_distribution_trifecta.png')
        plt.show()
        print("回収金額の分布が 'payout_distribution_trifecta.png' として保存されました。")
    
        # 賭け金額のヒストグラム
        plt.figure(figsize=(10, 6))
        sns.histplot(trifecta_bets['Bet_Amount'], bins=50, kde=True, color='orange')
        plt.title("賭け金額の分布")
        plt.xlabel("賭け金額 (円)")
        plt.ylabel("度数")
        plt.grid(True)
        plt.savefig('bet_amount_distribution_trifecta.png')
        plt.show()
        print("賭け金額の分布が 'bet_amount_distribution_trifecta.png' として保存されました。")
    
        # 投資金額と回収金額の比較プロット
        plt.figure(figsize=(12, 6))
        trifecta_bets_sorted = trifecta_bets.sort_values(by=['Date'])
        plt.bar(trifecta_bets_sorted['Date'], trifecta_bets_sorted['Bet_Amount'], label='投資金額', alpha=0.6, color='red')
        plt.bar(trifecta_bets_sorted['Date'], trifecta_bets_sorted['Payout'], label='回収金額', alpha=0.6, color='blue')
        plt.title("投資金額と回収金額の比較")
        plt.xlabel("日付")
        plt.ylabel("金額 (円)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('investment_vs_payout_trifecta.png')
        plt.show()
        print("投資金額と回収金額の比較が 'investment_vs_payout_trifecta.png' として保存されました。")
    
        # 日付を横軸とした累積収益の推移プロット
        if 'Net' in capital_evolution.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(capital_evolution['Date'], capital_evolution['Net'].cumsum(), marker='o', linestyle='-', color='purple')
            plt.title("日付を横軸とした累積収益の推移")
            plt.xlabel("日付")
            plt.ylabel("累積収益 (円)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('cumulative_profit_trifecta.png')
            plt.show()
            print("日付を横軸とした累積収益の推移が 'cumulative_profit_trifecta.png' として保存されました。")
        else:
            print("Net列が存在しないため、累積収益の推移をプロットできません。")

# メイン処理関数
# def calculate_trifecta_bets(trifecta_odds_df, data_boats, total_capital=100000, unit=100, prob_threshold=0.001):
#     """
#     全レースに対して三連単の賭け金額を決定し、回収率や資金の推移を計算します。
#     Args:
#         trifecta_odds_df (DataFrame): 三連単オッズのデータ
#         data_boats (DataFrame): 各レースの各艇の確率データ
#         total_capital (int): 総資金
#         unit (int): 賭け金の単位（円）
#         prob_threshold (float): この確率以下の組み合わせは賭けない
#     Returns:
#         DataFrame: 全ての賭け金と回収金額
#         float: 総投資額
#         float: 総回収額
#         float: 回収率
#     """
#     # ユニークなレースIDを取得
#     race_ids = trifecta_odds_df['レースID'].unique()
    
#     all_trifecta_bets = []
    
#     print("Processing races and calculating bets...")
#     for race_id in tqdm(race_ids, desc="Processing Races"):
#         race_odds = trifecta_odds_df[trifecta_odds_df['レースID'] == race_id]
#         trifecta_bet = process_race(race_id, race_odds, data_boats, 
#                                    total_capital=total_capital, 
#                                    unit=unit, 
#                                    prob_threshold=prob_threshold)
#         if not trifecta_bet.empty:
#             all_trifecta_bets.append(trifecta_bet)
#             # 最初の数レースの結果を表示（ここでは最初の3レースに限定）
#             if len(all_trifecta_bets) <= 3:
#                 print(f"\nレースID: {race_id} の賭け結果:")
#                 print(trifecta_bet)
    
#     if all_trifecta_bets:
#         trifecta_bets = pd.concat(all_trifecta_bets, ignore_index=True)
#     else:
#         # 全レースで賭けがなかった場合、必要な列を持つ空のDataFrameを作成
#         columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'Kelly_Fraction', 'Bet_Amount', 'Payout']
#         trifecta_bets = pd.DataFrame(columns=columns)
    
#     # 日付情報の抽出
#     if not trifecta_bets.empty:
#         trifecta_bets['Date'] = trifecta_bets['レースID'].apply(lambda x: datetime.strptime(x[:8], '%Y%m%d'))
#     else:
#         trifecta_bets['Date'] = pd.to_datetime([])  # 空のDatetime列
    
#     # 総投資額と総回収額を計算
#     if not trifecta_bets.empty:
#         total_investment = trifecta_bets['Bet_Amount'].sum()
#         total_payout = trifecta_bets['Payout'].sum()
#         return_rate = (total_payout / total_investment) * 100 if total_investment > 0 else 0
#     else:
#         total_investment = 0
#         total_payout = 0
#         return_rate = 0
    
#     return trifecta_bets, total_investment, total_payout, return_rate

# # 賭け金額を決定し、回収金額を計算する関数
# def process_race(race_id, race_odds, data_boats, total_capital=100000, unit=100, prob_threshold=0.001):
#     """
#     各レースごとに三連単の賭け金額を決定し、回収金額を計算します。
#     Args:
#         race_id (str): レースID
#         race_odds (DataFrame): 該当レースの三連単オッズデータ
#         data_boats (DataFrame): 各レースの各艇の確率データ
#         total_capital (int): 総資金
#         unit (int): 賭け金の単位（円）
#         prob_threshold (float): この確率以下の組み合わせは賭けない
#     Returns:
#         DataFrame: 各三連単の賭け金額と回収金額
#     """
#     # 該当レースの艇の確率データを取得
#     boats_probs = data_boats[data_boats['レースID'] == race_id].set_index('艇番')['pred_score'].to_dict()
    
#     # 該当レースの実際の結果を取得
#     actual_results = data_boats[data_boats['レースID'] == race_id].sort_values('着')['艇番'].tolist()
#     if len(actual_results) < 3:
#         # 実際の結果が3着未満の場合、回収金額を計算できない
#         actual_combo = ()
#     else:
#         actual_combo = tuple(actual_results[:3])
    
#     def calculate_bet(row):
#         # 型変換を明示的に行う
#         boat1 = int(row['Boat1']) if pd.notnull(row['Boat1']) and str(row['Boat1']).isdigit() else None
#         boat2 = int(row['Boat2']) if pd.notnull(row['Boat2']) and str(row['Boat2']).isdigit() else None
#         boat3 = int(row['Boat3']) if pd.notnull(row['Boat3']) and str(row['Boat3']).isdigit() else None
#         odds = float(row['Odds']) if pd.notnull(row['Odds']) else 0.0
        
#         # 各艇の確率を取得
#         p1 = boats_probs.get(boat1, 0)
#         p2 = boats_probs.get(boat2, 0)
#         p3 = boats_probs.get(boat3, 0)
        
#         # 三連単の確率を計算
#         prob = calculate_trifecta_probability(p1, p2, p3)
        
#         # 重複する艇番号を持つ組み合わせをスキップ
#         if len({boat1, boat2, boat3}) != 3:
#             return None
        
#         if prob < prob_threshold:
#             return None  # 確率が低い組み合わせは賭けない
        
#         # ケリー基準に基づいて掛け金割合を計算
#         kelly_fraction = calculate_kelly_criterion(prob, odds)
#         if kelly_fraction <= 0:
#             return None  # ケリー基準により賭けない
        
#         # 掛け金を計算
#         bet_amount = round_bet_amount(kelly_fraction * total_capital, unit)
#         if bet_amount < unit:
#             bet_amount = 0  # 最低賭け金を設定
        
#         # 回収金額を計算
#         if (boat1, boat2, boat3) == actual_combo:
#             payout = bet_amount * odds
#         else:
#             payout = 0
        
#         if bet_amount == 0:
#             return None  # 賭け金が0の場合はスキップ
        
#         return {
#             'レースID': race_id,
#             'Boat1': boat1,
#             'Boat2': boat2,
#             'Boat3': boat3,
#             'Probability': prob,
#             'Odds': odds,
#             'Kelly_Fraction': kelly_fraction,
#             'Bet_Amount': bet_amount,
#             'Payout': payout
#         }
    
#     # applyを使用してベットを計算
#     bets = race_odds.apply(calculate_bet, axis=1)
    
#     # Noneを除外し、辞書をリストに変換
#     trifecta_data = bets.dropna().tolist()
    
#     if not trifecta_data:
#         # 賭けがなかった場合でも列を持つ空のDataFrameを返す
#         columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'Kelly_Fraction', 'Bet_Amount', 'Payout']
#         return pd.DataFrame(columns=columns)
    
#     trifecta_df = pd.DataFrame(trifecta_data)
    
#     return trifecta_df

# # 資金の推移を計算する関数
# def calculate_capital_evolution(trifecta_bets, initial_capital=100000):
#     """
#     資金の推移を計算します。
#     Args:
#         trifecta_bets (DataFrame): 全ての賭け金と回収金額
#         initial_capital (int): 初期資金
#     Returns:
#         DataFrame: 資金の推移
#     """
#     if trifecta_bets.empty:
#         # 賭けがなかった場合、初期資金のみのDataFrameを返す
#         return pd.DataFrame({
#             'Cumulative_Investment': [0],
#             'Cumulative_Payout': [0],
#             'Net': [0],
#             'Capital': [initial_capital]
#         })
    
#     # 日付順にソート
#     trifecta_bets_sorted = trifecta_bets.sort_values(by=['Date']).reset_index(drop=True)
#     trifecta_bets_sorted['Cumulative_Investment'] = trifecta_bets_sorted['Bet_Amount'].cumsum()
#     trifecta_bets_sorted['Cumulative_Payout'] = trifecta_bets_sorted['Payout'].cumsum()
#     trifecta_bets_sorted['Net'] = trifecta_bets_sorted['Cumulative_Payout'] - trifecta_bets_sorted['Cumulative_Investment']
#     trifecta_bets_sorted['Capital'] = initial_capital + trifecta_bets_sorted['Net']
#     return trifecta_bets_sorted

# # 結果の可視化関数
# def plot_results(trifecta_bets, capital_evolution):
#     """
#     回収率や資金の推移をプロットします。
#     Args:
#         trifecta_bets (DataFrame): 全ての賭け金と回収金額
#         capital_evolution (DataFrame): 資金の推移
#     """
#     # 回収率の表示
#     total_investment = trifecta_bets['Bet_Amount'].sum() if not trifecta_bets.empty else 0
#     total_payout = trifecta_bets['Payout'].sum() if not trifecta_bets.empty else 0
#     return_rate = (total_payout / total_investment) * 100 if total_investment > 0 else 0
#     print(f"\n総投資額: {total_investment:.2f}円")
#     print(f"総回収額: {total_payout:.2f}円")
#     print(f"回収率: {return_rate:.2f}%")
    
#     # 最初の数レースの結果を表示
#     if not trifecta_bets.empty:
#         print("\n最初の数レースの詳細:")
#         print(trifecta_bets.head(5))
#     else:
#         print("\n賭けがなかったため、レースの詳細はありません。")
    
#     # 資金の推移プロット
#     if capital_evolution.empty or capital_evolution.shape[0] == 1 and capital_evolution['Capital'].iloc[0] == 100000:
#         print("\n資金の推移をプロットできません。")
#     else:
#         plt.figure(figsize=(12, 6))
#         plt.plot(capital_evolution['Date'], capital_evolution['Capital'], marker='o', linestyle='-', color='blue')
#         plt.title("資金の推移")
#         plt.xlabel("日付")
#         plt.ylabel("資金 (円)")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig('capital_evolution_trifecta.png')
#         plt.show()
#         print("資金の推移が 'capital_evolution_trifecta.png' として保存されました。")
    
#     # 回収金額のヒストグラム
#     if not trifecta_bets.empty:
#         plt.figure(figsize=(10, 6))
#         sns.histplot(trifecta_bets['Payout'], bins=50, kde=True, color='green')
#         plt.title("回収金額の分布")
#         plt.xlabel("回収金額 (円)")
#         plt.ylabel("度数")
#         plt.grid(True)
#         plt.savefig('payout_distribution_trifecta.png')
#         plt.show()
#         print("回収金額の分布が 'payout_distribution_trifecta.png' として保存されました。")
    
#         # 賭け金額のヒストグラム
#         plt.figure(figsize=(10, 6))
#         sns.histplot(trifecta_bets['Bet_Amount'], bins=50, kde=True, color='orange')
#         plt.title("賭け金額の分布")
#         plt.xlabel("賭け金額 (円)")
#         plt.ylabel("度数")
#         plt.grid(True)
#         plt.savefig('bet_amount_distribution_trifecta.png')
#         plt.show()
#         print("賭け金額の分布が 'bet_amount_distribution_trifecta.png' として保存されました。")
    
#         # 投資金額と回収金額の比較プロット
#         plt.figure(figsize=(12, 6))
#         trifecta_bets_sorted = trifecta_bets.sort_values(by=['Date'])
#         plt.bar(trifecta_bets_sorted['Date'], trifecta_bets_sorted['Bet_Amount'], label='投資金額', alpha=0.6, color='red')
#         plt.bar(trifecta_bets_sorted['Date'], trifecta_bets_sorted['Payout'], label='回収金額', alpha=0.6, color='blue')
#         plt.title("投資金額と回収金額の比較")
#         plt.xlabel("日付")
#         plt.ylabel("金額 (円)")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig('investment_vs_payout_trifecta.png')
#         plt.show()
#         print("投資金額と回収金額の比較が 'investment_vs_payout_trifecta.png' として保存されました。")
    
#         # 日付を横軸とした収益の推移プロット
#         if 'Net' in capital_evolution.columns:
#             plt.figure(figsize=(12, 6))
#             plt.plot(capital_evolution['Date'], capital_evolution['Net'].cumsum(), marker='o', linestyle='-', color='purple')
#             plt.title("日付を横軸とした累積収益の推移")
#             plt.xlabel("日付")
#             plt.ylabel("累積収益 (円)")
#             plt.grid(True)
#             plt.tight_layout()
#             plt.savefig('cumulative_profit_trifecta.png')
#             plt.show()
#             print("日付を横軸とした累積収益の推移が 'cumulative_profit_trifecta.png' として保存されました。")
#         else:
#             print("Net列が存在しないため、累積収益の推移をプロットできません。")


# サンプルデータの作成（実際のデータに置き換えてください）
def create_sample_data():
    """
    サンプルデータを作成します。実際のデータに置き換えて使用してください。
    """
    # 三連単オッズのサンプルデータ
    trifecta_odds_data = {
        'Date': ['2023-11-01'] * 10,
        'JCD': [15] * 10,
        'Race': [6] * 10,
        'Boat1': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        'Boat2': [2, 1, 1, 1, 1, 1, 3, 3, 3, 3],
        'Boat3': [3, 3, 2, 2, 2, 2, 4, 4, 4, 3],
        'Odds': [6.8, 29.4, 31.3, 57.3, 2331.0, 1100.0, 8.8, 34.6, 29.5, 47.6]
    }
    trifecta_odds_df = pd.DataFrame(trifecta_odds_data)
    
    # 各レースの各艇の確率データ
    boats_data = {
        'Date': ['2023-11-01'] * 4,
        'JCD': [15] * 4,
        'Race': [6] * 4,
        'Boat': [1, 2, 3, 4],
        'pred_score': [0.15, 0.10, 0.05, 0.07],
        '着': [2, 1, 3, 4]
    }
    data_boats = pd.DataFrame(boats_data)
    
    return trifecta_odds_df, data_boats

from datetime import datetime, timedelta
from tqdm import tqdm

# 三連単オッズデータの読み込み関数
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
    # if denominator1 <= 0:

        return 0
    # return (p1 * p2 * p3) / (denominator1 * denominator2)
    return (p1 * p2 * p3) 
# ケリー基準の計算関数を定義
def calculate_kelly_criterion(prob, odds):
    """
    ケリー基準に基づく最適な掛け金割合を計算します。
    prob: 予測確率（'pred_score'）
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
