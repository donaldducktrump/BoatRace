# inspection.py

import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, load_before_data, merge_before_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

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
    month_folders = ['2410']
    date_files = {
        # '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        # '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(16,20)],   # 241001 - 241008
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
    data = preprocess_data(data)

    print("data columns:", data.columns.tolist())
    print(data)

    # 特徴量の指定
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h']
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather', 'wind_d','ESC']
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
    y_pred = gbm.predict(X_processed)
    # data['prob_0'] = y_pred[:, 0]  # 1,2着の確率
    # data['prob_1'] = y_pred[:, 1]
    # data['prob_2'] = y_pred[:, 2]

    data['prob_0'] = y_pred[:, 0]  # 1,着の確率
    data['prob_1'] = y_pred[:, 1]

    # data['prob_0'] = y_pred[:, 0]  # 1,着の確率
    # data['prob_1'] = y_pred[:, 1]  # 2,着の確率
    # data['prob_2'] = y_pred[:, 2]  # 3,着の確率
    # data['prob_3'] = y_pred[:, 3]  # 4,着以下の確率

    # 'prob_0' の分布をプロット
    plt.figure(figsize=(8, 6))
    plt.hist(data['prob_0'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title("予測確率 'prob_0' の分布")
    plt.xlabel("予測確率 (prob_0)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('prob_0_distribution.png')
    plt.show()
    print("プロットが 'prob_0_distribution.png' として保存されました。")
    # correct_predictions = 0
    # total_races = 0
    # total_bet = 0
    # total_return = 0
    # payout_list = []

    # # オッズ情報をデータにマージ
    # if '艇番' not in odds_list.columns:
    #     odds_list['艇番'] = odds_list['枠番']  # 必要に応じて修正

    # オッズ情報をデータにマージ
    print("odds_list columns:", odds_list.columns.tolist())
    # # 'data' にオッズ情報をマージ
    # data = pd.merge(data, odds_list[['レースID', '艇番', '単勝オッズ']], on=['レースID', '艇番'], how='left')

    # 'data' と 'odds_list' を 'レースID' でマージ
    print(odds_list)

    data = pd.merge(data, odds_list, on='レースID', how='left')
    print(data)

    # # 'data' における各艇の単勝オッズを取得
    # def get_odds(row):
    #     # 'odds_list' の '単勝オッズ' 列には各組み合わせのオッズが含まれていると仮定
    #     # ここでは各艇番のオッズを適切に取得するために、以下のような処理を行います
    #     boat_number = int(row['艇番'])
    #     # '単勝オッズ1' ～ '単勝オッズ6' の列から対応する艇番のオッズを取得
    #     odds_column = f'単勝オッズ{boat_number}'
    #     if odds_column in data.columns:
    #         return row[odds_column]
    #     else:
    #         return np.nan

    # # 'data' に各艇の単勝オッズを取得
    # data['単勝オッズ'] = data.apply(get_odds, axis=1)
    # print(data['単勝オッズ'])
    # # オッズが欠損している行を削除
    data = data.dropna(subset=['win_odds'])

    # オッズをデシマルオッズに変換
    data['decimal_odds'] = data['win_odds'] 
    print(data['decimal_odds'])
    print(data['prob_0'])

    # ケリー基準に基づいて掛け金割合を計算
    data['bet_fraction'] = data.apply(lambda row: calculate_kelly_criterion(row['prob_0'], row['decimal_odds']), axis=1)

    # 掛け金を計算（総資金を100,000円と仮定
    total_capital = 1000  # 総資金（円）
    data['bet_amount'] = data['bet_fraction'] * total_capital
    # 【追加部分】bet_amount を100円単位に調整
    # bet_amount を100円の倍数に丸め、bet_fractionが0より大きい場合は最低100円に設定
    data['bet_amount'] = data['bet_amount'].apply(round_bet_amount)

    # 掛け金がNaNの場合は0に置換
    data['bet_amount'] = data['bet_amount'].fillna(0)

    # 各艇の勝敗を判定（1着の場合勝利）
    data['win'] = (data['着'] == 1).astype(int)

    # 回収金額を計算
    data['payout'] = data['bet_amount'] * data['decimal_odds'] * data['win']

    pd.set_option('display.max_rows', 500)
    print(data[['レースID','艇番','prob_0','win_odds','bet_amount', 'payout']].head(500))
    # 総投資額と総回収額を計算
    total_investment = data['bet_amount'].sum()
    total_return = data['payout'].sum()

    # 回収率を計算
    return_rate = (total_return / total_investment) * 100 if total_investment > 0 else 0

    print(f"投資したレース数: {len(data)/6}")
    print(f"\n総投資額: {total_investment:.2f}円")
    print(f"総回収額: {total_return:.2f}円")
    print(f"回収率: {return_rate:.2f}%")

    # 掛け金の統計情報
    print("\n掛け金の統計情報:")
    print(f"最大値: {data['bet_amount'].max():.2f}円")
    print(f"最小値: {data['bet_amount'].min():.2f}円")
    print(f"平均: {data['bet_amount'].mean():.2f}円")
    print(f"標準偏差: {data['bet_amount'].std():.2f}円")

    # 回収金額の統計情報
    print("\n回収金額の統計情報:")
    print(f"最大値: {data['payout'].max():.2f}円")
    print(f"最小値: {data['payout'].min():.2f}円")
    print(f"平均: {data['payout'].mean():.2f}円")
    print(f"標準偏差: {data['payout'].std():.2f}円")

    # 単勝の掛け金と回収金額の累積を計算
    data_sorted = data.sort_values(by=['レースID', '艇番'])  # 必要に応じてソート
    data_sorted = data_sorted.reset_index(drop=True)
    data_sorted['cumulative_bet'] = data_sorted['bet_amount'].cumsum()
    data_sorted['cumulative_payout'] = data_sorted['payout'].cumsum()

    # 単勝の累積回収をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_payout'], label='累積回収金額', color='green')
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_bet'], label='累積投資金額', color='red')
    plt.title("単勝の累積回収金額 vs 累積投資金額")
    plt.xlabel("ベット回数")
    plt.ylabel("金額 (円)")
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_return_single_win.png')
    plt.show()
    print("プロットが 'cumulative_return_single_win.png' として保存されました。")

    # 掛け金と回収金額のヒストグラムをプロット
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(data['bet_amount'], bins=50, color='green', edgecolor='black', alpha=0.7)
    plt.title("掛け金の分布")
    plt.xlabel("掛け金額 (円)")
    plt.ylabel("度数")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(data['payout'], bins=50, color='orange', edgecolor='black', alpha=0.7)
    plt.title("回収金額の分布")
    plt.xlabel("回収金額 (円)")
    plt.ylabel("度数")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('bet_payout_distribution.png')
    plt.show()
    print("プロットが 'bet_payout_distribution.png' として保存されました。")

    # 【追加部分】日付ごとのリターンの計算とプロット
    # 日付を datetime 型に変換
    # 'レースID' から 'date' カラムを生成
    if 'レースID' in data.columns:
        data['レースID'] = data['レースID'].astype(str)
        data['date'] = pd.to_datetime(data['レースID'].str[:8], format='%Y%m%d')
    else:
        print("Error: 'レースID' カラムがデータに存在しません。")
        return
    # 日付ごとの総投資額と総回収額を計算
    daily_summary = data.groupby('date').agg(
        daily_investment=pd.NamedAgg(column='bet_amount', aggfunc='sum'),
        daily_payout=pd.NamedAgg(column='payout', aggfunc='sum')
    ).reset_index()

    # 日付ごとの純利益を計算
    daily_summary['daily_return'] = daily_summary['daily_payout'] - daily_summary['daily_investment']

    # 累積リターンを計算
    daily_summary = daily_summary.sort_values('date')
    daily_summary['cumulative_return'] = daily_summary['daily_return'].cumsum()

    # 日付を文字列形式に変換（プロットの見やすさのため）
    daily_summary['date_str'] = daily_summary['date'].dt.strftime('%Y-%m-%d')

    # 日付ごとの累積リターンをプロット
    plt.figure(figsize=(12, 6))
    plt.plot(daily_summary['date'], daily_summary['cumulative_return'], marker='o', linestyle='-', color='purple')
    plt.title("日付ごとの累積リターンの推移")
    plt.xlabel("日付")
    plt.ylabel("累積リターン (円)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cumulative_return_by_date.png')
    plt.show()
    print("プロットが 'cumulative_return_by_date.png' として保存されました。")

    # 【オプション】日付ごとの純利益のヒストグラムをプロット
    plt.figure(figsize=(8, 6))
    sns.barplot(x='date_str', y='daily_return', data=daily_summary, palette='viridis')
    plt.title("日付ごとの純利益")
    plt.xlabel("日付")
    plt.ylabel("純利益 (円)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('daily_return_barplot.png')
    plt.show()
    print("プロットが 'daily_return_barplot.png' として保存されました。")

    # 総リターンの表示
    total_daily_return = daily_summary['daily_return'].sum()
    print(f"\n総日付ごとの純利益: {total_daily_return:.2f}円")

    #     # 'data' における各艇の複勝オッズを取得
    # def get_fukusho_odds(row):
    #     boat_number = int(row['艇番'])
    #     odds_column = f'複勝オッズ{boat_number}'
    #     if odds_column in data.columns:
    #         return row[odds_column]
    #     else:
    #         return np.nan

    # data['複勝オッズ'] = data.apply(get_fukusho_odds, axis=1)

    # # オッズが欠損している行を削除
    # data = data.dropna(subset=['複勝オッズ'])

    # # オッズをデシマルオッズに変換
    # data['fukusho_decimal_odds'] = data['複勝オッズ'] / 100

    # # 複勝の勝率を推定（1着または2着になる確率）
    # data['prob_fukusho'] = data['prob_0'] + data['prob_1']

    # # ケリー基準に基づいて複勝の掛け金割合を計算
    # data['bet_fraction_fukusho'] = data.apply(
    #     lambda row: calculate_kelly_criterion(row['prob_fukusho'], row['fukusho_decimal_odds']),
    #     axis=1
    # )

    # # 掛け金を計算（複勝）
    # data['bet_amount_fukusho'] = data['bet_fraction_fukusho'] * total_capital

    # # 掛け金がNaNの場合は0に置換
    # data['bet_amount_fukusho'] = data['bet_amount_fukusho'].fillna(0)

    # # 各艇の勝敗を判定（複勝：1着または2着の場合勝利）
    # data['win_fukusho'] = data['着'].isin([1, 2]).astype(int)

    # # 回収金額を計算（複勝）
    # data['payout_fukusho'] = data['bet_amount_fukusho'] * data['fukusho_decimal_odds'] * data['win_fukusho']

    # # 複勝の総投資額と総回収額を計算
    # total_investment_fukusho = data['bet_amount_fukusho'].sum()
    # total_return_fukusho = data['payout_fukusho'].sum()

    # # 回収率を計算（複勝）
    # return_rate_fukusho = (total_return_fukusho / total_investment_fukusho) * 100 if total_investment_fukusho > 0 else 0

    # print(f"\n複勝の総投資額: {total_investment_fukusho:.2f}円")
    # print(f"複勝の総回収額: {total_return_fukusho:.2f}円")
    # print(f"複勝の回収率: {return_rate_fukusho:.2f}%")
def adjust_odds(row, alpha=0.1):
    """
    自分のベットによるオッズの変化を計算します。
    row: データ行
    alpha: オッズ変化の度合いを調整するパラメータ
    """
    # 各レースごとに総ベット額を計算
    race_id = row['レースID']
    race_bets = row['bet_amount']

    # 総ベット額（仮定：各レースの総ベット額が20,000円）
    total_bet_per_race = 20000

    # 自分のベット額の割合
    bet_fraction = row['bet_amount'] / total_bet_per_race if total_bet_per_race > 0 else 0

    # オッズの調整
    adjusted_odds = row['decimal_odds'] / (1 + alpha * bet_fraction)

    return adjusted_odds

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
