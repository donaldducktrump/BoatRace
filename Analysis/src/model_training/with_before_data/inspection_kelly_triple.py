import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, load_before_data, merge_before_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

def inspect_triple_model():
    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2410']
    date_files = {
        '2410': [f'2410{day:02d}' for day in range(1,15)],  # 241001 - 241014
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

    # 特徴量の指定
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST', 'ESC',
                'weather', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h']
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather', 'wind_d', 'ESC']
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # 数値データの選択
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]

    # 特徴量の結合
    X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

    # 学習時の特徴量と順序を合わせる
    for col in feature_names:
        if col not in X_processed.columns:
            X_processed[col] = 0

    X_processed = X_processed[feature_names]

    # 予測
    y_pred = gbm.predict(X_processed)
    data['prob_0'] = y_pred[:, 0]  # 1着の確率
    data['prob_1'] = y_pred[:, 1]  # 2着の確率
    data['prob_2'] = y_pred[:, 2]  # 3着の確率

    # オッズ情報をデータにマージ
    data = pd.merge(data, odds_list[['レースID', '三連単結果', '三連単オッズ']], on='レースID', how='left')

    # 各レースごとに1着、2着、3着を予測
    def predict_triple(group):
        # 1着、2着、3着をそれぞれ確率が高い順に並べる
        first_place = group.loc[group['prob_0'].idxmax(), '艇番']
        second_place = group.loc[group['prob_1'].idxmax(), '艇番']
        third_place = group.loc[group['prob_2'].idxmax(), '艇番']
        group['pred_triple'] = f"{first_place}-{second_place}-{third_place}"
        return group

    data = data.groupby('レースID').apply(predict_triple)

    # 勝敗を判定
    data['win'] = (data['pred_triple'] == data['三連単結果']).astype(int)

    # 一定額を賭ける（例：100円）
    bet_amount = 100
    data['bet_amount'] = bet_amount

    # 回収金額を計算
    data['payout'] = data['bet_amount'] * data['三連単オッズ'] * data['win']

    # 投資結果を表示
    pd.set_option('display.max_rows', 500)
    print(data[['レースID', '艇番', 'prob_0', 'prob_1', 'prob_2', 'pred_triple', '三連単結果', '三連単オッズ', 'bet_amount', 'payout']].head(500))

    # 総投資額と総回収額
    total_investment = data['bet_amount'].sum()
    total_return = data['payout'].sum()

    # 回収率
    return_rate = (total_return / total_investment) * 100 if total_investment > 0 else 0

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

    # 累積回収金額と累積投資金額のプロット
    data_sorted = data.sort_values(by=['レースID', '艇番']).reset_index(drop=True)
    data_sorted['cumulative_bet'] = data_sorted['bet_amount'].cumsum()
    data_sorted['cumulative_payout'] = data_sorted['payout'].cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_payout'], label='累積回収金額', color='green')
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_bet'], label='累積投資金額', color='red')
    plt.title("累積回収金額 vs 累積投資金額")
    plt.xlabel("ベット回数")
    plt.ylabel("金額 (円)")
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_return_triple.png')
    plt.show()

    # 掛け金と回収金額のヒストグラム
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
    plt.savefig('bet_payout_distribution_triple.png')
    plt.show()

# ケリー基準の計算関数を定義
def calculate_kelly_criterion(prob, odds):
    if odds <= 1:
        return 0
    else:
        f = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
    return max(0, f)

def round_bet_amount(amount, unit=100):
    if amount <= 0:
        return 0
    else:
        return int(np.ceil(amount / unit)) * unit

if __name__ == '__main__':
    inspect_triple_model()
