# model_1min.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, preprocess_data1min, preprocess_data_old, load_before_data, load_before1min_data, merge_before_data, load_dataframe
from get_data import load_fan_data
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
import joblib
from datetime import datetime

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

def load_fan_data_from_csv():
    file_path = 'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/fan_dataframe/fan_data.csv'
    if os.path.exists(file_path):
        df_fan = pd.read_csv(file_path)
        return df_fan
    else:
        print(f"ファイルが見つかりません: {file_path}")
        return pd.DataFrame()

def prepare_multi_output_data(data):
    """
    レースごとに6艇分のデータを1サンプルにまとめ、マルチアウトプット回帰用のデータを作成します。
    """
    # レースIDごとにグループ化
    grouped = data.groupby('レースID')
    
    # 必要な特徴量
    features = ['会場', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
               '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
               'ET', 'tilt', 'EST','ESC',
               'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
               'win_odds1min', 'prob_0','place_odds1min','win_odds_mean', 
               '性別', '勝率', '複勝率', '優勝回数', '優出回数','前期能力指数', '今期能力指数','平均スタートタイミング']
    
    # カテゴリカル変数の指定
    categorical_features = ['会場', 'weather','wind_d', 'ESC', '性別', '支部', '級別']
    
    # 特徴量とターゲットを格納するリスト
    X_list = []
    y_list = []
    
    for race_id, group in grouped:
        if len(group) != 6:
            # 6艇すべてのデータが揃っていないレースはスキップ
            continue
        
        # 各艇の特徴量を連結
        X_race = []
        for boat in sorted(group['艇番'].unique()):
            boat_data = group[group['艇番'] == boat].iloc[0]
            # 特徴量の取得
            boat_features = boat_data[features].values.tolist()
            X_race.extend(boat_features)
        
        X_list.append(X_race)
        
        # ターゲット（6艇分のオッズ）
        y_race = group.sort_values('艇番')['win_odds'].values
        y_list.append(y_race)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y

def adjust_odds(predicted_odds, target_sum_inverse=1.37931):
    """
    予測されたオッズを調整して、逆数の和がtarget_sum_inverseになるようにする。
    """
    inverses = 1 / predicted_odds
    current_sum_inverse = inverses.sum()
    scaling_factor = target_sum_inverse / current_sum_inverse
    adjusted_odds = predicted_odds / scaling_factor
    return adjusted_odds

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

def inspect_model():
    # マルチアウトプットモデルの読み込み
    try:
        multi_output_model = joblib.load('boatrace_odds_multi_output_model.joblib')
    except Exception as e:
        print(f"Failed to load multi-output model: {e}")
        return

    # 特徴量名の読み込み
    feature_names_odds = pd.read_csv('feature_names_odds_multi_output.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2410']
    date_files = {
        '2410': [f'2410{day:02d}' for day in range(15,25)],   # 241015 - 241024
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
                before1min_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data_1min/{month}/beforeinfo1min_{date}.txt'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'

                b_data = load_b_file(b_file)
                k_data, odds_list_part = load_k_file(k_file)
                before_data = load_before_data(before_file)
                before1min_data = load_before1min_data(before1min_file)

                if not b_data.empty and not k_data.empty and not before_data.empty:
                    # データのマージ
                    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
                    before_data = remove_common_columns(data, before_data, on_columns=['選手登番', 'レースID', '艇番'])
                    data = merge_before_data(data, before_data)
                    before1min_data = remove_common_columns(data, before1min_data, on_columns=['選手登番', 'レースID', '艇番'])
                    data = merge_before_data(data, before1min_data)
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

    # 全データを結合
    data = pd.concat(data_list, ignore_index=True)
    odds_list = pd.concat(odds_list1, ignore_index=True)

    # 前処理
    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    # 必要な列を指定してマージ
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')

    data = preprocess_data1min(data)

    # 特徴量の指定
    features = ['会場', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                'win_odds1min', 'prob_0','place_odds1min','win_odds_mean', 
                '性別', '勝率', '複勝率', '優勝回数', '優出回数','前期能力指数', '今期能力指数','平均スタートタイミング']

    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather', 'wind_d','ESC','性別','支部','級別']
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # 数値データの選択
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]

    # 特徴量の結合
    X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

    # 特徴量名の読み込み
    feature_names = feature_names_odds

    # 天候_雪 カラムが存在しない場合に追加する
    for col in feature_names:
        if col not in X_processed.columns:
            X_processed[col] = 0

    # 学習時の特徴量と順序を合わせる
    X_processed = X_processed[feature_names]

    # 特徴量をマルチアウトプットモデルに適合する形式に変換
    # 必要に応じてスケーリングなどの前処理を追加
    # 例: StandardScaler の適用
    # scaler = StandardScaler()
    # X_processed = scaler.fit_transform(X_processed)

    # レースごとに6艇分のデータを1サンプルにまとめる
    X_multi, y_multi = prepare_multi_output_data(data)

    # 特徴量の準備
    # X_processed は個々の艇の特徴量を含むため、再構築が必要
    # ここでは、元のコードのデータ結合部分を活用

    # 特徴量とターゲットの準備
    # X_multi, y_multi はマルチアウトプット用に準備済み

    # 予測
    y_pred = multi_output_model.predict(X_processed)

    # オッズの調整
    y_pred_adjusted = np.array([adjust_odds(odds) for odds in y_pred])

    # 各レースごとに予測されたオッズをデータに割り当てる
    # データが1レースあたり6艇分のデータであることを確認
    data['predicted_odds'] = np.repeat(y_pred_adjusted, 6)

    # マージ用にレースIDを取得
    data['レースID'] = data['レースID'].astype(str)
    
    # ベット戦略の適用
    data['bet_fraction'] = data.apply(lambda row: calculate_kelly_criterion(row['prob_0'], row['predicted_odds']), axis=1)

    # 掛け金を計算（総資金を100,000円と仮定）
    total_capital = 100000  # 総資金（円）
    data['bet_amount'] = data['bet_fraction'] * total_capital

    # bet_amount を100円の倍数に丸め、bet_fractionが0より大きい場合は最低100円に設定
    data['bet_amount'] = data['bet_amount'].apply(round_bet_amount)
    data['bet_amount'] = np.where(data['prob_0'] < 0.2, 0, data['bet_amount'])  # 確率が低い場合は掛け金を0にする
    data['bet_amount'] = np.where(data['bet_amount'] <= 100, 0, data['bet_amount'])  # 掛け金が100円未満の場合は0にする

    # 掛け金がNaNの場合は0に置換
    data['bet_amount'] = data['bet_amount'].fillna(0)

    # 各艇の勝敗を判定（1着の場合勝利）
    data['win'] = (data['着'] == 1).astype(int)

    # 回収金額を計算
    data['payout'] = data['bet_amount'] * data['predicted_odds'] * data['win']

    # 結果の保存
    data.to_csv('result_odds_pred.csv', index=False)
    print("最終オッズの予測結果を保存しました。")

    # 総投資額と総回収額を計算
    total_investment = data['bet_amount'].sum()
    total_return = data['payout'].sum()

    # 回収率を計算
    return_rate = (total_return / total_investment) * 100 if total_investment > 0 else 0

    # 結果の表示
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

    # 日付を datetime 型に変換
    if 'レースID' in data_sorted.columns:
        data_sorted['レースID'] = data_sorted['レースID'].astype(str)
        # レースIDの先頭8文字を日付として扱う（フォーマットに応じて調整）
        data_sorted['date'] = pd.to_datetime(data_sorted['レースID'].str[:8], format='%Y%m%d', errors='coerce')
    else:
        print("Error: 'レースID' カラムがデータに存在しません。")
        return

    # 日付ごとの総投資額と総回収額を計算
    daily_summary = data_sorted.groupby('date').agg(
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

    # 単勝の累積回収をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_payout'], label='累積回収金額', color='green')
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_bet'], label='累積投資金額', color='red')
    plt.title("単勝の累積回収金額 vs 累積投資金額")
    plt.xlabel("ベット回数")
    plt.ylabel("金額 (円)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_return_single_win_1min.png')
    plt.show()
    print("プロットが 'cumulative_return_single_win_1min.png' として保存されました。")

    # 掛け金と回収金額のヒストグラムをプロット
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(data[data['艇番'] == 1]['bet_amount'], bins=50, color='green', edgecolor='black', alpha=0.7)
    plt.title("掛け金の分布")
    plt.xlabel("掛け金額 (円)")
    plt.ylabel("度数")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(data[data['艇番'] == 1]['payout'], bins=50, color='orange', edgecolor='black', alpha=0.7)
    plt.title("回収金額の分布")
    plt.xlabel("回収金額 (円)")
    plt.ylabel("度数")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('bet_payout_distribution_1min.png')
    plt.show()
    print("プロットが 'bet_payout_distribution_1min.png' として保存されました。")

    # 日付ごとの累積リターンをプロット
    plt.figure(figsize=(12, 6))
    plt.plot(daily_summary['date'], daily_summary['cumulative_return'], marker='o', linestyle='-', color='purple')
    plt.title("日付ごとの累積リターンの推移")
    plt.xlabel("日付")
    plt.ylabel("累積リターン (円)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cumulative_return_by_date_1min.png')
    plt.show()
    print("プロットが 'cumulative_return_by_date_1min.png' として保存されました。")

    # 日付ごとの純利益のヒストグラムをプロット
    plt.figure(figsize=(8, 6))
    sns.barplot(x='date_str', y='daily_return', data=daily_summary, palette='viridis')
    plt.title("日付ごとの純利益")
    plt.xlabel("日付")
    plt.ylabel("純利益 (円)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('daily_return_barplot_1min.png')
    plt.show()
    print("プロットが 'daily_return_barplot_1min.png' として保存されました。")

    # 総リターンの表示
    total_daily_return = daily_summary['daily_return'].sum()
    print(f"\n総日付ごとの純利益: {total_daily_return:.2f}円")

if __name__ == '__main__':
    inspect_model()
