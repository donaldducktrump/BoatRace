# model_1min.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random
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
import joblib

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
    # モデルの読み込み
    try:
        model = joblib.load('multioutput.pkl')
        print("モデルを読み込みました。")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # OneHotEncoderの読み込み
    ohe = joblib.load('onehot_encoder.pkl')
    print("OneHotEncoderを読み込みました。")

    # 特徴量名の読み込み
    feature_names_multi = pd.read_csv('feature_names.csv', header=None).squeeze().tolist()
    print("特徴量名を読み込みました。")

    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_first.txt')
        gbm_odds = lgb.Booster(model_file='boatrace_odds_model.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_first.csv').squeeze().tolist()
    feature_names_odds = pd.read_csv('feature_names_odds.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2410']
    date_files = {
        '2410': [f'2410{day:02d}' for day in range(15,26)],
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
    # 'win_odds_mean' を計算
    data_odds['win_odds_mean'] = data_odds.groupby(['選手登番', '艇番'])['win_odds'].transform('mean')
    data_odds = data_odds[['win_odds_mean', '選手登番', '艇番', 'win_odds']]

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
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                '前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather', 'wind_d','ESC','艇番','性別','支部','級別']
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # 数値データの選択
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]

    # 特徴量の結合
    X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

    for col in feature_names:
        if col not in X_processed.columns:
            X_processed[col] = 0
    X_processed = X_processed[feature_names]  # 学習時の特徴量と順序を合わせる

    # 予測
    y_pred = gbm.predict(X_processed)

    data['prob_0'] = y_pred[:, 0]  # 1着の確率
    data['prob_1'] = y_pred[:, 1]

    # 'prob_0' の分布をプロット
    plt.figure(figsize=(8, 6))
    plt.hist(data['prob_0'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title("予測確率 'prob_0' の分布")
    plt.xlabel("予測確率 (prob_0)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('prob_0_distribution_1min.png')
    plt.show()
    print("プロットが 'prob_0_distribution.png' として保存されました。")

    # 確定後オッズの予測
    # 特徴量の指定
    boats = [1, 2, 3, 4, 5, 6]
    data_list_new = []
    for boat in boats:
        boat_data = data.copy()
        boat_data = boat_data[boat_data['艇番'] == boat].copy()
        boat_data['final_odds'] = boat_data['win_odds']  # 確定後のオッズをターゲット
        boat_data['before1min_odds'] = boat_data['win_odds1min']  # 1分前のオッズを特徴量
        boat_data['boat_number'] = boat  # 舟番号を特徴量として追加
        data_list_new.append(boat_data)

    # 各艇のデータを結合
    data = pd.concat(data_list_new, ignore_index=True)

    # '選手登番' と '艇番' ごとに 'win_odds_mean' を集約
    data_odds_grouped = data_odds.groupby(['選手登番', '艇番'], as_index=False)['win_odds_mean'].mean()
    data['選手登番'] = data['選手登番'].astype(int)
    data_odds_grouped['選手登番'] = data_odds_grouped['選手登番'].astype(int)
    data['艇番'] = data['艇番'].astype(int)
    data_odds_grouped['艇番'] = data_odds_grouped['艇番'].astype(int)
    data = data.merge(data_odds_grouped, on=['選手登番', '艇番'], how='left')

    # データの再構成
    data = data.dropna()
    data = data.sort_values(['レースID', '艇番'])
    grouped = data.groupby('レースID')

    X_list = []
    race_ids_list = []

    features_multi = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                      '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                      'ET', 'tilt', 'EST', 'ESC',
                      'weather', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                      'win_odds1min', 'prob_0', 'place_odds1min', 'win_odds_mean',
                      '性別', '勝率', '複勝率', '優勝回数', '優出回数',
                      '前期能力指数', '今期能力指数', '平均スタートタイミング']

    for race_id, group in grouped:
        if group.shape[0] != 6:
            continue

        group = group.sort_values('艇番')
        # オッズ関連の特徴量を逆数に変換
        for col in ['win_odds1min', 'place_odds1min', 'win_odds_mean']:
            group[col] = group[col].replace(0, np.nan)  # 0をNaNに置換して除算エラーを防ぐ
            group[col] = 0.725 / group[col]
            group[col] = group[col].fillna(group[col].mean())  # NaNを平均値で補完
        

        X_race = group[features_multi].values.flatten()
        X_list.append(X_race)
        
        race_ids_list.append(race_id)

    # 特徴量名の作成
    feature_names_multi_expanded = []
    for i in range(1, 7):
        for feat in features_multi:
            feature_names_multi_expanded.append(f'{feat}_{i}')

    # 特徴量のデータフレームを作成
    X_df = pd.DataFrame(X_list, columns=feature_names_multi_expanded)

    # エンコーダを読み込む
    ohe = joblib.load('onehot_encoder.pkl')
    print("OneHotEncoder を読み込みました。")

    # カテゴリカル変数のリストは同じ
    categorical_features = []
    for i in range(1, 7):
        for cat_feat in ['会場', 'weather', 'wind_d', 'ESC', '艇番', '性別', '支部', '級別']:
            categorical_features.append(f'{cat_feat}_{i}')

    # カテゴリカル変数を文字列型に変換
    X_df[categorical_features] = X_df[categorical_features].astype(str)

    # 数値データの選択
    numeric_features = [col for col in X_df.columns if col not in categorical_features]
    X_numeric = X_df[numeric_features].astype(float).reset_index(drop=True)

    # カテゴリカル変数のエンコーディング
    X_categorical = ohe.transform(X_df[categorical_features])

    # エンコード後のカテゴリカル特徴量名を取得
    feature_names_categorical = ohe.get_feature_names_out(categorical_features)

    # カテゴリカルデータを DataFrame に変換し、特徴量名を設定
    X_categorical_df = pd.DataFrame(X_categorical, columns=feature_names_categorical).reset_index(drop=True)

    # 特徴量の結合
    X_processed_multi = pd.concat([X_numeric, X_categorical_df], axis=1)

    # 訓練時に存在しなかった特徴量カラムを追加
    for col in feature_names_multi:
        if col not in X_processed_multi.columns:
            X_processed_multi[col] = 0

    # 余分なカラムを削除
    X_processed_multi = X_processed_multi[feature_names_multi]

    # 予測
    y_pred_inv = model.predict(X_processed_multi)


    # 予測値を正の値にクリッピング
    y_pred_inv = np.maximum(y_pred_inv, 0)

    # 予測値を元のオッズに変換
    y_pred = 0.725 / y_pred_inv

    # 無限大になる値を適切な最大値に置換
    max_odds = 1000
    y_pred = np.minimum(y_pred, max_odds)

    # 予測結果の整形
    y_pred_flat = y_pred.reshape(-1)
    boat_numbers = np.tile(np.arange(1, 7), len(y_pred))
    race_ids_flat = np.repeat(race_ids_list, 6)
 
    pred_df = pd.DataFrame({
        'レースID': race_ids_flat,
        '艇番': boat_numbers,
        'predicted_odds': y_pred_flat,
    })

    # 実際のデータとのマージ
    data = data.sort_values(['レースID', '艇番'])
    data.reset_index(drop=True, inplace=True)
    pred_df = pred_df.sort_values(['レースID', '艇番'])
    pred_df.reset_index(drop=True, inplace=True)

    # 予測結果と実際のデータを結合
    result_df = pd.concat([data, pred_df['predicted_odds']], axis=1)
    
    # 回収率の計算
    # ケリー基準に基づいて掛け金割合を計算
    result_df['bet_fraction'] = result_df.apply(lambda row: calculate_kelly_criterion(row['prob_0'], row['predicted_odds']), axis=1)

    # 総資金を設定（例：100,000円）
    total_capital = 1000
    result_df['bet_amount'] = result_df['bet_fraction'] * total_capital

    # 掛け金を100円単位に調整
    result_df['bet_amount'] = result_df['bet_amount'].apply(round_bet_amount)
    result_df['bet_amount'] = np.where(result_df['prob_0'] < 0.2, 0, result_df['bet_amount'])  # 確率が低い場合は掛け金を0にする
    result_df['bet_amount'] = np.where(result_df['bet_amount']<=100, 0, result_df['bet_amount'])  # 掛け金が100円未満の場合は0にする

    # 掛け金がNaNの場合は0に置換
    result_df['bet_amount'] = result_df['bet_amount'].fillna(0)

    # 各艇の勝敗を判定（1着の場合勝利）
    result_df['win'] = (result_df['着'] == 1).astype(int)

    # 回収金額を計算
    result_df['payout'] = result_df['bet_amount'] * result_df['win_odds'] * result_df['win']

    # 総投資額と総回収額を計算
    total_investment = result_df['bet_amount'].sum()
    total_return = result_df['payout'].sum()

    # 回収率を計算
    return_rate = (total_return / total_investment) * 100 if total_investment > 0 else 0

    print(f"総投資額: {total_investment:.2f}円")
    print(f"総回収額: {total_return:.2f}円")
    print(f"回収率: {return_rate:.2f}%")
    pd.set_option('display.max_rows', 50)
    # pd.set_option('display.max_columns', 50)
    filtered_data = result_df[result_df['艇番'] == 1]
    print(filtered_data[['レースID','艇番','prob_0','win_odds_mean','win_odds1min','win_odds','predicted_odds','bet_amount','payout']].head(50))
    
    # 2. 艇番ごとの実際のオッズと予測オッズの散布図
    for boat in range(1, 7):
        plt.figure(figsize=(8, 8))
        boat_data = result_df[result_df['艇番'] == boat]
        plt.scatter(boat_data['win_odds'], boat_data['predicted_odds'], alpha=0.5)
        plt.plot([boat_data['win_odds'].min(), boat_data['win_odds'].max()],
                 [boat_data['win_odds'].min(), boat_data['win_odds'].max()], 'r--')
        plt.xlabel('実際のオッズ')
        plt.ylabel('予測オッズ')
        plt.title(f'実際のオッズと予測オッズの比較（艇番 {boat}）')
        plt.savefig(f'predicted_vs_true_boat_{boat}.png')
        plt.show()

    # 必要に応じて可視化や追加の分析を行う
    plt.figure(figsize=(10, 6))
    plt.scatter(result_df['win_odds'], result_df['predicted_odds'], alpha=0.5)
    plt.plot([result_df['win_odds'].min(), result_df['win_odds'].max()],
            [result_df['win_odds'].min(), result_df['win_odds'].max()], 'r--')
    plt.xlabel('実際のオッズ (win_odds)')
    plt.ylabel('予測オッズ (predicted_odds)')
    plt.title('予測オッズと実際のオッズの比較')
    plt.savefig('predicted_odds_vs_actual_odds_1min_inv.png')
    plt.show()

    # 単勝の掛け金と回収金額の累積を計算
    data_sorted = result_df.sort_values(by=['レースID', '艇番'])  # 必要に応じてソート
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
    plt.savefig('cumulative_return_single_win_1min_inv.png')
    plt.show()
    print("プロットが 'cumulative_return_single_win.png' として保存されました。")

# ケリー基準の計算関数
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
    inspect_model()
