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

import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
import seaborn as sns
from sklearn.isotonic import IsotonicRegression

from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, preprocess_data1min, preprocess_data_old, load_before_data, load_before1min_data, merge_before_data, load_dataframe
from get_data import load_fan_data

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

def load_fan_data_from_csv():
    file_path = 'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/fan_dataframe/fan_data.csv'
    if os.path.exists(file_path):
        df_fan = pd.read_csv(file_path)
        return df_fan
    else:
        print(f"ファイルが見つかりません: {file_path}")
        return pd.DataFrame()

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

def calculate_kelly_criterion(prob, odds):
    """
    ケリー基準に基づく最適な掛け金割合を計算します。
    prob: 予測確率
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

    # DNNモデルの定義
    class MultiOutputRegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(MultiOutputRegressionModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(128, 6)  # 6艇分の出力

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = torch.clamp(x, min=1e-6)  # 予測値を正の値に制限
            return x

    # モデルと前処理器のロード
    try:
        # エンコーダーとスケーラーのロード
        encoder = joblib.load('encoder.joblib')
        scaler = joblib.load('scaler.joblib')

        # モデルのインスタンス化
        # input_dim はエンコーディング後の特徴量数
        # 一時的に0を設定し、後で上書き
        model = MultiOutputRegressionModel(input_dim=0)
        model.load_state_dict(torch.load('trained_dnn_model.pth'))
        model.eval()
    except Exception as e:
        print(f"Failed to load DNN model or preprocessing tools: {e}")
        return

    # 評価期間のデータ日付リスト作成
    month_folders = ['2410']
    date_files = {
        '2410': [f'2410{day:02d}' for day in range(15, 26)],   # 241015 - 241025
    }

    # データを結合
    data_list = []
    odds_list1 = []

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
    # 全データを結合
    data = pd.concat(data_list, ignore_index=True)
    odds_list = pd.concat(odds_list1, ignore_index=True)

    # ファンデータの読み込みとマージ
    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    # 必要な列を指定してマージ
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')

    # データの前処理
    data = preprocess_data1min(data)

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

    # 予測
    y_pred = gbm.predict(X_processed)

    # data['prob_0'] = y_pred[:, 0]  # 1,2着の確率
    # data['prob_1'] = y_pred[:, 1]
    # data['prob_2'] = y_pred[:, 2]

    data['prob_0'] = y_pred[:, 0]  # 1,着の確率
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


    #　確定後オッズの予測
    # 特徴量の指定
    boats = [1, 2, 3, 4, 5, 6]
    data_list = []
    for boat in boats:
        boat_data = data.copy()
        boat_data = boat_data[boat_data['艇番'] == boat].copy()
        boat_data['final_odds'] = boat_data['win_odds']  # 確定後のオッズをターゲット
        boat_data['before1min_odds'] = boat_data['win_odds1min']  # 1分前のオッズを特徴量
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

    # 特徴量とターゲットの設定
    features_multi = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                      '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                      'ET', 'tilt', 'EST', 'ESC',
                      'weather', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                      'win_odds1min', 'place_odds1min', 'win_odds_mean',
                      '性別', '勝率', '複勝率', '優勝回数', '優出回数',
                      '前期能力指数', '今期能力指数', '平均スタートタイミング']

    # オッズ関連の特徴量を逆数に変換
    for col in ['win_odds1min', 'place_odds1min', 'win_odds_mean']:
        data[col] = data[col].replace(0, np.nan)
        data[col] = 1 / data[col]
        data[col] = data[col].fillna(data[col].mean())

    # データの前処理
    data = data.dropna()
    data = data.sort_values(['レースID', '艇番']).reset_index(drop=True)

    # レースごとにデータをまとめる
    grouped = data.groupby('レースID')

    X_list = []
    race_ids_list = []
    boat_numbers_list = []
    y_true_list = []

    for race_id, group in grouped:
        if len(group) != 6:
            continue  # 6艇揃っていないレースは除外
        group = group.sort_values('艇番')
        X_race = group[features_multi].values.flatten()
        X_list.append(X_race)
        race_ids_list.append(race_id)
        boat_numbers_list.append(group['艇番'].values)
        y_true_list.append(group['win_odds'].values)  # 実際のオッズ

    # X を DataFrame に変換
    X = pd.DataFrame(X_list)
    race_ids = pd.Series(race_ids_list)
    boat_numbers = pd.DataFrame(boat_numbers_list)
    y_true = pd.DataFrame(y_true_list)

    # 特徴量名の作成
    feature_names_multi = []
    for i in range(1, 7):
        for feat in features_multi:
            feature_names_multi.append(f'{feat}_{i}')
    X.columns = feature_names_multi

    # カテゴリカル変数のエンコーディング
    categorical_features = []
    for i in range(1, 7):
        for cat_feat in ['会場', 'weather', 'wind_d', 'ESC', '艇番', '性別', '支部', '級別']:
            categorical_features.append(f'{cat_feat}_{i}')

    X_categorical = X[categorical_features]
    X_numeric = X.drop(columns=categorical_features)

    # One-Hotエンコーディング
    X_categorical_encoded = encoder.transform(X_categorical)

    # 数値データのスケーリング
    X_numeric_scaled = scaler.transform(X_numeric)

    # 特徴量の結合
    X_processed = np.hstack([X_numeric_scaled, X_categorical_encoded])

    # input_dim の設定
    input_dim = X_processed.shape[1]

    # モデルのインスタンス化（input_dim を設定）
    model = MultiOutputRegressionModel(input_dim)
    model.load_state_dict(torch.load('trained_dnn_model.pth'))
    model.eval()

    # モデルの予測
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()

    # 予測値を元のオッズに変換
    y_pred_odds = 1 / y_pred
    y_pred_odds = np.clip(y_pred_odds, 1, 200)  # 最大値を200に制限

    # 結果の整形
    num_samples = y_pred_odds.shape[0]
    boat_numbers = boat_numbers.values
    race_ids_repeated = race_ids.values
    y_true_odds = y_true.values

    # 予測結果を整形
    pred_list = []
    for i in range(num_samples):
        for j in range(6):
            pred_list.append({
                'レースID': race_ids_repeated[i],
                '艇番': boat_numbers[i][j],
                'predicted_odds': y_pred_odds[i][j],
                'win_odds': y_true_odds[i][j]
            })

    pred_df = pd.DataFrame(pred_list)

    # 元のデータとマージ
    data = pd.merge(data, pred_df, on=['レースID', '艇番'], how='inner')

    # 予測オッズから確率を計算
    data['prob_predicted'] = 1 / data['predicted_odds']

    # 各レースごとに確率を正規化して合計が1になるように調整
    data['prob_predicted'] = data.groupby('レースID')['prob_predicted'].apply(lambda x: x / x.sum())

    # ケリー基準に基づいて掛け金割合を計算
    data['bet_fraction'] = data.apply(lambda row: calculate_kelly_criterion(row['prob_predicted'], row['win_odds']), axis=1)

    # 掛け金を計算（総資金を100,000円と仮定）
    total_capital = 100000  # 総資金（円）
    data['bet_amount'] = data['bet_fraction'] * total_capital

    # bet_amount を100円単位に調整
    data['bet_amount'] = data['bet_amount'].apply(round_bet_amount)
    data['bet_amount'] = np.where(data['prob_predicted'] < 0.2, 0, data['bet_amount'])  # 確率が低い場合は掛け金を0にする
    data['bet_amount'] = np.where(data['bet_amount'] < 100, 0, data['bet_amount'])  # 掛け金が100円未満の場合は0にする

    # 掛け金がNaNの場合は0に置換
    data['bet_amount'] = data['bet_amount'].fillna(0)

    # 各艇の勝敗を判定（1着の場合勝利）
    data['win'] = (data['着'] == 1).astype(int)

    # 回収金額を計算
    data['payout'] = data['bet_amount'] * data['win_odds'] * data['win']

    # 総投資額と総回収額を計算
    total_investment = data['bet_amount'].sum()
    total_return = data['payout'].sum()

    # 回収率を計算
    return_rate = (total_return / total_investment) * 100 if total_investment > 0 else 0

    print(f"\n総投資額: {total_investment:.2f}円")
    print(f"総回収額: {total_return:.2f}円")
    print(f"回収率: {return_rate:.2f}%")

    # 結果の保存
    data.to_csv('result_odds_pred.csv', index=False)

    # 可視化

    # 予測オッズと実際のオッズの散布図
    plt.figure(figsize=(8, 8))
    plt.scatter(data['win_odds'], data['predicted_odds'], alpha=0.5)
    plt.plot([data['win_odds'].min(), data['win_odds'].max()],
             [data['win_odds'].min(), data['win_odds'].max()], 'r--')
    plt.xlabel('実際のオッズ')
    plt.ylabel('予測オッズ')
    plt.title('実際のオッズと予測オッズの比較')
    plt.savefig('predicted_vs_true_odds.png')
    plt.show()

    # 残差プロット
    data['residuals'] = data['win_odds'] - data['predicted_odds']

    plt.figure(figsize=(10, 6))
    plt.scatter(data['predicted_odds'], data['residuals'], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('予測オッズ')
    plt.ylabel('残差 (実際のオッズ - 予測オッズ)')
    plt.title('残差プロット')
    plt.savefig('residuals_plot.png')
    plt.show()

    # 累積リターンのプロット
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
    plt.savefig('cumulative_return.png')
    plt.show()

    # その他の分析やプロットも必要に応じて追加できます

if __name__ == '__main__':
    inspect_model()
