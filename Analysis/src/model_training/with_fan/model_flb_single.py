# model_1min.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import pandas as pd
import optuna.integration.lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, preprocess_data1min,preprocess_data_old, load_before_data, load_before1min_data ,merge_before_data
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss

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

def create_model():
    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_first.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_first.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410']
    date_files = {
        '2311': [f'2311{day:02d}' for day in range(1, 31)],  # 231101 - 231130
        '2312': [f'2312{day:02d}' for day in range(1, 32)],  # 231201 - 231231
        '2401': [f'2401{day:02d}' for day in range(1, 32)],  # 240101 - 240131
        '2402': [f'2402{day:02d}' for day in range(1, 29)],  # 240201 - 240228
        '2403': [f'2403{day:02d}' for day in range(1, 32)],  # 240301 - 240331
        '2404': [f'2404{day:02d}' for day in range(1, 31)],  # 240401 - 240430
        '2405': [f'2405{day:02d}' for day in range(1, 32)],  # 240501 - 240531
        '2406': [f'2406{day:02d}' for day in range(1, 31)],  # 240601 - 240630
        '2407': [f'2407{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(1,31)],   # 241001 - 241008
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
    
    data_odds = load_processed_odds_data()

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


    # 特徴量の指定
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h', '前期能力指数', '今期能力指数','平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather', 'wind_d','ESC', '艇番', '性別', '支部', '級別']
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
    # 既存の data DataFrame が存在すると仮定

    # 1. データのフィルタリング: win_odds が1以上であり、NaNでないものを選択
    initial_count = len(data)
    data = data[(data['win_odds'] >= 1) & (~data['win_odds'].isna()) & (~data['着'].isna())]
    filtered_count = len(data)
    print(f"Filtered data: {filtered_count} records out of {initial_count}.")

    # 2. 予測確率の計算
    data['predicted_prob'] = 0.725 / data['win_odds']

    # 3. 予測確率が0から0.725の範囲内にあることを確認
    # 必要に応じてクリップ
    data['predicted_prob'] = data['predicted_prob'].clip(lower=0, upper=0.725)

    # 4. 実際の結果を0と1に変換
    data['actual'] = (data['着'] == 1).astype(int)

    # 5. NaN や無限大の確認
    nan_probs = data['predicted_prob'].isna().sum()
    inf_probs = np.isinf(data['predicted_prob']).sum()
    if nan_probs > 0 or inf_probs > 0:
        print(f"Warning: {nan_probs} NaN and {inf_probs} infinite predicted probabilities found. These will be excluded.")
        data = data[(~data['predicted_prob'].isna()) & (~np.isinf(data['predicted_prob']))]

    # 6. キャリブレーションプロット（Calibration Plot）
    n_bins = 80
    data['prob_bin'] = pd.cut(data['predicted_prob'], bins=n_bins, labels=False, include_lowest=True)

    calibration = data.groupby('prob_bin').agg(
        mean_pred_prob=('predicted_prob', 'mean'),
        actual_win_rate=('actual', 'mean')
    ).reset_index()

    plt.figure(figsize=(8, 6))
    plt.plot(calibration['mean_pred_prob'], calibration['actual_win_rate'], marker='o', linestyle='-', label='Actual Win Rate')
    plt.plot([0, 0.725], [0, 0.725], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Win Rate')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig('flb_calibration_single.png')
    plt.show()

    n_bins = 10
    data['prob_bin_0'] = pd.cut(data['prob_0'], bins=n_bins, labels=False, include_lowest=True)

    calibration = data.groupby('prob_bin_0').agg(
        mean_pred_prob=('prob_0', 'mean'),
        actual_win_rate=('actual', 'mean')
    ).reset_index()

    plt.figure(figsize=(8, 6))
    plt.plot(calibration['mean_pred_prob'], calibration['actual_win_rate'], marker='o', linestyle='-', label='Actual Win Rate')
    plt.plot([0, 0.725], [0, 0.725], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Win Rate')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig('flb_calibration_single_0.png')
    plt.show()


    # 7. ヒストグラムと実際の勝率の比較
    plt.figure(figsize=(10, 6))
    sns.histplot(data['predicted_prob'], bins=n_bins, kde=False, color='skyblue', label='Predicted Probability', alpha=0.6)

    # バケットごとの実際の勝率をプロット（右側のy軸を使用）
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(calibration['mean_pred_prob'], calibration['actual_win_rate'], color='red', marker='o', linestyle='-', label='Actual Win Rate')

    # ラベルとタイトルの設定
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Frequency')
    ax2.set_ylabel('Actual Win Rate')
    plt.title('Histogram of Predicted Probabilities and Actual Win Rates')

    # 凡例の追加
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    plt.savefig('flb_histogram_single.png')
    plt.show()

    # # 8. 散布図（Scatter Plot）
    # plt.figure(figsize=(8, 6))
    # sns.scatterplot(x='predicted_prob', y='actual', data=data, alpha=0.3)
    # plt.xlabel('Predicted Probability')
    # plt.ylabel('Actual Result (1=Win, 0=Loss)')
    # plt.title('Scatter Plot of Predicted Probability vs Actual Result')

    # # 実際の結果の平均を示すラインを追加
    # mean_result = data['actual'].mean()
    # plt.axhline(mean_result, color='red', linestyle='--', label=f'Average Result ({mean_result:.2f})')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # 9. Brierスコアの計算
    brier_score = brier_score_loss(data['actual'], data['predicted_prob'])
    print(f'Brier Score: {brier_score:.4f}')

if __name__ == '__main__':
    create_model()
