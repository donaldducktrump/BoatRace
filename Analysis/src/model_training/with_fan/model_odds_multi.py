import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, preprocess_data1min,preprocess_data_old, load_before_data, load_before1min_data ,merge_before_data
from get_data import load_fan_data
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers


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
    month_folders = ['2410']
    date_files = {
        # '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        # '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(15,25)],   # 241001 - 241008
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

    data = preprocess_data1min(data)

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
    data['prob_1'] = y_pred[:, 1]

    # 目的変数の作成: 確定後のオッズ
    # 'before_odds' は確定後のオッズ、 'before1min_odds' は1分前のオッズ
    # これらのカラム名は適宜変更してください
    # 各艇ごとにデータを整形
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
    print(data)
    # '選手登番' と '艇番' ごとに 'win_odds_mean' を集約（例: 平均を取る）
    data_odds_grouped = data_odds.groupby(['選手登番', '艇番'], as_index=False)['win_odds_mean'].mean()
    data['選手登番'] = data['選手登番'].astype(int)
    data_odds_grouped['選手登番'] = data_odds_grouped['選手登番'].astype(int)

    # '選手登番' と '艇番' をキーにして、'win_odds_mean' を data にマージ
    data = data.merge(data_odds_grouped, on=['選手登番', '艇番'], how='left')
    # data = data.merge(data_odds[['win_odds_mean','選手登番','艇番']], on=['選手登番','艇番'], how='left')
    print(data)
    # 不要な列の削除（ターゲット以外に含まれる最終オッズなど）
    # 必要に応じて調整
    # 例: data = data.drop(['before_odds'], axis=1)
    # data = data.dropna(subset=features_multi + ['win_odds'])  # 欠損値の除去
    data = data.dropna()

    # 乱数シードの固定
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    # データの準備（省略）あなたのコードでデータを読み込んでください

    # ここまでのデータ準備はあなたのコードに従います

    # --- データの再構成 ---
    # 'レースID'と'艇番'でソート
    data = data.sort_values(['レースID', '艇番'])

    # データをレースごとにまとめる
    grouped = data.groupby('レースID')

    X_list = []
    y_list = []
    race_ids_list = []

    for race_id, group in grouped:
        if group.shape[0] != 6:
            # 出走艇が6艇でない場合は除外
            continue

        # 各艇の特徴量を取得
        group = group.sort_values('艇番')  # 艇番でソート
        # 使用する特徴量を選択
        features_multi = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                          '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                          'ET', 'tilt', 'EST', 'ESC',
                          'weather', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                          'win_odds1min', 'prob_0', 'place_odds1min', 'win_odds_mean', '性別', '勝率', '複勝率', '優勝回数', '優出回数', '前期能力指数', '今期能力指数', '平均スタートタイミング']

        X_race = group[features_multi].values.flatten()  # 6艇分の特徴量を連結
        y_race = group['win_odds'].values  # 6艇分のオッズ

        X_list.append(X_race)
        y_list.append(y_race)
        race_ids_list.append(race_id)

    X_race = np.array(X_list)
    y_race = np.array(y_list)
    race_ids_array = np.array(race_ids_list)


    # --- 特徴量のエンコーディング ---
    # 特徴量名の作成
    feature_names_multi = []
    for i in range(1, 7):  # 1艇目から6艇目まで
        for feat in features_multi:
            feature_names_multi.append(f'{feat}_{i}')

    # 特徴量のデータフレームを作成
    X_df = pd.DataFrame(X_race, columns=feature_names_multi)

    # カテゴリカル変数のリスト
    categorical_features = []
    for i in range(1, 7):
        for cat_feat in ['会場', 'weather', 'wind_d', 'ESC', '艇番', '性別', '支部', '級別']:
            categorical_features.append(f'{cat_feat}_{i}')

    # One-Hotエンコーディング
    X_categorical = pd.get_dummies(X_df[categorical_features], drop_first=True)

    # 数値データの選択
    numeric_features = [col for col in X_df.columns if col not in categorical_features]
    X_numeric = X_df[numeric_features].astype(float)

    # 特徴量の結合
    X_processed_multi = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

    # --- データの分割 ---
    # データの分割
    X_train, X_val, y_train, y_val, race_ids_train, race_ids_val = train_test_split(
        X_processed_multi, y_race, race_ids_array, test_size=0.2, random_state=SEED)

    # --- モデルの構築 ---
    # LightGBMのパラメータ
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'random_state': SEED,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }

    # LightGBMのマルチアウトプット回帰モデル
    model = MultiOutputRegressor(lgb.LGBMRegressor(**params))

    # --- モデルの学習 ---
    # 学習
    model.fit(X_train, y_train)

    # --- 予測と正規化 ---
    # 予測
    y_pred = model.predict(X_val)

    # 予測値を正の値にクリッピング（オッズは1以上）
    y_pred = np.maximum(y_pred, 1.0)

    # 逆数の和を計算
    reciprocal_sum = np.sum(1 / y_pred, axis=1, keepdims=True)

    # スケーリングファクターを計算
    scaling_factor = 1.37931 / reciprocal_sum

    # 予測値をスケーリング
    y_pred_scaled = y_pred * scaling_factor

    # 再度クリッピング
    y_pred_scaled = np.maximum(y_pred_scaled, 1.0)

    # --- 評価 ---
    from sklearn.metrics import mean_absolute_error

    # MAEの計算
    mae = mean_absolute_error(y_val, y_pred_scaled)
    print(f'Mean Absolute Error on validation data: {mae:.4f}')

    # R²スコアの計算
    r2 = r2_score(y_val, y_pred_scaled)
    print(f'R² Score on validation data: {r2:.4f}')

    # --- 予測結果の整形 ---
    # レースIDと艇番を取得
    boat_numbers = np.tile(np.arange(1, 7), len(X_val))  # 艇番1~6を繰り返す

    # 予測結果をフラット化
    y_pred_flat = y_pred_scaled.reshape(-1)
    y_true_flat = y_val.reshape(-1)
    race_ids_flat = np.repeat(race_ids_val, 6)

    # 結果をデータフレームにまとめる
    pred_df = pd.DataFrame({
        'レースID': race_ids_flat,
        '艇番': boat_numbers,
        'predicted_odds': y_pred_flat,
        'true_odds': y_true_flat
    })

    print(pred_df.head(12))
    # --- 可視化 ---
    # 1. 実際のオッズと予測オッズの散布図（全体）
    plt.figure(figsize=(8, 8))
    plt.scatter(pred_df['true_odds'], pred_df['predicted_odds'], alpha=0.5)
    plt.plot([pred_df['true_odds'].min(), pred_df['true_odds'].max()],
             [pred_df['true_odds'].min(), pred_df['true_odds'].max()], 'r--')
    plt.xlabel('実際のオッズ')
    plt.ylabel('予測オッズ')
    plt.title('実際のオッズと予測オッズの比較（全体）')
    plt.savefig('predicted_vs_true_overall.png')
    plt.show()

    # 2. 艇番ごとの実際のオッズと予測オッズの散布図
    for boat in range(1, 7):
        plt.figure(figsize=(8, 8))
        boat_data = pred_df[pred_df['艇番'] == boat]
        plt.scatter(boat_data['true_odds'], boat_data['predicted_odds'], alpha=0.5)
        plt.plot([boat_data['true_odds'].min(), boat_data['true_odds'].max()],
                 [boat_data['true_odds'].min(), boat_data['true_odds'].max()], 'r--')
        plt.xlabel('実際のオッズ')
        plt.ylabel('予測オッズ')
        plt.title(f'実際のオッズと予測オッズの比較（艇番 {boat}）')
        plt.savefig(f'predicted_vs_true_boat_{boat}.png')
        plt.show()

    # 3. 実際のオッズと予測オッズのヒストグラム（全体）
    plt.figure(figsize=(10, 6))
    plt.hist(pred_df['true_odds'], bins=50, alpha=0.5, label='実際のオッズ')
    plt.hist(pred_df['predicted_odds'], bins=50, alpha=0.5, label='予測オッズ')
    plt.xlabel('オッズ')
    plt.ylabel('頻度')
    plt.title('実際のオッズと予測オッズの分布（全体）')
    plt.legend()
    plt.savefig('odds_distribution_overall.png')
    plt.show()

    # 4. 艇番ごとの実際のオッズと予測オッズのヒストグラム
    for boat in range(1, 7):
        plt.figure(figsize=(10, 6))
        boat_data = pred_df[pred_df['艇番'] == boat]
        plt.hist(boat_data['true_odds'], bins=50, alpha=0.5, label='実際のオッズ')
        plt.hist(boat_data['predicted_odds'], bins=50, alpha=0.5, label='予測オッズ')
        plt.xlabel('オッズ')
        plt.ylabel('頻度')
        plt.title(f'実際のオッズと予測オッズの分布（艇番 {boat}）')
        plt.legend()
        plt.savefig(f'odds_distribution_boat_{boat}.png')
        plt.show()

    # 5. 実際のオッズと予測オッズの誤差分布
    pred_df['error'] = pred_df['predicted_odds'] - pred_df['true_odds']
    plt.figure(figsize=(10, 6))
    sns.histplot(pred_df['error'], bins=50, kde=True)
    plt.xlabel('予測誤差（予測オッズ - 実際のオッズ）')
    plt.ylabel('頻度')
    plt.title('予測誤差の分布')
    plt.savefig('prediction_error_distribution.png')
    plt.show()

# メイン関数として呼び出し
if __name__ == '__main__':
    create_model()
