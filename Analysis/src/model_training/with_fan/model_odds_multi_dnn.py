# model_1min.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import pandas as pd
import lightgbm as lgb
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

    # 'レースID'と'艇番'でソート
    data = data.sort_values(['レースID', '艇番'])

    # データをレースごとにまとめる
    grouped = data.groupby('レースID')

    X_list = []
    y_list = []

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

    X_race = np.array(X_list)
    y_race = np.array(y_list)

    # 特徴量のリストを取得（6艇分なので特徴量名を調整）
    feature_names_multi = []
    for i in range(1, 7):  # 1艇目から6艇目まで
        for feat in features_multi:
            feature_names_multi.append(f'{feat}_{i}')

    # 特徴量のデータフレームを作成
    X_df = pd.DataFrame(X_race, columns=feature_names_multi)

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features_multi = []
    for i in range(1, 7):
        for cat_feat in ['会場', 'weather', 'wind_d', 'ESC', '艇番', '性別', '支部', '級別']:
            categorical_features_multi.append(f'{cat_feat}_{i}')

    X_categorical = pd.get_dummies(X_df[categorical_features_multi], drop_first=True)

    # 数値データの選択
    numeric_features_multi = [col for col in X_df.columns if col not in categorical_features_multi]
    X_numeric = X_df[numeric_features_multi]

    # 特徴量の結合
    X_processed_multi = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
    # データ型の確認
    print("Type of X_processed_multi:", type(X_processed_multi))
    print("X_processed_multi dtypes:", X_processed_multi.dtypes)

    # numpy.ndarrayに変換してデータ型をfloat32に
    X_processed_multi = X_processed_multi.to_numpy().astype('float32')

    # ラベルデータの型を確認・変換
    print("Type of y_race:", type(y_race))
    print("y_race dtype before conversion:", y_race.dtype)
    y_race = y_race.astype('float32')
    print("y_race dtype after conversion:", y_race.dtype)

    input_dim = X_processed_multi.shape[1]  # 特徴量の次元数


    
    def custom_output(x):
        x = tf.maximum(x, 1.0 + epsilon)
        x = tf.minimum(x, 100.0)  # オッズの上限を100とする
        return x
    
    # モデルの構築
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(6, activation='softplus'),  # Softplus activation
        layers.Lambda(lambda x: x + 1.0)  # 出力に1を加える
    ])

    lambda_penalty = 10  # ペナルティ項の重み（適宜調整）

    epsilon = 1e-7  # 非常に小さな値

    def custom_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

        # オッズの逆数の和を計算
        reciprocal_sum = tf.reduce_sum(1 / y_pred, axis=-1)
        penalty = tf.square(reciprocal_sum - 1.37931)

        return mse + lambda_penalty * penalty



    model.compile(optimizer='adam', loss=custom_loss)


    # データの分割
    X_train, X_val, y_train, y_val = train_test_split(X_processed_multi, y_race, test_size=0.2, random_state=42)
    # print("X_train dtype:", X_train.dtype)
    # print("y_train dtype:", y_train.dtype)
    # X_train = X_train.astype('float32')
    # y_train = y_train.astype('float32')
    # X_val = X_val.astype('float32')
    # y_val = y_val.astype('float32')

    # データ型の確認
    print("Type of X_train:", type(X_train))
    print("X_train dtype:", X_train.dtype)
    print("Type of y_train:", type(y_train))
    print("y_train dtype:", y_train.dtype)

    # 学習
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
    )

    # テストデータで予測
    y_pred = model.predict(X_val)

    # `NaN`や無限大が含まれていないか確認
    print("予測値にNaNが含まれるか:", np.isnan(y_pred).any())
    print("予測値にinfが含まれるか:", np.isinf(y_pred).any())
    # 評価
    mse = np.mean((y_val - y_pred) ** 2)
    print(f'Mean Squared Error: {mse}')

    model.save('multi_output_odds_model.h5')

    reciprocal_sums = np.sum(1 / y_pred, axis=1)
    print(f'平均逆数の和: {np.mean(reciprocal_sums)}')


    # テストデータで予測
    y_pred = model.predict(X_processed_multi)

    # 予測結果の形状を確認
    print("y_pred shape:", y_pred.shape)
    # 元のデータフレームから、'レースID'と'艇番'を取得
    race_ids = []
    boat_numbers = []
    for race_id, group in grouped:
        if group.shape[0] != 6:
            continue
        group = group.sort_values('艇番')
        race_ids.append([race_id]*6)
        boat_numbers.append(group['艇番'].values)

    race_ids = np.array(race_ids).reshape(-1)
    boat_numbers = np.array(boat_numbers).reshape(-1)

    # 予測結果をフラット化
    y_pred_flat = y_pred.reshape(-1)

    # 結果をデータフレームにまとめる
    pred_df = pd.DataFrame({
        'レースID': race_ids,
        '艇番': boat_numbers,
        'predicted_odds': y_pred_flat
    })

    # 予測オッズのデータ型を確認
    print(pred_df.head(500))

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    for boat in range(1, 2):
        plt.hist(pred_df[pred_df['艇番'] == boat]['predicted_odds'], bins=100, alpha=0.5, label=f'艇番 {boat}')

    plt.title('艇番ごとの予測オッズの分布（ヒストグラム）')
    plt.xlabel('予測オッズ')
    plt.ylabel('頻度')
    plt.legend()
    plt.savefig('predicted_odds_distribution_histogram.png')
    plt.show()




if __name__ == '__main__':
    create_model()
