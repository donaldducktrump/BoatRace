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
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


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
    # データの前処理
    data = data.dropna()
    data = data.sort_values(['レースID', '艇番']).reset_index(drop=True)

    # 特徴量とターゲットの設定
    features_multi = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                      '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                      'ET', 'tilt', 'EST', 'ESC',
                      'weather', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                      'win_odds1min', 'prob_0', 'place_odds1min', 'win_odds_mean',
                      '性別', '勝率', '複勝率', '優勝回数', '優出回数',
                      '前期能力指数', '今期能力指数', '平均スタートタイミング']

    # オッズ関連の特徴量を逆数に変換
    for col in ['win_odds1min', 'place_odds1min', 'win_odds_mean']:
        data[col] = data[col].replace(0, np.nan)
        data[col] = 1 / data[col]
        data[col] = data[col].fillna(data[col].mean())

    # ターゲット（win_odds）を逆数に変換
    data['win_odds'] = data['win_odds'].replace(0, np.nan)
    data['win_odds'] = 1 / data['win_odds']
    data['win_odds'] = data['win_odds'].fillna(data['win_odds'].mean())

    # レースごとにデータをまとめる
    grouped = data.groupby('レースID')

    X_list = []
    y_list = []
    race_ids_list = []
    boat_numbers_list = []

    for race_id, group in grouped:
        if len(group) != 6:
            continue  # 6艇揃っていないレースは除外
        group = group.sort_values('艇番')
        X_race = group[features_multi].values.flatten()
        y_race = group['win_odds'].values  # 逆数化したオッズ
        X_list.append(X_race)
        y_list.append(y_race)
        race_ids_list.append(race_id)
        boat_numbers_list.append(group['艇番'].values)

    # X, y を DataFrame に変換
    X = pd.DataFrame(X_list)
    y = pd.DataFrame(y_list)
    race_ids = pd.Series(race_ids_list)
    boat_numbers = pd.DataFrame(boat_numbers_list)

    # 特徴量名の作成
    feature_names_multi = []
    for i in range(1, 7):
        for feat in features_multi:
            feature_names_multi.append(f'{feat}_{i}')
    X.columns = feature_names_multi

    # カテゴリカル変数のエンコーディング
    categorical_features = []
    for i in range(1,7):
        for cat_feat in ['会場', 'weather', 'wind_d', 'ESC', '艇番', '性別', '支部', '級別']:
            categorical_features.append(f'{cat_feat}_{i}')

    X_categorical = X[categorical_features]
    X_numeric = X.drop(columns=categorical_features)

    # One-Hotエンコーディング
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_categorical_encoded = encoder.fit_transform(X_categorical)

    # NumPy配列を DataFrame に変換
    X_categorical_encoded = pd.DataFrame(
        X_categorical_encoded,
        columns=encoder.get_feature_names_out(categorical_features)
    )

    # 数値データのスケーリング
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=X_numeric.columns)

    # インデックスのリセット
    X_categorical_encoded.reset_index(drop=True, inplace=True)
    X_numeric_scaled.reset_index(drop=True, inplace=True)

    # 特徴量の結合
    X_processed = pd.concat(
        [X_numeric_scaled, X_categorical_encoded],
        axis=1
    )

    # 目的変数
    y.reset_index(drop=True, inplace=True)

    # データの分割
    X_train, X_val, y_train, y_val, race_ids_train, race_ids_val, boat_numbers_train, boat_numbers_val = train_test_split(
        X_processed, y, race_ids, boat_numbers, test_size=0.2, random_state=42
    )

    # PyTorch Dataset の作成
    class BoatRaceDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X.values, dtype=torch.float32)
            self.y = torch.tensor(y.values, dtype=torch.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = BoatRaceDataset(X_train, y_train)
    val_dataset = BoatRaceDataset(X_val, y_val)

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # モデルの定義
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

    # カスタム損失関数の定義
    def custom_loss(y_pred, y_true):
        # MSE損失
        mse_loss = nn.MSELoss()(y_pred, y_true)

        # 逆数の和の制約
        sum_inverse_pred = torch.sum(y_pred, dim=1)  # 各レースごとに合計
        penalty = torch.mean((sum_inverse_pred - 1.37931) ** 2)

        # 総損失
        loss = mse_loss + lambda_penalty * penalty
        return loss

    # モデルの学習
    import torch.optim as optim

    input_dim = X_processed.shape[1]
    model = MultiOutputRegressionModel(input_dim)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    lambda_penalty = 10  # 制約項の重み

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = custom_loss(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # モデルの評価
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            y_pred.append(outputs.numpy())
            y_true.append(y_batch.numpy())

        y_pred = np.vstack(y_pred)
        y_true = np.vstack(y_true)

    # 予測値を元のオッズに変換
    y_pred_odds = 1 / y_pred
    y_pred_odds = np.clip(y_pred_odds, 1, 200)  # 最大値を200に制限

    # 実際の値も元のオッズに変換
    y_true_odds = 1 / y_true
    y_true_odds = np.clip(y_true_odds, 1, 200)  # 最大値を200に制限

    from sklearn.metrics import mean_absolute_error, r2_score

    # MAEの計算
    mae = mean_absolute_error(y_true_odds, y_pred_odds)
    print(f'Mean Absolute Error on validation data: {mae:.4f}')

    # R²スコアの計算
    r2 = r2_score(y_true_odds, y_pred_odds)
    print(f'R² Score on validation data: {r2:.4f}')

    # 結果の整形
    num_samples = y_pred_odds.shape[0]
    boat_numbers = boat_numbers_val.values
    race_ids_val_repeated = race_ids_val.values

    # 予測結果を整形
    pred_list = []
    for i in range(num_samples):
        for j in range(6):
            pred_list.append({
                'レースID': race_ids_val_repeated[i],
                '艇番': boat_numbers[i][j],
                'predicted_odds': y_pred_odds[i][j],
                'true_odds': y_true_odds[i][j]
            })

    pred_df = pd.DataFrame(pred_list)

    # 可視化
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 散布図（全体）
    plt.figure(figsize=(8, 8))
    plt.scatter(pred_df['true_odds'], pred_df['predicted_odds'], alpha=0.5)
    plt.plot([pred_df['true_odds'].min(), pred_df['true_odds'].max()],
             [pred_df['true_odds'].min(), pred_df['true_odds'].max()], 'r--')
    plt.xlabel('実際のオッズ')
    plt.ylabel('予測オッズ')
    plt.title('実際のオッズと予測オッズの比較（全体）')
    plt.savefig('predicted_vs_true_overall_pytorch.png')
    plt.show()

    # 予測誤差の分布（全体）
    pred_df['error'] = pred_df['predicted_odds'] - pred_df['true_odds']
    plt.figure(figsize=(10, 6))
    sns.histplot(pred_df['error'], bins=50, kde=True)
    plt.xlabel('予測誤差（予測オッズ - 実際のオッズ）')
    plt.ylabel('頻度')
    plt.title('予測誤差の分布（全体）')
    plt.savefig('prediction_error_distribution_pytorch.png')
    plt.show()

    # === 追加部分：艇番ごとの結果の図示 ===

    # 艇番ごとの散布図とヒストグラム
    for boat in range(1, 7):
        boat_data = pred_df[pred_df['艇番'] == boat]

        # 散布図
        plt.figure(figsize=(8, 8))
        plt.scatter(boat_data['true_odds'], boat_data['predicted_odds'], alpha=0.5)
        plt.plot([boat_data['true_odds'].min(), boat_data['true_odds'].max()],
                 [boat_data['true_odds'].min(), boat_data['true_odds'].max()], 'r--')
        plt.xlabel('実際のオッズ')
        plt.ylabel('予測オッズ')
        plt.title(f'実際のオッズと予測オッズの比較（艇番 {boat}）')
        plt.savefig(f'predicted_vs_true_boat_{boat}_pytorch.png')
        plt.show()

        # ヒストグラム（予測オッズと実際のオッズ）
        plt.figure(figsize=(10, 6))
        sns.histplot(boat_data['true_odds'], bins=50, color='blue', label='実際のオッズ', alpha=0.5)
        sns.histplot(boat_data['predicted_odds'], bins=50, color='orange', label='予測オッズ', alpha=0.5)
        plt.xlabel('オッズ')
        plt.ylabel('度数')
        plt.title(f'オッズの分布（艇番 {boat}）')
        plt.legend()
        plt.savefig(f'odds_distribution_boat_{boat}_pytorch.png')
        plt.show()

        # 予測誤差のヒストグラム
        # plt.figure(figsize=(10, 6))
        # sns.histplot(boat_data['error'], bins=50, kde=True)
        # plt.xlabel('予測誤差（予測オッズ - 実際のオッズ）')
        # plt.ylabel('度数')
        # plt.title(f'予測誤差の分布（艇番 {boat}）')
        # plt.savefig(f'prediction_error_distribution_boat_{boat}_pytorch.png')
        # plt.show()

if __name__ == '__main__':
    create_model()
