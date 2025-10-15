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
from catboost import CatBoostRanker, Pool

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
def create_model():
    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_first.txt')
        ranker = CatBoostRanker()
        ranker.load_model('boatrace_ranker_catboost.cbm')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_ranker_catboost.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2410']
    date_files = {
        # '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        # '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(15,28,2)],   # 241001 - 241008
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
                'win_odds1min','place_odds1min','weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h', '前期能力指数', '今期能力指数','平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
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
    y_pred = ranker.predict(X_processed)
    data['pred_score'] = y_pred
    # print(y_pred)
    # data['prob_0'] = y_pred[:, 0]  # 1,2着の確率
    # data['prob_1'] = y_pred[:, 1]
    # data['prob_2'] = y_pred[:, 2]

    # data['prob_0'] = y_pred[:, 0]  # 1,着の確率
    # data['prob_1'] = y_pred[:, 1]

    # data['prob_0'] = y_pred[0]  # 1,着の確率
    # data['prob_1'] = y_pred[1]


    probabilities = calculate_rank_probabilities(data)
    print(probabilities)
    data = data.merge(probabilities, on=['レースID', '艇番'], how='left')

    data['prob_0'] = data['順位_1_確率']
    data['prob_1'] = data['順位_2_確率']

    pd.set_option('display.max_rows', 500)
    # 'prob_0' の分布をプロット
    plt.figure(figsize=(8, 6))
    plt.hist(data['順位_1_確率'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title("予測確率 'prob_0' の分布")
    plt.xlabel("予測確率 (prob_0)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('prob_0_distribution_1min.png')
    plt.show()
    print("プロットが 'prob_0_distribution.png' として保存されました。")

    print(data[['レースID', '艇番', 'prob_0','prob_1', '順位_1_確率', '順位_2_確率', '順位_3_確率', '順位_4_確率', '順位_5_確率', '順位_6_確率']].head(30))

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

    # 特徴量と目的変数
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                'win_odds1min', 'prob_0','place_odds1min','win_odds_mean', '性別', '勝率', '複勝率', '優勝回数', '優出回数','前期能力指数', '今期能力指数','平均スタートタイミング']

    target = 'win_odds'
    data = data.dropna()
    X = data[features]
    y = data[target]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather','wind_d', 'ESC', '艇番', '性別', '支部', '級別']
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # 数値データの選択
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]

    # 特徴量の結合
    X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

    # print("X_processed columns:", X_processed.columns.tolist())
    # print(X_processed.head())
    # print(X_processed[['prob_0','prob_1']].head(20))
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.25, random_state=42)

    # LightGBMのデータセット作成
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    # 使用するパラメータ
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'random_state': 42,
        'boosting_type': 'gbdt',
        'verbose': -1,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
    }

    verbose_eval = 100  # 学習時のスコア推移を100ラウンドごとに表示

    # LightGBM学習
    gbm = lgb.train(params,
                    dtrain,
                    valid_sets=[dvalid],
                    num_boost_round=10000,
                    callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    lgb.log_evaluation(verbose_eval)
                ])

    # テストデータで予測
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # 精度の計算 (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error on test data: {mae:.4f}')

    # 特徴量の重要度を取得してプロット
    importance = gbm.feature_importance(importance_type='gain')
    feature_name = gbm.feature_name()
    importance_df = pd.DataFrame({'feature': feature_name, 'importance': importance})
    importance_df = importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 12))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_odds.png')
    plt.show()

    odds_pred = gbm.predict(X_processed)
    data['odds_pred'] = odds_pred

    # # Reshape data to fit the linear model
    # X = data['odds_pred'].values.reshape(-1, 1)  # Reshape to (n_samples, 1)
    # y = data['win_odds'].values

    # # Fit a linear regression model
    # reg = LinearRegression().fit(X, y)

    # # Obtain the scaling factor and intercept from the regression model
    # scaling_factor = reg.coef_[0]
    # intercept = reg.intercept_

    # # Adjust the predicted odds based on the scaling factor and intercept
    # data['odds_pred_linear'] = scaling_factor * data['odds_pred'] + intercept

    # # Scale your predicted odds for Platt scaling
    # from sklearn.isotonic import IsotonicRegression

    # # Fit isotonic regression
    # iso_reg = IsotonicRegression().fit_transform(data['odds_pred_linear'], data['win_odds'])

    # # Calibrate the predictions
    # data['odds_pred_adjusted'] = iso_reg.transform(data['odds_pred_linear'])


    # モデルの保存
    gbm.save_model('boatrace_odds_model_rank_cat.txt')

    # 特徴量名の保存
    X_processed.columns.to_series().to_csv('feature_names_odds_rank_cat.csv', index=False)

if __name__ == '__main__':
    create_model()
