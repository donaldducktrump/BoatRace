#model_odds_pastdata.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, preprocess_data1min, preprocess_data_old,load_before_data, load_before1min_data ,merge_before_data
from get_data import load_fan_data
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder

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

def create_model():
    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_first.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_first.csv').squeeze().tolist()

    # 過去のwin_oddsを取得
    # month_folders = ['2210', '2211', '2212', '2301', '2302', '2303', '2304', '2305', '2306', '2307', '2308', '2309', '2310','2311','2312','2401','2402','2403','2404','2405','2406','2407']
    month_folders =['2402','2403','2404','2405','2406','2407']

    date_files = {
        # '2210': [f'2210{day:02d}' for day in range(1, 32)],  # 221001 - 221031
        # '2211': [f'2211{day:02d}' for day in range(1, 31)],  # 221101 - 221130
        # '2212': [f'2212{day:02d}' for day in range(1, 32)],  # 221201 - 221231
        # '2301': [f'2301{day:02d}' for day in range(1, 31)],  # 230101 - 230131
        # '2302': [f'2302{day:02d}' for day in range(1, 29)],  # 230201 - 230228
        # '2303': [f'2303{day:02d}' for day in range(1, 32)],  # 230301 - 230331
        # '2304': [f'2304{day:02d}' for day in range(1, 31)],  # 230401 - 230430
        # '2305': [f'2305{day:02d}' for day in range(1, 32)],  # 230501 - 230531
        # '2306': [f'2306{day:02d}' for day in range(1, 31)],  # 230601 - 230630
        # '2307': [f'2307{day:02d}' for day in range(1, 32)],  # 230701 - 230731
        # '2308': [f'2308{day:02d}' for day in range(1, 32)],  # 230801 - 230831
        # '2309': [f'2309{day:02d}' for day in range(1, 31)],  # 230901 - 230930
        # '2310': [f'2310{day:02d}' for day in range(1, 32)],  # 231001 - 231031
        # '2311': [f'2311{day:02d}' for day in range(1, 31)],  # 231101 - 231130
        # '2312': [f'2312{day:02d}' for day in range(1, 32)],  # 231201 - 231231
        # '2401': [f'2401{day:02d}' for day in range(1, 32)],  # 240101 - 240131
        '2402': [f'2402{day:02d}' for day in range(1, 29)],  # 240201 - 240228
        '2403': [f'2403{day:02d}' for day in range(1, 32)],  # 240301 - 240331
        '2404': [f'2404{day:02d}' for day in range(1, 31)],  # 240401 - 240430
        '2405': [f'2405{day:02d}' for day in range(1, 32)],  # 240501 - 240531
        '2406': [f'2406{day:02d}' for day in range(1, 31)],  # 240601 - 240630
        '2407': [f'2407{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        # '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        # '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        # '2410': [f'2410{day:02d}' for day in range(1,21)],   # 241001 - 241008
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
                # before1min_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data_1min/{month}/beforeinfo1min_{date}.txt'
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
    

    data_list_odds = []
    for month in month_folders:
        for date in date_files[month]:
            try:
                b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'
                bdata_odds = load_b_file(b_file)
                before_data_odds = load_before_data(before_file)

                if not bdata_odds.empty and not before_data.empty:
                    # データのマージ
                    before_data_odds = remove_common_columns(bdata_odds, before_data_odds, on_columns=['選手登番', 'レースID', '艇番'])
                    bdata_odds = merge_before_data(bdata_odds, before_data_odds)
                    data_list_odds.append(bdata_odds)
                else:
                    print(f"データが不足しています: {date}")
            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file}, または {before_file}")
    data_odds = pd.concat(data_list_odds, ignore_index=True)
    data_odds = preprocess_data_old(data_odds)
    # print("data_odds columns:", data_odds.columns.tolist())
    # print("data_odds:", data_odds)
    data_odds['win_odds_mean']=data_odds.groupby(['選手登番','艇番'])['win_odds'].transform(lambda x: x.mean())
    # data_odds.sort_values('選手登番')
    # print(data_odds)
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
    # data = preprocess_data1min(data)
    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)

    # 必要な列を指定してマージ
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')

    # 各行に対してコース別データを追加
    # data = data.apply(lambda row: add_course_data(row, df_fan), axis=1)

    data = preprocess_data(data)

    # print("data columns:", data.columns.tolist())
    # print(data)

    # 特徴量の指定
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h','前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather', 'wind_d','ESC','性別','支部','級別','艇番']
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
        # boat_data['before1min_odds'] = boat_data['win_odds1min']  # 1分前のオッズを特徴量
        boat_data['boat_number'] = boat  # 舟番号を特徴量として追加
        data_list.append(boat_data)

    # 各艇のデータを結合
    data = pd.concat(data_list, ignore_index=True)
    print(data)
    # '選手登番' と '艇番' ごとに 'win_odds_mean' を集約（例: 平均を取る）
    data_odds_grouped = data_odds.groupby(['選手登番', '艇番'], as_index=False)['win_odds_mean'].mean()

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
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h','前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数','prob_0','win_odds_mean']
                # 'prob_0','prob_1','win_odds_mean']
    
    # # 多クラス分類の場合
    # def target_label(x):
    #     if 1<=x<=1.25:
    #         return 0
    #     elif 1.25<x<=1.5:
    #         return 1
    #     elif 1.5<x<=1.75:
    #         return 2
    #     elif 1.75<x<=2:
    #         return 3
    #     elif 2<=x<=2.5:
    #         return 4
    #     elif 2.5<x<=3:
    #         return 5
    #     elif 3<x<=4:
    #         return 6
    #     elif 4<x<=5:
    #         return 7
    #     elif 5<x<=7.5:
    #         return 8
    #     elif 7.5<x<=10:
    #         return 9
    #     else:
    #         return 10

    target = 'win_odds'
    # data['target'] = data['win_odds'].apply(target_label)
    X = data[features]
    y = data[target]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather','wind_d', 'ESC', '艇番','性別','支部','級別']
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

    # #　多クラス分類
    # # 使用するパラメータ
    # params = {'objective': 'multiclass',
    #           'num_class': 11,
    #           'metric': 'multi_logloss',
    #           'random_state': 42,
    #           'boosting_type': 'gbdt',
    #           'verbose': -1}

    # verbose_eval = 10  # 学習時のスコア推移を10ラウンドごとに表示

    # # early_stoppingを指定してLightGBM学習
    # gbm = lgb.train(params,
    #                 dtrain,
    #                 valid_sets=[dvalid],
    #                 num_boost_round=10000,
    #                 callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True),
    #                            lgb.log_evaluation(verbose_eval)]
    #                 )

    # y_pred = gbm.predict(X_test)
    # y_pred_labels = np.argmax(y_pred, axis=1)

    # # 精度の計算
    # accuracy = accuracy_score(y_test, y_pred_labels)
    # print(f'Accuracy on test data: {accuracy:.4f}')

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
    plt.savefig('feature_importance_pred_past.png')
    plt.show()

    # モデルの保存
    gbm.save_model('boatrace_predpast_model.txt')

    # 特徴量名の保存
    X_processed.columns.to_series().to_csv('feature_names_predpast.csv', index=False)

if __name__ == '__main__':
    create_model()
