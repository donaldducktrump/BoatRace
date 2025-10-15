# model_1min.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import joblib
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
import pickle
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool


def load_processed_odds_data():
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
    # 学習済みモデルの読み込み
    try:
        model = CatBoostClassifier()
        model.load_model('boatrace_model_catboost.cbm')
        cat_model = CatBoostRegressor()
        cat_model.load_model('boatrace_odds_model_inv_catboost.cbm')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_catboost.csv').squeeze().tolist()
    feature_names_odds = pd.read_csv('feature_names_odds_inv_catboost.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2410']
    date_files = {
        # '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        # '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(15,27)]  # 241001 - 241008
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

    # データの確認
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
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h', '前期能力指数', '今期能力指数','平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    X = data[features]

    # カテゴリカル変数の指定
    categorical_features = ['会場', 'weather', 'wind_d','ESC', '艇番', '性別', '支部', '級別']
    
    # カテゴリカル変数を文字列型に変換
    for col in categorical_features:
        X.loc[:, col] = X[col].astype(str)

    # 特徴量の確認
    print("X_processed columns:", X.columns.tolist())
    print(X.head())

    # 特徴量名を使用して列を一致させる
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0  # 存在しない特徴量は0で埋める

    # 特徴量の順序を合わせる
    X = X[feature_names]

    # カテゴリカル特徴量のインデックスを取得
    cat_features_indices = [feature_names.index(col) for col in categorical_features]

    # CatBoostのPoolを作成
    pool = Pool(X, cat_features=cat_features_indices)

    # 予測
    try:
        y_pred = model.predict(pool)
        y_pred_proba = model.predict_proba(pool)
        print("予測が正常に完了しました。")
    except Exception as e:
        print(f"予測時にエラーが発生しました: {e}")
        return

    # 予測結果をデータフレームに追加
    data['prob_0'] = y_pred_proba[:, 0]  # クラス0の確率
    data['prob_1'] = y_pred_proba[:, 1]  # クラス1の確率
    # 'prob_0' の分布をプロット
    plt.figure(figsize=(8, 6))
    plt.hist(data['prob_0'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title("予測確率 'prob_0' の分布")
    plt.xlabel("予測確率 (prob_0)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('prob_0_distribution_1min_inv.png')
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

    # 不要な列の削除（ターゲット以外に含まれる最終オッズなど）
    # 必要に応じて調整
    # 例: data = data.drop(['before_odds'], axis=1)


    # 特徴量と目的変数
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                'win_odds1min', 'prob_0','place_odds1min','win_odds_mean','前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    target = 'odds_inv'
    data = data.dropna()
    data['win_odds']=np.where(data['win_odds']==0, data['win_odds_mean'], data['win_odds'])
    data['win_odds1min']=np.where(data['win_odds1min']==0, data['win_odds_mean'], data['win_odds1min'])
    data['place_odds1min']=np.where(data['place_odds1min']==0, data['win_odds_mean'], data['place_odds1min'])

    data['odds_inv']=0.725/data['win_odds']

    X = data[features]
    y = data[target]

    # 'win_odds1min' と 'place_odds1min' の NaN 値やゼロの処理
    X.loc[:, 'win_odds1min'] = np.where(X['win_odds1min'] == 0, X['win_odds_mean'], X['win_odds1min'])
    X.loc[:, 'win_odds1min'] = 0.725 / X['win_odds1min']
    X.loc[:, 'place_odds1min'] = np.where(X['place_odds1min'] == 0, X['win_odds_mean'], X['place_odds1min'])
    X.loc[:, 'place_odds1min'] = 0.725 / X['place_odds1min']
    X.loc[:, 'win_odds_mean'] = 0.725 / X['win_odds_mean']

    categorical_features = ['会場', 'weather', 'wind_d', 'ESC', '艇番', '性別', '支部', '級別']
    # カテゴリカル変数を文字列型に変換
    for col in categorical_features:
        X.loc[:, col] = X[col].astype(str)

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    # print("X_processed columns:", X_processed.columns.tolist())
    X = X[feature_names_odds]  # 学習時の特徴量と順序を合わせる

    # CatBoostモデルでの予測
    odds_inv_pred = cat_model.predict(X)

    data['odds_inv_pred'] = odds_inv_pred
    data['odds_pred'] = 0.725 / data['odds_inv_pred']
    data['odds_pred'] = data['odds_pred'].apply(lambda x: 1 if x < 1 else x)

    # 差分計算
    data['diff'] = data['win_odds'] - data['odds_pred']
    data['diff_between_pred_and_1min'] = abs((data['odds_pred'] - data['final_odds'])) - abs((data['win_odds1min'] - data['final_odds']))

    print(data[['before1min_odds', 'final_odds', 'odds_pred', 'diff', 'diff_between_pred_and_1min']])

    # 結果の保存
    data.to_csv('result_odds_pred_inv.csv', index=False)
    print("最終オッズの予測結果を保存しました。")

        # Reshape data to fit the linear model
    X = data['odds_pred'].values.reshape(-1, 1)  # Reshape to (n_samples, 1)
    y = data['win_odds'].values

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
    # iso_reg = IsotonicRegression().fit(data['odds_pred'], data['win_odds'])

    # # Calibrate the predictions
    # data['odds_pred_adjusted'] = iso_reg.transform(data['odds_pred'])

    # reg2 = LinearRegression().fit(data['odds_pred_adjusted'].values.reshape(-1, 1), data['win_odds'].values)
    # scaling_factor2 = reg2.coef_[0]
    # intercept2 = reg2.intercept_

    # data['odds_pred_linear2'] = scaling_factor2 * data['odds_pred_adjusted'] + intercept2

    # # Scale your predicted odds for Platt scaling
    # from sklearn.isotonic import IsotonicRegression

    # # Fit isotonic regression
    # iso_reg = IsotonicRegression().fit(data['odds_pred_linear'], data['win_odds'])

    # # Calibrate the predictions
    # data['odds_pred_linadj'] = iso_reg.transform(data['odds_pred_linear'])

    # モデルの読み込み
    with open('linear_reg_model_cat.pkl', 'rb') as f:
        loaded_reg = pickle.load(f)

    with open('isotonic_reg_model1_cat.pkl', 'rb') as f:
        loaded_iso_reg = pickle.load(f)

    with open('isotonic_reg_model2_cat.pkl', 'rb') as f:
        loaded_iso_reg2 = pickle.load(f)


    data_pred = data['odds_pred'].values.reshape(-1, 1)
    data['odds_pred_linear'] = loaded_reg.predict(data_pred)

    data['odds_pred_adjusted'] = loaded_iso_reg.transform(data['odds_pred'])
    data['odds_pred_linadj'] = loaded_iso_reg2.transform(data['odds_pred_linear'])
    # data.loc[data['艇番'] == 1, 'odds_pred_adjusted'] = data.loc[data['艇番'] == 1, 'odds_pred_adjusted'] / 2
    # data['odds_pred_adjusted'] = data['odds_pred_adjusted'].apply(lambda x: 1 if x < 1 else x)

 
    total_capital = 1000  # 総資産
    # ケリー基準に基づいて掛け金割合を計算
    data['bet_fraction'] = data.apply(lambda row: calculate_kelly_criterion(row['prob_0'], row['odds_pred']), axis=1)
    data['bet_fraction_ans'] = data.apply(lambda row: calculate_kelly_criterion(row['prob_0'], row['win_odds']), axis=1)
    data['bet_fraction_mean'] = data.apply(lambda row: calculate_kelly_criterion(row['prob_0'], row['win_odds_mean']), axis=1)


    data['bet_amount'] = data['bet_fraction'] * total_capital
    # 【追加部分】bet_amount を100円単位に調整
    # bet_amount を100円の倍数に丸め、bet_fractionが0より大きい場合は最低100円に設定
    data['bet_amount'] = data['bet_amount'].apply(round_bet_amount)
     # 確率が低い場合は掛け金を0にする
    data['bet_amount'] = np.where(data['prob_0'] < 0.2, 0, data['bet_amount'])  # 確率が低い場合は掛け金を0にする
    data['bet_amount'] = np.where(data['bet_amount']<=100, 0, data['bet_amount'])  # 掛け金が100円未満の場合は0にする

    data['bet_amount_ans'] = data['bet_fraction_ans'] * total_capital
    data['bet_amount_ans'] = data['bet_amount_ans'].apply(round_bet_amount)
    data['bet_amount_ans'] = np.where(data['prob_0'] < 0.2, 0, data['bet_amount_ans'])  # 確率が低い場合は掛け金を0にする
    data['bet_amount_ans'] = np.where(data['bet_amount_ans']<=100, 0, data['bet_amount_ans'])  # 掛け金が100円未満の場合は0にする

    data['bet_amount_mean'] = data['bet_fraction_mean'] * total_capital
    data['bet_amount_mean'] = data['bet_amount_mean'].apply(round_bet_amount)
    data['bet_amount_mean'] = np.where(data['prob_0'] < 0.2, 0, data['bet_amount_mean'])  # 確率が低い場合は掛け金を0にする
    data['bet_amount_mean'] = np.where(data['bet_amount_mean']<=100, 0, data['bet_amount_mean'])  # 掛け金が100円未満の場合は0にする

    # 掛け金がNaNの場合は0に置換
    data['bet_amount'] = data['bet_amount'].fillna(0)
    data['bet_amount_ans'] = data['bet_amount_ans'].fillna(0)
    data['bet_amount_mean'] = data['bet_amount_mean'].fillna(0)

    # 各艇の勝敗を判定（1着の場合勝利）
    data['win'] = (data['着'] == 1).astype(int)

    # 回収金額を計算
    data['payout'] = data['bet_amount'] * data['win_odds'] * data['win']
    data['payout_ans'] = data['bet_amount_ans'] * data['win_odds'] * data['win']
    data['payout_mean'] = data['bet_amount_mean'] * data['win_odds'] * data['win']

    pd.set_option('display.max_rows', 50)
    filtered_data = data[data['艇番'] == 1]
    print(filtered_data[['レースID','艇番','prob_0','win_odds_mean','win_odds1min','win_odds','odds_pred','bet_amount', 'bet_amount_ans','bet_amount_mean','payout']].head(50))
    
    # data['odds_pred'] = data['odds_pred'].apply(lambda x: np.round(x, decimals=1))
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_data['odds_pred'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title("予測オッズ 'odds_pred' の分布")
    plt.xlabel("予測オッズ (odds_pred)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('odds_pred_distribution_1min_inv.png')

    #　実際のオッズの分布
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_data['win_odds'], bins=500, color='blue', edgecolor='black', alpha=0.7)
    plt.title("実際のオッズ 'win_odds' の分布")
    plt.xlabel("実際のオッズ (win_odds)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('odds_actual_distribution_1min_inv.png')


    # 総投資額と総回収額を計算
    total_investment = data['bet_amount'].sum()
    total_investment_ans = data['bet_amount_ans'].sum()
    total_investment_mean = data['bet_amount_mean'].sum()

    total_return = data['payout'].sum()
    total_return_ans = data['payout_ans'].sum()
    total_return_mean = data['payout_mean'].sum()

    # 回収率を計算
    return_rate = (total_return / total_investment) * 100 if total_investment > 0 else 0
    return_rate_ans = (total_return_ans / total_investment_ans) * 100 if total_investment_ans > 0 else 0
    return_rate_mean = (total_return_mean / total_investment_mean) * 100 if total_investment_mean > 0 else 0


    plt.figure(figsize=(10, 6))
    plt.scatter(data['win_odds'], data['odds_pred'], alpha=0.5)
    plt.plot([data['win_odds'].min(), data['win_odds'].max()],
            [data['win_odds'].min(), data['win_odds'].max()], 'r--')
    plt.xlabel('実際のオッズ (win_odds)')
    plt.ylabel('予測オッズ (odds_pred)')
    plt.title('予測オッズと実際のオッズの比較')
    plt.savefig('odds_pred_vs_actual_odds_1min_inv.png')
    plt.show()

    data['residuals'] = data['win_odds'] - data['odds_pred']

    plt.figure(figsize=(10, 6))
    plt.scatter(data['odds_pred'], data['residuals'], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('予測オッズ (odds_pred)')
    plt.ylabel('残差 (win_odds - odds_pred)')
    plt.title('残差プロット')
    plt.savefig('residuals_plot_1min_inv.png')
    plt.show()

    # linear
    # plt.figure(figsize=(10, 6))
    # plt.scatter(data['win_odds'], data['odds_pred_linear'], alpha=0.5)
    # plt.plot([data['win_odds'].min(), data['win_odds'].max()],
    #         [data['win_odds'].min(), data['win_odds'].max()], 'r--')
    # plt.xlabel('実際のオッズ (win_odds)')
    # plt.ylabel('調整後の予測オッズ (odds_pred_linear)')
    # plt.title('調整後の予測オッズと実際のオッズの比較')
    # plt.savefig('odds_pred_linear_vs_actual_odds_1min.png')
    # # plt.show()

    # data['residuals_linear'] = data['win_odds'] - data['odds_pred_linear']

    # plt.figure(figsize=(10, 6))
    # plt.scatter(data['odds_pred_linear'], data['residuals_linear'], alpha=0.5)
    # plt.axhline(y=0, color='r', linestyle='--')
    # plt.xlabel('調整後の予測オッズ (odds_pred_linear)')
    # plt.ylabel('残差 (win_odds - odds_pred_linear)')
    # plt.title('残差プロット')
    # plt.savefig('residuals_plot_adjusted_1min.png')
    # plt.show()

    # isometric
    # plt.figure(figsize=(10, 6))
    # plt.scatter(data['win_odds'], data['odds_pred_adjusted'], alpha=0.5)
    # plt.plot([data['win_odds'].min(), data['win_odds'].max()],
    #         [data['win_odds'].min(), data['win_odds'].max()], 'r--')
    # plt.xlabel('実際のオッズ (win_odds)')
    # plt.ylabel('isotonic調整後の予測オッズ (odds_pred_adjusted)')
    # plt.title('isotonic調整後の予測オッズと実際のオッズの比較')
    # plt.savefig('odds_pred_isotonic_vs_actual_odds_1min.png')
    # # plt.show()

    data['residuals_isotonic'] = data['win_odds'] - data['odds_pred_adjusted']

    # plt.figure(figsize=(10, 6))
    # plt.scatter(data['odds_pred_adjusted'], data['residuals_isotonic'], alpha=0.5)
    # plt.axhline(y=0, color='r', linestyle='--')
    # plt.xlabel('isotonic調整後の予測オッズ (odds_pred_adjusted)')
    # plt.ylabel('残差 (win_odds - odds_pred_adjusted)')
    # plt.title('残差プロット')
    # plt.savefig('residuals_plot_isotonic_1min.png')
    # # plt.show()

    # linadj
    plt.figure(figsize=(10, 6))
    plt.scatter(data['win_odds'], data['odds_pred_linadj'], alpha=0.5)
    plt.plot([data['win_odds'].min(), data['win_odds'].max()],
            [data['win_odds'].min(), data['win_odds'].max()], 'r--')
    plt.xlabel('実際のオッズ (win_odds)')
    plt.ylabel('isotonic+linear調整後の予測オッズ (odds_pred_linadj)')
    plt.title('isotonic+linear調整後の予測オッズと実際のオッズの比較')
    plt.savefig('odds_pred_linadj_vs_actual_odds_1min_inv.png')
    plt.show()

    data['residuals_linadj'] = data['win_odds'] - data['odds_pred_linadj']

    plt.figure(figsize=(10, 6))
    plt.scatter(data['odds_pred_linadj'], data['residuals_linadj'], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('isotonic+linear調整後の予測オッズ (odds_pred_linadj)')
    plt.ylabel('残差 (win_odds - odds_pred_linadj)')
    plt.title('残差プロット')
    plt.savefig('residuals_plot_linadj_1min_inv.png')
    plt.show()

    print(f"投資したレース数: {len(data)/6}")
    print(f"\n総投資額: {total_investment:.2f}円")
    print(f"総回収額: {total_return:.2f}円")
    print(f"回収率: {return_rate:.2f}%")

    print(f"\n総投資額_ans: {total_investment_ans:.2f}円")
    print(f"総回収額_ans: {total_return_ans:.2f}円")
    print(f"回収率_ans: {return_rate_ans:.2f}%")
    
    print(f"\n総投資額_mean: {total_investment_mean:.2f}円")
    print(f"総回収額_mean: {total_return_mean:.2f}円")
    print(f"回収率_mean: {return_rate_mean:.2f}%")

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
    data_sorted['cumulative_bet_ans'] = data_sorted['bet_amount_ans'].cumsum()
    data_sorted['cumulative_payout_ans'] = data_sorted['payout_ans'].cumsum()
    data_sorted['cumulative_bet_mean'] = data_sorted['bet_amount_mean'].cumsum()
    data_sorted['cumulative_payout_mean'] = data_sorted['payout_mean'].cumsum()

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

    # 単勝の累積回収をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_payout_ans'], label='累積回収金額', color='green')
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_bet_ans'], label='累積投資金額', color='red')
    plt.title("単勝の累積回収金額 vs 累積投資金額")
    plt.xlabel("ベット回数")
    plt.ylabel("金額 (円)")
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_return_single_win_ans_1min_inv.png')
    plt.show()
    print("プロットが 'cumulative_return_single_win_ans.png' として保存されました。")

    # 単勝の累積回収をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_payout_mean'], label='累積回収金額', color='green')
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_bet_mean'], label='累積投資金額', color='red')
    plt.title("単勝の累積回収金額 vs 累積投資金額")
    plt.xlabel("ベット回数")
    plt.ylabel("金額 (円)")
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_return_single_win_mean_1min_inv.png')
    plt.show()
    print("プロットが 'cumulative_return_single_win_mean.png' として保存されました。")

    # 掛け金と回収金額のヒストグラムをプロット
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(data[data['艇番']==1]['bet_amount'], bins=50, color='green', edgecolor='black', alpha=0.7)
    plt.title("掛け金の分布")
    plt.xlabel("掛け金額 (円)")
    plt.ylabel("度数")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(data[data['艇番']==1]['payout'], bins=50, color='orange', edgecolor='black', alpha=0.7)
    plt.title("回収金額の分布")
    plt.xlabel("回収金額 (円)")
    plt.ylabel("度数")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('bet_payout_distribution_1min_inv.png')
    plt.show()
    print("プロットが 'bet_payout_distribution.png' として保存されました。")

    # 【追加部分】日付ごとのリターンの計算とプロット
    # 日付を datetime 型に変換
    # 'レースID' から 'date' カラムを生成
    if 'レースID' in data.columns:
        data['レースID'] = data['レースID'].astype(str)
        data['date'] = pd.to_datetime(data['レースID'].str[:8], format='%Y%m%d')
    else:
        print("Error: 'レースID' カラムがデータに存在しません。")
        return
    # 日付ごとの総投資額と総回収額を計算
    daily_summary = data.groupby('date').agg(
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

    # 日付ごとの累積リターンをプロット
    plt.figure(figsize=(12, 6))
    plt.plot(daily_summary['date'], daily_summary['cumulative_return'], marker='o', linestyle='-', color='purple')
    plt.title("日付ごとの累積リターンの推移")
    plt.xlabel("日付")
    plt.ylabel("累積リターン (円)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cumulative_return_by_date_1min_inv.png')
    plt.show()
    print("プロットが 'cumulative_return_by_date.png' として保存されました。")

    # 【オプション】日付ごとの純利益のヒストグラムをプロット
    plt.figure(figsize=(8, 6))
    sns.barplot(x='date_str', y='daily_return', data=daily_summary, palette='viridis')
    plt.title("日付ごとの純利益")
    plt.xlabel("日付")
    plt.ylabel("純利益 (円)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('daily_return_barplot_1min_inv.png')
    plt.show()
    print("プロットが 'daily_return_barplot.png' として保存されました。")

    # 総リターンの表示
    total_daily_return = daily_summary['daily_return'].sum()
    print(f"\n総日付ごとの純利益: {total_daily_return:.2f}円")

# ケリー基準の計算関数を定義
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

if __name__ == '__main__':
    inspect_model()
