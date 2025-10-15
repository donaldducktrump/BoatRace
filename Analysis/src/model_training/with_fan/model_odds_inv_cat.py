# model_1min.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pickle
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
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from catboost import CatBoostClassifier, Pool

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
    # 学習済みCatBoostモデルの読み込み
    try:
        model = CatBoostClassifier()
        model.load_model('boatrace_model_catboost.cbm')
        print("CatBoostモデルを正常に読み込みました。")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    try:
        feature_names = pd.read_csv('feature_names_catboost.csv').squeeze().tolist()
        print("特徴量名を正常に読み込みました。")
    except Exception as e:
        print(f"Failed to load feature names: {e}")
        return

    # 評価期間のデータ日付リスト作成
    month_folders = ['2410']
    date_files = {
        '2410': [f'2410{day:02d}' for day in range(15,27)],   # 241015 - 241026
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
                    if odds_list_part is not None and not odds_list_part.empty:
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

    target = 'odds_inv'
    data = data.dropna()
    data['win_odds']=np.where(data['win_odds']==0, data['win_odds_mean'], data['win_odds'])
    data['win_odds1min']=np.where(data['win_odds1min']==0, data['win_odds_mean'], data['win_odds1min'])
    data['place_odds1min']=np.where(data['place_odds1min']==0, data['win_odds_mean'], data['place_odds1min'])

    data['odds_inv']=0.725/data['win_odds']

    X = data[features]
    y = data[target]

    X['win_odds1min'] = np.where(X['win_odds1min']==0, X['win_odds_mean'], data['win_odds1min'])
    X['win_odds1min'] = 0.725/X['win_odds1min']
    X['place_odds1min'] = np.where(X['place_odds1min']==0, X['win_odds_mean'], data['place_odds1min'])
    X['place_odds1min'] = 0.725/X['place_odds1min']
    X['win_odds_mean'] = 0.725/X['win_odds_mean']

    # pd.set_option('display.max_columns', 100)
    # print(X)
    # カテゴリカル変数のリスト
    categorical_features = ['会場', 'weather', 'wind_d', 'ESC', '艇番', '性別', '支部', '級別']
    X[categorical_features] = X[categorical_features].astype(str)
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # CatBoostのパラメータ設定
    params = {
        'iterations': 10000,
        'learning_rate': 0.05,
        'depth': 6,
        'eval_metric': 'MAE',
        'random_seed': 42,
        'early_stopping_rounds': 100,
        'verbose': 100
    }

    # CatBoostモデルの作成と学習
    cat_model = CatBoostRegressor(**params)
    cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=categorical_features)

    # テストデータで予測
    y_pred = cat_model.predict(X_test)

    # 精度の計算 (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error on test data: {mae:.4f}')

    # 特徴量の重要度を取得してプロット
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': cat_model.get_feature_importance()})
    importance_df = importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 12))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_odds_inv_catboost.png')
    plt.show()

    # 予測値を使って他のモデルを構築
    data['odds_inv_pred'] = cat_model.predict(X)
    data['odds_pred'] = 0.725 / data['odds_inv_pred']

    # Reshape data to fit the linear model
    X = data['odds_pred'].values.reshape(-1, 1)  # Reshape to (n_samples, 1)
    y = data['win_odds'].values

    # Fit a linear regression model
    reg = LinearRegression().fit(X, y)

    # Obtain the scaling factor and intercept from the regression model
    scaling_factor = reg.coef_[0]
    intercept = reg.intercept_

    # Adjust the predicted odds based on the scaling factor and intercept
    data['odds_pred_linear'] = scaling_factor * data['odds_pred'] + intercept

    # Scale your predicted odds for Platt scaling
    from sklearn.isotonic import IsotonicRegression

    # Fit isotonic regression
    iso_reg = IsotonicRegression().fit(data['odds_pred'], data['win_odds'])

    # Calibrate the predictions
    data['odds_pred_adjusted'] = iso_reg.transform(data['odds_pred'])

    # Fit isotonic regression on the linearly adjusted odds

    iso_reg2 = IsotonicRegression().fit(data['odds_pred_linear'], data['win_odds'])

    data['odds_pred_linadj'] = iso_reg2.transform(data['odds_pred_linear'])

    # モデルを保存
    with open('linear_reg_model_cat.pkl', 'wb') as f:
        pickle.dump(reg, f)

    with open('isotonic_reg_model1_cat.pkl', 'wb') as f:
        pickle.dump(iso_reg, f)

    with open('isotonic_reg_model2_cat.pkl', 'wb') as f:
        pickle.dump(iso_reg2, f)

    # モデルの保存
    cat_model.save_model('boatrace_odds_model_inv_catboost.cbm')
    
    # 特徴量データの作成
    X = data[features]  # Xをデータから選択した特徴量で再代入して確実に形状が一致するようにする

    # XがDataFrameであることを確認、必要に応じてDataFrameに変換
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=features)

    # 特徴量名の保存
    X.columns.to_series().to_csv('feature_names_odds_inv_catboost.csv', index=False)

if __name__ == '__main__':
    create_model()
