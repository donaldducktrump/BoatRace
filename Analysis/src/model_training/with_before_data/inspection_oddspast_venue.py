# model_1min.py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, preprocess_data1min, load_before_data, load_before1min_data ,merge_before_data
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

def inspect_model():
    # モデルが保存されているディレクトリ
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print(f"モデル保存ディレクトリ '{model_dir}' が存在しません。モデルを作成してください。")
        return

    # トレーニング済みのモデルのリストを取得
    model_files = [f for f in os.listdir(model_dir) if f.startswith('boatrace_predpast_model_') and f.endswith('.txt')]
    if not model_files:
        print(f"モデル保存ディレクトリ '{model_dir}' にモデルファイルが存在しません。モデルを作成してください。")
        return
    
    # venue_models のキーを整数に変換
    venue_models = {}
    for f in model_files:
        try:
            # ファイル名から会場名を抽出し、整数に変換
            venue_str = f[-6:-4]
            venue_int = int(venue_str)
            model_path = os.path.join(model_dir, f)
            venue_models[venue_int] = lgb.Booster(model_file=model_path)
            print(f"モデルロード成功: 会場 {venue_int} -> {model_path}")
        except (IndexError, ValueError) as e:
            print(f"モデルファイル名の解析に失敗しました: {f}. エラー: {e}")

    if not venue_models:
        print("有効な会場ごとのモデルがロードされていません。モデルファイル名を確認してください。")
        return

    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_first.txt')
    #     gbm_odds = lgb.Booster(model_file='boatrace_predpast_model.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_first.csv').squeeze().tolist()
    feature_names_odds = pd.read_csv('feature_names_predpast.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2408','2409','2410']
    date_files = {
        '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(1,20)],   # 241001 - 241008
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

    # 過去のwin_oddsを取得
    month_folders = ['2310','2311','2312','2401','2402','2403','2404','2405','2406','2407','2408','2409','2410']
    date_files = {
        '2310': [f'2310{day:02d}' for day in range(1, 32)],  # 231001 - 231031
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
        '2410': [f'2410{day:02d}' for day in range(1,20)],   # 241001 - 241008
    }
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
    data_odds = preprocess_data(data_odds)
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
    data = preprocess_data(data)

    # print("data columns:", data.columns.tolist())
    # print(data)

    # 特徴量の指定
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h']
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['会場', 'weather', 'wind_d','ESC']
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
    plt.savefig('prob_0_distribution_predpast_venue.png')
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
        # boat_data['before1min_odds'] = boat_data['win_odds1min']  # 1分前のオッズを特徴量
        boat_data['boat_number'] = boat  # 舟番号を特徴量として追加
        data_list.append(boat_data)

    # 各艇のデータを結合
    data = pd.concat(data_list, ignore_index=True)
    # '選手登番' と '艇番' ごとに 'win_odds_mean' を集約（例: 平均を取る）
    data_odds_grouped = data_odds.groupby(['選手登番', '艇番'], as_index=False)['win_odds_mean'].mean()

    # '選手登番' と '艇番' をキーにして、'win_odds_mean' を data にマージ
    data = data.merge(data_odds_grouped, on=['選手登番', '艇番'], how='left')

    # 不要な列の削除（ターゲット以外に含まれる最終オッズなど）
    # 必要に応じて調整
    # 例: data = data.drop(['before_odds'], axis=1)

    # 特徴量と目的変数
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h']
                #  'prob_0','prob_1','win_odds_mean']

    target = 'win_odds'

    X = data[features]
    # y = data[target]

    # デバッグ用: data['会場'] のユニークな値を表示
    print("デバッグ: データ内の会場のユニークな値:")
    print(data['会場'].unique())
    print("デバッグ: data['会場'] のデータ型:", data['会場'].dtype)
    print("デバッグ: venue_models.keys():", venue_models.keys())

    # data['会場'] の型を venue_models のキーと合わせる
    # venue_models のキーは int 型なので、data['会場'] を int 型に変換
    data['会場'] = data['会場'].astype(int)

    for venue in venue_models.keys():
        print(f"会場 {venue} のモデルを使用して予測中...")
        venue_data = data[data['会場'] == venue]

        if venue_data.empty:
            print(f"会場 {venue} のデータが存在しません。スキップします。")
            continue

        # デバッグ用: venue_data 内のレースの日付を表示
        race_dates = pd.to_datetime(venue_data['レースID'].astype(str).str[:8], format='%Y%m%d').unique()
        race_dates = [date.strftime('%Y-%m-%d') for date in race_dates]
        print(f"会場 {venue} で分析しているレースの日付: {race_dates}")

        model = venue_models[venue]

        # 特徴量の選択
        X_venue = venue_data[features]
        # y = venue_data[target]

        # カテゴリカル変数のOne-Hotエンコーディング
        categorical_features = ['会場', 'weather','wind_d', 'ESC', '艇番']
        X_categorical = pd.get_dummies(X_venue[categorical_features], drop_first=True)

        # 数値データの選択
        numeric_features = [col for col in X_venue.columns if col not in categorical_features]
        X_numeric = X_venue[numeric_features]

        # 特徴量の結合
        X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
        venue_str = str(venue).zfill(2)  # '1' -> '01', '9' -> '09', '10' -> '10'

        # feature_names.csv を基準にする
        feature_names_odds = pd.read_csv(f'feature_names/feature_names_predpast_{venue_str}.csv').squeeze().tolist()

        for col in feature_names_odds:
            if col not in X_processed.columns:
                X_processed[col] = 0
        # print("X_processed columns:", X_processed.columns.tolist())
        X_processed = X_processed[feature_names_odds]  # 学習時の特徴量と順序を合わせる
        #　特徴量と目的変数の準備
        X_venue = X_categorical

        y_pred = model.predict(X_processed)

        venue_data = venue_data.copy()
        venue_data['odds_pred'] = y_pred

        # 'レースID'と'艇番'の両方が一致するものを基にマージする
        merged_data = data[data['会場'] == venue].merge(venue_data[['レースID', '艇番', 'odds_pred']], on=['レースID', '艇番'], how='left')

        # 必要に応じて元のDataFrameに'odds_pred'列を追加
        data.loc[data['会場']==venue,['odds_pred']] = venue_data['odds_pred']

    data['diff'] = data['odds_pred'] - data['final_odds']

    # 結果の表示
    # data['diff_between_pred_and_1min']=abs((data['odds_pred']-data['final_odds']))-abs((data['win_odds1min']-data['final_odds']))
    # print(data[['before1min_odds', 'final_odds', 'odds_pred', 'diff','diff_between_pred_and_1min']])

    # 結果の保存
    data.to_csv('result_odds_pred_venue.csv', index=False)

    print("最終オッズの予測結果を保存しました。")

    # ケリー基準に基づいて掛け金割合を計算
    data['bet_fraction'] = data.apply(lambda row: calculate_kelly_criterion(row['prob_0'], row['odds_pred']), axis=1)

    # 掛け金を計算（総資金を100,000円と仮定）
    total_capital = 1000  # 総資金（円）
    data['bet_amount'] = data['bet_fraction'] * total_capital
    # 【追加部分】bet_amount を100円単位に調整
    # bet_amount を100円の倍数に丸め、bet_fractionが0より大きい場合は最低100円に設定
    data['bet_amount'] = data['bet_amount'].apply(round_bet_amount)
    # data['bet_amount'] = np.where(data['prob_0']<0.2, 0, data['bet_amount'])  # 確率が低い場合は掛け金を0にする

    # 掛け金がNaNの場合は0に置換
    data['bet_amount'] = data['bet_amount'].fillna(0)

    # 各艇の勝敗を判定（1着の場合勝利）
    data['win'] = (data['着'] == 1).astype(int)

    # 回収金額を計算
    data['payout'] = data['bet_amount'] * data['win_odds'] * data['win']

    pd.set_option('display.max_rows', 500)
    print(data[['レースID','艇番','prob_0','win_odds','odds_pred','bet_amount', 'payout']].head(50))
    # print(data[['レースID','艇番','prob_0','win_odds1min','win_odds','odds_pred','bet_amount', 'payout']].head(500))

    # 総投資額と総回収額を計算
    total_investment = data['bet_amount'].sum()
    total_return = data['payout'].sum()

    # 回収率を計算
    return_rate = (total_return / total_investment) * 100 if total_investment > 0 else 0

    print(f"投資したレース数: {len(data)/6}")
    print(f"\n総投資額: {total_investment:.2f}円")
    print(f"総回収額: {total_return:.2f}円")
    print(f"回収率: {return_rate:.2f}%")

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

    # 単勝の累積回収をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_payout'], label='累積回収金額', color='green')
    plt.plot(data_sorted.index + 1, data_sorted['cumulative_bet'], label='累積投資金額', color='red')
    plt.title("単勝の累積回収金額 vs 累積投資金額")
    plt.xlabel("ベット回数")
    plt.ylabel("金額 (円)")
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_return_single_win_predpast_venue.png')
    plt.show()
    print("プロットが 'cumulative_return_single_win.png' として保存されました。")

    # 掛け金と回収金額のヒストグラムをプロット
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(data['bet_amount'], bins=50, color='green', edgecolor='black', alpha=0.7)
    plt.title("掛け金の分布")
    plt.xlabel("掛け金額 (円)")
    plt.ylabel("度数")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(data['payout'], bins=50, color='orange', edgecolor='black', alpha=0.7)
    plt.title("回収金額の分布")
    plt.xlabel("回収金額 (円)")
    plt.ylabel("度数")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('bet_payout_distribution_predpast_venue.png')
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
    plt.savefig('cumulative_return_by_date_predpast_venue.png')
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
    plt.savefig('daily_return_barplot_predpast_venue.png')
    plt.show()
    print("プロットが 'daily_return_barplot.png' として保存されました。")

    # 総リターンの表示
    total_daily_return = daily_summary['daily_return'].sum()
    print(f"\n総日付ごとの純利益: {total_daily_return:.2f}円")

    # 会場ごとの投資結果を出力する部分の追加
    # 1. 会場ごとの総投資額、総回収額、回収率を計算する
    venue_summary = data.groupby('会場').agg(
        total_investment=pd.NamedAgg(column='bet_amount', aggfunc='sum'),
        total_return=pd.NamedAgg(column='payout', aggfunc='sum')
    ).reset_index()

    # 2. 回収率を計算（総回収額 / 総投資額 * 100）
    venue_summary['return_rate'] = venue_summary.apply(
        lambda row: (row['total_return'] / row['total_investment']) * 100 if row['total_investment'] > 0 else 0, axis=1
    )

    # 3. 各会場ごとの結果を表示
    print("\n各会場ごとの投資結果:")
    for index, row in venue_summary.iterrows():
        print(f"会場 {int(row['会場'])}: 総投資額: {row['total_investment']:.2f}円, 総回収額: {row['total_return']:.2f}円, 回収率: {row['return_rate']:.2f}%")

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
