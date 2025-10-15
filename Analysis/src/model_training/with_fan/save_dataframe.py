import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合
import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, preprocess_data1min,preprocess_data_old, load_before_data, load_before1min_data ,merge_before_data
from get_data import load_fan_data, add_course_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
def remove_common_columns(df_left, df_right, on_columns):
    """
    df_leftとdf_rightのマージ時に、on_columnsを除く共通の列をdf_rightから削除する。
    """
    common_cols = set(df_left.columns).intersection(set(df_right.columns)) - set(on_columns)
    if common_cols:
        print(f"共通列: {common_cols}。df_rightからこれらの列を削除します。")
        df_right = df_right.drop(columns=common_cols)
    return df_right

def save_dataframe(df, file_path):
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # データフレームをピクル形式で保存
    df.to_pickle(file_path)

def load_dataframe(file_path):
    # ファイルが存在する場合は読み込み
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        return None

def main():
    # 2403/240301～2407/240731 のデータを使用してモデルを学習
    # トレーニング期間のデータ日付リスト作成
    month_folders = ['2210', '2211', '2212', '2301', '2302', '2303', '2304', '2305', '2306', '2307', '2308', '2309', '2310', '2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407' ]
    # month_folders = ['2310', '2311', '2312', '2401', '2402', '2403', '2404', '2405', '2406', '2407', '2408', '2409', '2410']
    # month_folders = ['2408', '2409', '2410']

    date_files = {
        '2210': [f'2210{day:02d}' for day in range(1, 32)],  # 221001 - 221031
        '2211': [f'2211{day:02d}' for day in range(1, 31)],  # 221101 - 221130
        '2212': [f'2212{day:02d}' for day in range(1, 32)],  # 221201 - 221231
        '2301': [f'2301{day:02d}' for day in range(1, 31)],  # 230101 - 230131
        '2302': [f'2302{day:02d}' for day in range(1, 29)],  # 230201 - 230228
        '2303': [f'2303{day:02d}' for day in range(1, 32)],  # 230301 - 230331
        '2304': [f'2304{day:02d}' for day in range(1, 31)],  # 230401 - 230430
        '2305': [f'2305{day:02d}' for day in range(1, 32)],  # 230501 - 230531
        '2306': [f'2306{day:02d}' for day in range(1, 31)],  # 230601 - 230630
        '2307': [f'2307{day:02d}' for day in range(1, 32)],  # 230701 - 230731
        '2308': [f'2308{day:02d}' for day in range(1, 32)],  # 230801 - 230831
        '2309': [f'2309{day:02d}' for day in range(1, 31)],  # 230901 - 230930
        '2310': [f'2310{day:02d}' for day in range(1, 32)],  # 231001 - 231031
        '2311': [f'2311{day:02d}' for day in range(1, 31)],  # 231101 - 231130
        '2312': [f'2312{day:02d}' for day in range(1, 32)],  # 231201 - 231231
        '2401': [f'2401{day:02d}' for day in range(1, 31)],  # 240101 - 240131
        '2402': [f'2402{day:02d}' for day in range(1, 29)],  # 240201 - 240228
        '2403': [f'2403{day:02d}' for day in range(1, 32)],  # 240301 - 240331
        '2404': [f'2404{day:02d}' for day in range(1, 31)],  # 240401 - 240430
        '2405': [f'2405{day:02d}' for day in range(1, 32)],  # 240501 - 240531
        '2406': [f'2406{day:02d}' for day in range(1, 31)],  # 240601 - 240630
        '2407': [f'2407{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(1, 24)],  # 241001 - 241014
    }

    # データを結合
    b_data_list = []

    for month in month_folders:
        for date in date_files[month]:
            # ファイルパスの定義
            b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'
            k_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_data/{month}/K{date}.TXT'
            before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'
            before1min_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data1min/{month}/beforeinfo_{date}.txt'

            # 保存先のファイルパス
            b_df_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_dataframe/{month}/B{date}.pkl'
            k_df_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_dataframe/{month}/K{date}.pkl'
            before_df_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_dataframe/{month}/beforeinfo_{date}.pkl'
            before1min_df_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before1min_dataframe/{month}/beforeinfo_{date}.pkl'

            # データフレームを読み込む（存在しない場合はNone）
            b_data = load_dataframe(b_df_file)
            k_data = load_dataframe(k_df_file)
            before_data = load_dataframe(before_df_file)
            before1min_data = load_dataframe(before1min_df_file)

            # データフレームが存在しない場合はファイルから読み込む
            if b_data is None:
                try:
                    b_data = load_b_file(b_file)
                    save_dataframe(b_data, b_df_file)
                except FileNotFoundError:
                    print(f"ファイルが見つかりません: {b_file}")
                    continue

            if k_data is None:
                try:
                    k_data, _ = load_k_file(k_file)
                    save_dataframe(k_data, k_df_file)
                except FileNotFoundError:
                    print(f"ファイルが見つかりません: {k_file}")
                    continue

            if before_data is None:
                try:
                    before_data = load_before_data(before_file)
                    save_dataframe(before_data, before_df_file)
                except FileNotFoundError:
                    print(f"ファイルが見つかりません: {before_file}")
                    continue
            
            if before1min_data is None:
                try:
                    before1min_data = load_before1min_data(before1min_file)
                    save_dataframe(before1min_data, before1min_df_file)
                except FileNotFoundError:
                    print(f"ファイルが見つかりません: {before1min_file}")
                    continue

            if not b_data.empty and not k_data.empty and not before_data.empty:
                # データのマージ
                data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
                before_data = remove_common_columns(data, before_data, on_columns=['選手登番', 'レースID', '艇番'])
                data = merge_before_data(data, before_data)
                before1min_data = remove_common_columns(data, before1min_data, on_columns=['選手登番', 'レースID', '艇番'])
                data = merge_before_data(data, before1min_data)

                # ファンデータを読み込み
                fan_df_file = 'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/fan_dataframe/fan2404.pkl'
                fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
                df_fan = load_dataframe(fan_df_file)
                if df_fan is None:
                    df_fan = load_fan_data(fan_file)
                    save_dataframe(df_fan, fan_df_file)

                # 必要な列を指定してマージ
                columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
                data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')

                # 各行に対してコース別データを追加
                data = data.apply(lambda row: add_course_data(row, df_fan), axis=1)

                # 最終的なデータフレームを保存
                data_df_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/dataframe/{month}/data_{date}.pkl'
                save_dataframe(data, data_df_file)

                b_data_list.append(data)
            else:
                print(f"データが不足しています: {date}")

    if not b_data_list:
        print("データが正しく読み込めませんでした。")
        return

    # 全データを結合
    data = pd.concat(b_data_list, ignore_index=True)

    # 前処理
    data = preprocess_data(data)
    print("data columns:", data.columns.tolist())
    # print(data)

    # 最終的なデータフレームを保存
    final_data_file = 'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/dataframe/final_data.pkl'
    save_dataframe(data, final_data_file)

if __name__ == '__main__':
    main()
