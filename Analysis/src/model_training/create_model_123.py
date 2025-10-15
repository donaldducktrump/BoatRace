import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from ..utils.mymodule import load_b_file, load_k_file, preprocess_data
import numpy as np
import matplotlib.pyplot as plt

def create_model():
    # 2408/240801～2408/240831 および 2409/240901～2409/240930 のデータを使用してモデルを学習
    # トレーニング期間のデータ日付リスト作成
    month_folders = ['2403','2404','2405','2406','2407']
    date_files = {
        '2403': [f'2403{day:02d}' for day in range(1, 32)],  # 240701 - 240731  
        '2404': [f'2404{day:02d}' for day in range(1, 31)],  # 240701 - 240731
        '2405': [f'2405{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        '2406': [f'2406{day:02d}' for day in range(1, 31)],  # 240701 - 240731
        '2407': [f'2407{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        # '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        # '2409': [f'2409{day:02d}' for day in range(1, 16)]   # 240901 - 240930
    }

    # データを結合
    b_data_list = []
    k_data_list = []
    
    for month in month_folders:
        for date in date_files[month]:
            try:
                b_file = f'b_data/{month}/B{date}.TXT'
                k_file = f'k_data/{month}/K{date}.TXT'
                b_data = load_b_file(b_file)
                k_data, _ = load_k_file(k_file)
                
                if not b_data.empty and not k_data.empty:
                    b_data_list.append(b_data)
                    k_data_list.append(k_data)
            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file} または {k_file}")

    # # データの読み込み
    # b_file = 'b_data/2409/B240901.TXT'
    # k_file = 'k_data/2409/K240901.TXT'

    # b_dataとk_dataの結合
    b_data = pd.concat(b_data_list, ignore_index=True)
    k_data = pd.concat(k_data_list, ignore_index=True)

#================================================================================================
    # try:
    #     b_data = load_b_file(b_file)
    #     k_data, odds_data = load_k_file(k_file)
    # except FileNotFoundError:
    #     print("ファイルが見つかりません。指定されたディレクトリにファイルが存在するか確認してください。")
    #     return
    
    # # NaNが含まれているか確認
    # print(b_data['会場'].isna().sum())

    # # NaNを適切に処理 (例：NaNを0で置換)
    # b_data['会場'] = b_data['会場'].fillna(0)

    # # '会場' カラムをint型に変換
    # b_data['会場'] = b_data['会場'].astype(int)

    # # 変換後のデータタイプを確認
    # print(b_data['会場'].dtype)
    
    # # デバッグ: BファイルとKファイルの内容を表示
    # print("Bファイルのデータフレーム（最初の数行）:")
    print(b_data.head(5))
    print("Bファイルのカラム情報:")
    print(b_data.dtypes)

    print("\nKファイルのデータフレーム（最初の数行）:")
    print(k_data.head(-10))
    print("Kファイルのカラム情報:")
    print(k_data.dtypes)
#================================================================================================
    # print(odds_data.head(5))
    # print(odds_data.dtypes)

    # # 各列のNaNの数を表示する
    # print(b_data.isnull().sum())
    # # DataFrame全体の情報を表示
    # b_data.info()
    # 各列のNaNの数を表示する
    # print(k_data.isnull().sum())
    # DataFrame全体の情報を表示
    # k_data.info()
    # print(odds_data.isnull().sum())
    # odds_data.info()

        # b_dataとk_dataのレースIDが異なる行を抽出
    merged_data = pd.merge(b_data, k_data, on='レースID', how='outer', indicator=True)

    # b_dataまたはk_dataにしか存在しない行を抽出
    different_rows = merged_data[merged_data['_merge'] != 'both']

    # 結果を表示
    # print(different_rows)

    if b_data.empty or k_data.empty:
        print("データが正しく読み込めませんでした。")
        return
    
    # データのマージ
    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
    if data.empty:
        print("Merged data is empty after merging B and K data.")
        return
    # _xと_yがついているカラムを探す
    columns_to_check = [col for col in data.columns if '_x' in col]

    for col_x in columns_to_check:
        col_y = col_x.replace('_x', '_y')
        
        # _yカラムが存在するか確認
        if col_y in data.columns:
            # 内容が一致するか確認
            if data[col_x].equals(data[col_y]):
                # 内容が一致している場合、_xの名前に統一して_yを削除
                data[col_x.replace('_x', '')] = data[col_x]  # 統一した名前に
                data.drop([col_x, col_y], axis=1, inplace=True)  # _x, _yを削除
            else:
                # 一致しない場合の処理（エラーメッセージ表示など）
                print(f"Warning: {col_x}と{col_y}の内容が一致しません。手動で確認してください。")

    # 選手名_x と 選手名_y の内容を比較して一致しない場合の処理
    for idx, row in data.iterrows():
        if row['選手名_x'] != row['選手名_y']:
            # 最初の4文字が一致するかを確認
            if row['選手名_x'][:4] == row['選手名_y'][:4]:
                # 選手名_y にそろえて選手名列を作成
                data.at[idx, '選手名'] = row['選手名_y']
            # else:
                # print(f"Warning: 選手名_x と 選手名_y の最初の4文字が一致しません。手動で確認してください。\n{row[['選手名_x', '選手名_y']]}")
        else:
            # 一致している場合は、どちらかを選んで選手名列に
            data.at[idx, '選手名'] = row['選手名_x']

    # 重複列を削除
    data.drop(['選手名_x', '選手名_y'], axis=1, inplace=True)

    # 結果のデバッグ表示
    print("選手名の統一が完了しました。")

    # print(data.columns)
    print(data)
    # nan_rows = data[data['会場'].isnull()]

    # NaNを含む行を削除
    # df_cleaned = data.dropna()

    # 結果を表示
    # print("NaNを削除した後のデータ:")
    # print(df_cleaned)

    # データの前処理
    data = preprocess_data(data)
    if data.empty:
        print("Data is empty after preprocessing.")
        return

    # 目的変数の作成([0,1],[2,3],[4,5]に分類)
    # def target_label(x):
    #     x = int(x)
    #     if x in [1, 2]:
    #         return 0
    #     elif x in [3, 4]:
    #         return 1
    #     else:
    #         return 2
        
    #1,2,3着,それ以外に分類
    def target_label(x):
        x = int(x)
        if x == 1:
            return 0
        elif x == 2:
            return 1
        elif x == 3:
            return 2
        else:
            return 3

    data['target'] = data['着'].apply(target_label)

    # 特徴量と目的変数
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                '展示タイム', '天候', '風向', '風量', '波']
    X = data[features]
    y = data['target']

    # データの分割
    if len(X) < 2:
        print("Not enough data to split. Need at least 2 samples.")
        return

    # テストデータと学習データの分割
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # early_stopping用の評価データをさらに分割
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_raw, y_train_raw, test_size=0.25, random_state=42)

    # データをDatasetクラスに格納
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    # 使用するパラメータ
    params = {'objective': 'multiclass',
              'num_class': 4,
              'metric': 'multi_logloss',
              'random_state': 42,
              'boosting_type': 'gbdt',
              'verbose': -1}

    verbose_eval = 10  # 学習時のスコア推移を10ラウンドごとに表示

    # early_stoppingを指定してLightGBM学習
    gbm = lgb.train(params,
                    dtrain,
                    valid_sets=[dvalid],
                    num_boost_round=10000,
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True),
                               lgb.log_evaluation(verbose_eval)]
                    )

    # テストデータで予測
    y_pred = gbm.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # 精度の計算
    accuracy = accuracy_score(y_test, y_pred_labels)
    print(f'Accuracy on test data: {accuracy:.4f}')

    # 特徴量の重要度を取得してプロット
    importance = gbm.feature_importance(importance_type='gain')
    feature_name = gbm.feature_name()
    importance_df = pd.DataFrame({'feature': feature_name, 'importance': importance})
    importance_df = importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig('feature_importance.png')
    plt.show()

    # モデルの保存
    gbm.save_model('boatrace_model.txt')

    # クロスバリデーションの実装
    print("\nCross-validation with early stopping:")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[test_idx]

        dtrain_cv = lgb.Dataset(X_train_cv, label=y_train_cv)
        dvalid_cv = lgb.Dataset(X_valid_cv, label=y_valid_cv)

        gbm_cv = lgb.train(params,
                           dtrain_cv,
                           valid_sets=[dvalid_cv],
                           num_boost_round=10000,
                           callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                           )

        y_pred_cv = gbm_cv.predict(X_valid_cv)
        y_pred_labels_cv = np.argmax(y_pred_cv, axis=1)
        accuracy_cv = accuracy_score(y_valid_cv, y_pred_labels_cv)
        scores.append(accuracy_cv)
        print(f'Fold {i+1} Accuracy: {accuracy_cv:.4f}')

    print(f'\nMean Accuracy: {np.mean(scores):.4f}')



if __name__ == '__main__':
    create_model()


