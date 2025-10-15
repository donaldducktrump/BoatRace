import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合
import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from ..utils.mymodule import load_b_file, load_k_file, preprocess_data
import numpy as np
import matplotlib.pyplot as plt

def create_model():
        # モデルを保存するディレクトリの指定
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"モデル保存用ディレクトリ '{model_dir}' を作成しました。")
    else:
        print(f"モデル保存用ディレクトリ '{model_dir}' が既に存在します。")

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

    # データの結合
    b_data = pd.concat(b_data_list, ignore_index=True) if b_data_list else pd.DataFrame()
    k_data = pd.concat(k_data_list, ignore_index=True) if k_data_list else pd.DataFrame()

    # データの読み込み結果のデバッグ表示
    print(b_data.head(5))
    print("Bファイルのカラム情報:")
    print(b_data.dtypes)

    print("\nKファイルのデータフレーム（最初の数行）:")
    print(k_data.head(-10))  # 不要な行を除外する場合は適宜調整
    print("Kファイルのカラム情報:")
    print(k_data.dtypes)

    # データのマージ（選手登番とレースIDで結合）
    merged_data = pd.merge(b_data, k_data, on='レースID', how='outer', indicator=True)

    # BデータまたはKデータにしか存在しない行を抽出（デバッグ用）
    different_rows = merged_data[merged_data['_merge'] != 'both']
    if not different_rows.empty:
        print("BデータとKデータにしか存在しない行が存在します。確認してください。")
        print(different_rows)

    if b_data.empty or k_data.empty:
        print("データが正しく読み込めませんでした。")
        return

    # 正しくデータが読み込まれた場合のマージ
    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
    if data.empty:
        print("Merged data is empty after merging B and K data.")
        return

    # カラム名の整合性を確認・修正
    columns_to_check = [col for col in data.columns if '_x' in col]

    for col_x in columns_to_check:
        col_y = col_x.replace('_x', '_y')
        
        if col_y in data.columns:
            if data[col_x].equals(data[col_y]):
                data[col_x.replace('_x', '')] = data[col_x]
                data.drop([col_x, col_y], axis=1, inplace=True)
            # else:
                # print(f"Warning: {col_x}と{col_y}の内容が一致しません。手動で確認してください。")

    # 選手名の統一
    for idx, row in data.iterrows():
        if row['選手名_x'] != row['選手名_y']:
            if row['選手名_x'][:4] == row['選手名_y'][:4]:
                data.at[idx, '選手名'] = row['選手名_y']
            # else:
                # print(f"Warning: 選手名_x と 選手名_y の最初の4文字が一致しません。手動で確認してください。\n{row[['選手名_x', '選手名_y']]}")
        else:
            data.at[idx, '選手名'] = row['選手名_x']

    # 重複列を削除
    data.drop(['選手名_x', '選手名_y'], axis=1, inplace=True)

    # 結果のデバッグ表示
    print("選手名の統一が完了しました。")
    print(data.head())

    # データの前処理
    data = preprocess_data(data)
    if data.empty:
        print("Data is empty after preprocessing.")
        return

    # 目的変数の作成（1,2着を0、3,4着を1、5着以上を2に分類）
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

    # 特徴量と目的変数の分離
    features = ['会場','艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                '展示タイム', '天候', '風向', '風量', '波']
    X = data[features]
    y = data['target']

    # 会場ごとにモデルをトレーニング
    venues = data['会場'].unique()
    print(f"トレーニング対象の会場数: {len(venues)}")

    for venue in venues:
        print(f"\n会場: {venue} のモデルをトレーニング中...")
        venue_data = data[data['会場'] == venue]

        if venue_data.empty:
            print(f"会場 {venue} のデータが存在しません。スキップします。")
            continue

        # 会場ごとのレース数を計算（ユニークなレースIDの数）
        number_of_races = venue_data['レースID'].nunique()
        print(f"会場 {venue} で分析対象となるレースの数: {number_of_races} レース")

        # 特徴量とターゲットの分離
        X_venue = venue_data[features]
        y_venue = venue_data['target']

        # データの分割
        if len(X_venue) < 2:
            print(f"会場 {venue} のデータが不足しています。スキップします。")
            continue

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_venue, y_venue, test_size=0.25, random_state=42, stratify=y_venue)

        # LightGBM用のデータセット作成
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        # 使用するパラメータ
        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'multi_logloss',
            'random_state': 42,
            'boosting_type': 'gbdt',
            'verbose': -1
        }

        verbose_eval = 10  # 学習時のスコア推移を10ラウンドごとに表示

        # LightGBM学習
        gbm = lgb.train(params,
                        dtrain,
                        valid_sets=[dvalid],
                        num_boost_round=10000,
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=10, verbose=True),
                            lgb.log_evaluation(verbose_eval)
                        ])

        # テストデータで予測
        y_pred_valid = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
        y_pred_labels = np.argmax(y_pred_valid, axis=1)

        # 精度の計算
        accuracy = accuracy_score(y_valid, y_pred_labels)
        print(f'バリデーション精度: {accuracy:.4f}')

        # 特徴量の重要度を取得してプロット
        importance = gbm.feature_importance(importance_type='gain')
        feature_names = gbm.feature_name()
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        importance_df = importance_df.sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title(f'First Model Feature Importance for Venue {venue}')
        plt.gca().invert_yaxis()
        plt.savefig(f'feature_importance/first_model_feature_importance_{venue}.png')
        plt.show()

        # モデルの保存
        model_filename = f'{model_dir}/first_model_{venue}.txt'
        gbm.save_model(model_filename)
        print(f"モデルを保存しました: {model_filename}")

        # クロスバリデーションの実装
        print(f"\n会場 {venue} のモデル: クロスバリデーションを実施中...")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for i, (train_idx, test_idx) in enumerate(cv.split(X_venue, y_venue)):
            X_train_cv, X_valid_cv = X_venue.iloc[train_idx], X_venue.iloc[test_idx]
            y_train_cv, y_valid_cv = y_venue.iloc[train_idx], y_venue.iloc[test_idx]

            dtrain_cv = lgb.Dataset(X_train_cv, label=y_train_cv)
            dvalid_cv = lgb.Dataset(X_valid_cv, label=y_valid_cv)

            gbm_cv = lgb.train(params,
                               dtrain_cv,
                               valid_sets=[dvalid_cv],
                               num_boost_round=10000,
                               callbacks=[
                                   lgb.early_stopping(stopping_rounds=10, verbose=False),
                                   lgb.log_evaluation(verbose_eval)
                               ])

            y_pred_cv = gbm_cv.predict(X_valid_cv, num_iteration=gbm_cv.best_iteration)
            y_pred_labels_cv = np.argmax(y_pred_cv, axis=1)
            accuracy_cv = accuracy_score(y_valid_cv, y_pred_labels_cv)
            scores.append(accuracy_cv)
            print(f'Fold {i+1} Accuracy: {accuracy_cv:.4f}')

        print(f'\nクロスバリデーション平均精度: {np.mean(scores):.4f}')

if __name__ == '__main__':
    create_model()
