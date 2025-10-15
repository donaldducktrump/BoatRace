# create_model.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合
import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, load_before_data, merge_before_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def create_model():
        # モデルを保存するディレクトリの指定
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"モデル保存用ディレクトリ '{model_dir}' を作成しました。")
    else:
        print(f"モデル保存用ディレクトリ '{model_dir}' が既に存在します。")

    # 2403/240301～2407/240731 のデータを使用してモデルを学習
    # トレーニング期間のデータ日付リスト作成
    month_folders = ['2403','2404','2405','2406','2407']
    date_files = {
        '2403': [f'2403{day:02d}' for day in range(1, 32)],  # 240301 - 240331
        '2404': [f'2404{day:02d}' for day in range(1, 31)],  # 240401 - 240430
        '2405': [f'2405{day:02d}' for day in range(1, 32)],  # 240501 - 240531
        '2406': [f'2406{day:02d}' for day in range(1, 31)],  # 240601 - 240630
        '2407': [f'2407{day:02d}' for day in range(1, 32)],  # 240701 - 240731
    }

    # データを結合
    b_data_list = []
    k_data_list = []
    before_data_list = []

    for month in month_folders:
        for date in date_files[month]:
            try:
                b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'
                k_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_data/{month}/K{date}.TXT'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data/{month}/beforeinfo_{date}.txt'

                b_data = load_b_file(b_file)
                k_data, _ = load_k_file(k_file)
                before_data = load_before_data(before_file)
                # print(f"before_data: {before_data}")

                if not b_data.empty and not k_data.empty and not before_data.empty:
                    # データのマージ
                    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
                    # print(f"data: {data}")
                    if '艇番' not in data.columns:
                        print(f"'艇番' 列が存在しません。データを確認してください。")
                        # 必要に応じて処理を中断または修正してください。    
                    data = merge_before_data(data, before_data)
                    b_data_list.append(data)
                else:
                    print(f"データが不足しています: {date}")

            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file}, {k_file}, または {before_file}")

    if not b_data_list:
        print("データが正しく読み込めませんでした。")
        return

    # 全データを結合
    data = pd.concat(b_data_list, ignore_index=True)

    # 前処理
    data = preprocess_data(data)
    print("data columns:", data.columns.tolist())
    print(data)
    # 目的変数の作成([0,1],[2,3],[4,5]に分類)
    def target_label(x):
        x = int(x)
        if x in [1, 2]:
            return 0
        elif x in [3, 4]:
            return 1
        else:
            return 2

    data['target'] = data['着'].apply(target_label)

    # 特徴量と目的変数
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                '展示タイム', 'tilt', 'EST','ESC',
                '天候', '風向', '風量', '波']
    
    # X = data[features]
    # y = data['target']

    # # カテゴリカル変数のOne-Hotエンコーディング
    # categorical_features = ['会場', '天候', '風向', 'ESC']
    # X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # # 数値データの選択
    # numeric_features = [col for col in X.columns if col not in categorical_features]
    # X_numeric = X[numeric_features]

    # # 特徴量の結合
    # X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

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

        # 特徴量と目的変数
        features = ['艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                    '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                    '展示タイム', 'tilt', 'EST','ESC',
                    '天候', '風向', '風量', '波']

        # 特徴量とターゲットの分離
        X_venue = venue_data[features]
        y_venue = venue_data['target']

        # カテゴリカル変数のOne-Hotエンコーディング
        categorical_features = ['天候', '風向', 'ESC']
        X_categorical = pd.get_dummies(X_venue[categorical_features], drop_first=True)

        # 数値データの選択
        numeric_features = [col for col in X_venue.columns if col not in categorical_features]
        X_numeric = X_venue[numeric_features]

        # 特徴量の結合
        X_processed = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

        # feature_names.csv を基準にする
        feature_names = pd.read_csv('feature_names.csv').squeeze().tolist()

        # 欠けている列を補完（全て0で埋める）
        for feature in feature_names:
            if feature not in X_processed.columns:
                X_processed[feature] = 0

        # 列の順序を feature_names に揃える
        X_processed = X_processed[feature_names]

        print(X_processed.columns.tolist())

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
        # 特徴量名の保存
        X_processed.columns.to_series().to_csv(f'feature_names/feature_names_{venue}.csv', index=False)
        print(f"モデル及び特徴量名を保存しました: {model_filename}, feature_names_{venue}.csv")

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
