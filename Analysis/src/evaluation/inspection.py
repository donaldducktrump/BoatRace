# inspection.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from ..utils.mymodule import load_b_file, load_k_file, preprocess_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

# mymodule.py

# def create_confidence_features(data):
#     """
#     第一モデルの予測確率を基に、第二モデルの特徴量を作成します。
#     各レースごとにprob_0が大きい順に上からA, B, C, Dを選出し、
#     それぞれのクラス0, 1, 2の確率を特徴量として抽出します。
    
#     Args:
#         data (pd.DataFrame): 第一モデルの予測確率が含まれたデータフレーム。
    
#     Returns:
#         pd.DataFrame: 第二モデル用の特徴量とターゲットを含むデータフレーム。
#     """
#     confidence_features = []
    
#     race_ids = data['レースID'].unique()
#     for race_id in race_ids:
#         race_data = data[data['レースID'] == race_id]
#         # prob_0が高い順にソートし、上位4艇（A, B, C, D）を選出
#         sorted_race = race_data.sort_values('prob_0', ascending=False).head(4)
        
#         # 必要な選手が4艇分揃っているか確認
#         if len(sorted_race) < 4:
#             print(f"レースID {race_id} は4艇未満のためスキップします。")
#             continue
        
#         # A, B, C, Dのクラス確率を取得
#         features = {}
#         boats = ['A', 'B', 'C', 'D']
#         for i, row in sorted_race.iterrows():
#             boat_label = chr(65 + i)  # A, B, C, D
#             features[f'A_prob0'] = sorted_race.iloc[0]['prob_0']
#             features[f'A_prob1'] = sorted_race.iloc[0]['prob_1']
#             features[f'A_prob2'] = sorted_race.iloc[0]['prob_2']
#             features[f'B_prob0'] = sorted_race.iloc[1]['prob_0']
#             features[f'B_prob1'] = sorted_race.iloc[1]['prob_1']
#             features[f'B_prob2'] = sorted_race.iloc[1]['prob_2']
#             features[f'C_prob0'] = sorted_race.iloc[2]['prob_0']
#             features[f'C_prob1'] = sorted_race.iloc[2]['prob_1']
#             features[f'C_prob2'] = sorted_race.iloc[2]['prob_2']
#             features[f'D_prob0'] = sorted_race.iloc[3]['prob_0']
#             features[f'D_prob1'] = sorted_race.iloc[3]['prob_1']
#             features[f'D_prob2'] = sorted_race.iloc[3]['prob_2']
#             break  # A, B, C, Dを一度だけ設定
#         features['レースID'] = race_id
    
#         # ターゲット変数: Aが実際に1着であるか
#         actual_first = race_data[race_data['着'] == 1]['艇番'].values
#         if len(actual_first) == 0:
#             print(f"レースID {race_id} に実際の1着情報がありません。スキップします。")
#             continue
#         actual_first = actual_first[0]
#         predicted_first = sorted_race.iloc[0]['艇番']
#         features['target'] = 1 if predicted_first == actual_first else 0
    
#         confidence_features.append(features)
    
#     confidence_df = pd.DataFrame(confidence_features)
#     return confidence_df


# def create_confidence_model():
#     # 第二モデルのトレーニング期間のデータ
#     # train_date_range = ['240701', '240702', ..., '240904']  # 2024年7月1日〜9月4日
#     # 実際には適切な日付リストを使用してください
#     train_date_range = [f'2409{day:02d}' for day in range(16, 26)] 
#                     #                                             + \
#                     #    [f'2408{day:02d}' for day in range(1, 32)] + \
#                     #    [f'2409{day:02d}' for day in range(1, 5)]

#     b_data_list = []
#     k_data_list = []
#     odds_list1 = []

#     for date in train_date_range:
#         try:
#             b_file = f'b_data/{date[:4]}/B{date}.TXT'
#             k_file = f'k_data/{date[:4]}/K{date}.TXT'
#             b_data = load_b_file(b_file)
#             k_data, odds_list_part = load_k_file(k_file)
            
#             if not b_data.empty and not k_data.empty:
#                 b_data_list.append(b_data)
#                 k_data_list.append(k_data)
#             if not odds_list_part.empty:
#                 odds_list1.append(odds_list_part)
                
#         except FileNotFoundError:
#             print(f"ファイルが見つかりません: {date}")

#     # データの結合
#     b_data = pd.concat(b_data_list, ignore_index=True)
#     k_data = pd.concat(k_data_list, ignore_index=True)
#     odds_list = pd.concat(odds_list1, ignore_index=True)

#     if b_data.empty or k_data.empty:
#         print("トレーニング用データが正しく読み込めませんでした。")
#         return

#     if odds_list.empty:
#         print("トレーニング用オッズデータが正しく読み込めませんでした。")

#     # データのマージ
#     data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
#     if data.empty:
#         print("Merged data is empty after merging B and K data.")
#         return
#     # _xと_yがついているカラムを探す
#     columns_to_check = [col for col in data.columns if '_x' in col]

#     for col_x in columns_to_check:
#         col_y = col_x.replace('_x', '_y')
        
#         # _yカラムが存在するか確認
#         if col_y in data.columns:
#             # 内容が一致するか確認
#             if data[col_x].equals(data[col_y]):
#                 # 内容が一致している場合、_xの名前に統一して_yを削除
#                 data[col_x.replace('_x', '')] = data[col_x]  # 統一した名前に
#                 data.drop([col_x, col_y], axis=1, inplace=True)  # _x, _yを削除
#             else:
#                 # 一致しない場合の処理（エラーメッセージ表示など）
#                 print(f"Warning: {col_x}と{col_y}の内容が一致しません。手動で確認してください。")

#     # 選手名_x と 選手名_y の内容を比較して一致しない場合の処理
#     for idx, row in data.iterrows():
#         if row['選手名_x'] != row['選手名_y']:
#             # 最初の4文字が一致するかを確認
#             if row['選手名_x'][:4] == row['選手名_y'][:4]:
#                 # 選手名_y にそろえて選手名列を作成
#                 data.at[idx, '選手名'] = row['選手名_y']
#             # else:
#                 # print(f"Warning: 選手名_x と 選手名_y の最初の4文字が一致しません。手動で確認してください。\n{row[['選手名_x', '選手名_y']]}")
#         else:
#             # 一致している場合は、どちらかを選んで選手名列に
#             data.at[idx, '選手名'] = row['選手名_x']

#     # 重複列を削除
#     data.drop(['選手名_x', '選手名_y'], axis=1, inplace=True)

#     # 結果のデバッグ表示
#     print("選手名の統一が完了しました。")

#     # print(data.columns)
#     print(data)
#     # nan_rows = data[data['会場'].isnull()]

#     # NaNを含む行を削除
#     # df_cleaned = data.dropna()

#     # 結果を表示
#     # print("NaNを削除した後のデータ:")
#     # print(df_cleaned)

#     # データの前処理
#     data = preprocess_data(data)
#     if data.empty:
#         print("Data is empty after preprocessing.")
#         return

#     # 第一モデルによる予測確率の取得
#     features = ['会場','艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
#                 '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
#                 '展示タイム', '天候', '風向', '風量', '波']
#     X = data[features]
    
#     # 第一モデルの予測
#     gbm = lgb.Booster(model_file='boatrace_model.txt')
#     y_pred = gbm.predict(X)
#     data['prob_0'] = y_pred[:, 0]
#     data['prob_1'] = y_pred[:, 1]
#     data['prob_2'] = y_pred[:, 2]


#     # 第二モデル用の特徴量作成
#     confidence_df = create_confidence_features(data)

#     # ターゲットと特徴量の分離
#     if 'target' not in confidence_df.columns:
#         print("ターゲット列が存在しません。")
#         return

#     X_conf = confidence_df.drop(['レースID', 'target'], axis=1)
#     y_conf = confidence_df['target']

#     # 特徴量数の出力
#     print(f"第二モデル作成に使用した特徴量の数: {X_conf.shape[1]}")

#     # データの分割
#     if len(X_conf) < 2:
#         print("Not enough data to split. Need at least 2 samples.")
#         return

#     X_train, X_valid, y_train, y_valid = train_test_split(
#         X_conf, y_conf, test_size=0.25, random_state=42, stratify=y_conf)

#     # LightGBM用のデータセット作成
#     dtrain = lgb.Dataset(X_train, label=y_train)
#     dvalid = lgb.Dataset(X_valid, label=y_valid)

#     # 使用するパラメータ
#     params = {
#         'objective': 'binary',
#         'metric': 'binary_logloss',
#         'random_state': 42,
#         'boosting_type': 'gbdt',
#         'verbose': -1
#     }

#     verbose_eval = 10  # 学習時のスコア推移を10ラウンドごとに表示

#     # early_stoppingを指定してLightGBM学習
#     gbm_conf = lgb.train(params,
#                          dtrain,
#                          valid_sets=[dvalid],
#                          num_boost_round=10000,
#                          callbacks=[
#                              lgb.early_stopping(stopping_rounds=10, verbose=True),
#                              lgb.log_evaluation(verbose_eval)
#                          ])

#     # テストデータで予測
#     y_pred_conf = gbm_conf.predict(X_valid, num_iteration=gbm_conf.best_iteration)
#     y_pred_labels_conf = (y_pred_conf >= 0.5).astype(int)

#     # 精度の計算
#     accuracy_conf = accuracy_score(y_valid, y_pred_labels_conf)
#     auc_conf = roc_auc_score(y_valid, y_pred_conf)
#     print(f'第二モデルのバリデーション精度: {accuracy_conf:.4f}')
#     print(f'第二モデルのAUC: {auc_conf:.4f}')

#     # 特徴量の重要度を取得してプロット
#     importance_conf = gbm_conf.feature_importance(importance_type='gain')
#     feature_name_conf = gbm_conf.feature_name()
#     importance_df_conf = pd.DataFrame({'feature': feature_name_conf, 'importance': importance_conf})
#     importance_df_conf = importance_df_conf.sort_values('importance', ascending=False)

#     # plt.figure(figsize=(10, 6))
#     # plt.barh(importance_df_conf['feature'], importance_df_conf['importance'])
#     # plt.xlabel('Feature Importance')
#     # plt.ylabel('Feature')
#     # plt.title('Confidence Model Feature Importance')
#     # plt.gca().invert_yaxis()
#     # plt.savefig('confidence_feature_importance.png')
#     # plt.show()

#     # モデルの保存
#     gbm_conf.save_model('confidence_model.txt')

#     # クロスバリデーションの実装
#     print("\nSecond Model: Cross-validation with early stopping:")
#     cv = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores_conf = []
#     for i, (train_idx, test_idx) in enumerate(cv.split(X_conf, y_conf)):
#         X_train_cv, X_valid_cv = X_conf.iloc[train_idx], X_conf.iloc[test_idx]
#         y_train_cv, y_valid_cv = y_conf.iloc[train_idx], y_conf.iloc[test_idx]

#         dtrain_cv = lgb.Dataset(X_train_cv, label=y_train_cv)
#         dvalid_cv = lgb.Dataset(X_valid_cv, label=y_valid_cv)

#         gbm_cv_conf = lgb.train(params,
#                                 dtrain_cv,
#                                 valid_sets=[dvalid_cv],
#                                 num_boost_round=10000,
#                                 callbacks=[
#                                     lgb.early_stopping(stopping_rounds=10, verbose=False),
#                                     lgb.log_evaluation(verbose_eval)
#                                 ])

#         y_pred_cv_conf = gbm_cv_conf.predict(X_valid_cv, num_iteration=gbm_cv_conf.best_iteration)
#         y_pred_labels_cv_conf = (y_pred_cv_conf >= 0.5).astype(int)
#         accuracy_cv_conf = accuracy_score(y_valid_cv, y_pred_labels_cv_conf)
#         scores_conf.append(accuracy_cv_conf)
#         print(f'Fold {i+1} Accuracy: {accuracy_cv_conf:.4f}')

#     print(f'\nSecond Model Mean Accuracy: {np.mean(scores_conf):.4f}')

# def inspect_with_confidence_model():
#     # 学習済みモデルの読み込み
#     try:
#         gbm = lgb.Booster(model_file='boatrace_model.txt')
#         gbm_conf = lgb.Booster(model_file='confidence_model.txt')
#     except Exception as e:
#         print(f"モデルの読み込みに失敗しました: {e}")
#         return

#     # 評価期間のデータ
#     # eval_date_range = ['241005', '241006', '241007', '241008']  # 2024年9月26日〜10月8日のデータ
#     # 実際には適切な日付リストを使用してください
#     eval_date_range = [f'2409{day:02d}' for day in range(26, 31)] + [f'2410{day:02d}' for day in range(1, 9)]

#     b_data_list = []
#     k_data_list = []
#     odds_list1 = []

#     for date in eval_date_range:
#         try:
#             b_file = f'b_data/{date[:4]}/B{date}.TXT'
#             k_file = f'k_data/{date[:4]}/K{date}.TXT'
#             b_data = load_b_file(b_file)
#             k_data, odds_list_part = load_k_file(k_file)
            
#             if not b_data.empty and not k_data.empty:
#                 b_data_list.append(b_data)
#                 k_data_list.append(k_data)
#             if not odds_list_part.empty:
#                 odds_list1.append(odds_list_part)
                
#         except FileNotFoundError:
#             print(f"ファイルが見つかりません: {date}")

#     # データの結合
#     b_data = pd.concat(b_data_list, ignore_index=True)
#     k_data = pd.concat(k_data_list, ignore_index=True)
#     odds_list = pd.concat(odds_list1, ignore_index=True)

#     if b_data.empty or k_data.empty:
#         print("評価用データが正しく読み込めませんでした。")
#         return

#     if odds_list.empty:
#         print("評価用オッズデータが正しく読み込めませんでした。")

#     # データのマージ
#     data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
#     if data.empty:
#         print("Merged data is empty after merging B and K data.")
#         return
#     # _xと_yがついているカラムを探す
#     columns_to_check = [col for col in data.columns if '_x' in col]

#     for col_x in columns_to_check:
#         col_y = col_x.replace('_x', '_y')
        
#         # _yカラムが存在するか確認
#         if col_y in data.columns:
#             # 内容が一致するか確認
#             if data[col_x].equals(data[col_y]):
#                 # 内容が一致している場合、_xの名前に統一して_yを削除
#                 data[col_x.replace('_x', '')] = data[col_x]  # 統一した名前に
#                 data.drop([col_x, col_y], axis=1, inplace=True)  # _x, _yを削除
#             else:
#                 # 一致しない場合の処理（エラーメッセージ表示など）
#                 print(f"Warning: {col_x}と{col_y}の内容が一致しません。手動で確認してください。")

#     # 選手名_x と 選手名_y の内容を比較して一致しない場合の処理
#     for idx, row in data.iterrows():
#         if row['選手名_x'] != row['選手名_y']:
#             # 最初の4文字が一致するかを確認
#             if row['選手名_x'][:4] == row['選手名_y'][:4]:
#                 # 選手名_y にそろえて選手名列を作成
#                 data.at[idx, '選手名'] = row['選手名_y']
#             # else:
#                 # print(f"Warning: 選手名_x と 選手名_y の最初の4文字が一致しません。手動で確認してください。\n{row[['選手名_x', '選手名_y']]}")
#         else:
#             # 一致している場合は、どちらかを選んで選手名列に
#             data.at[idx, '選手名'] = row['選手名_x']

#     # 重複列を削除
#     data.drop(['選手名_x', '選手名_y'], axis=1, inplace=True)

#     # 結果のデバッグ表示
#     print("選手名の統一が完了しました。")

#     # print(data.columns)
#     print(data)
#     # nan_rows = data[data['会場'].isnull()]

#     # NaNを含む行を削除
#     # df_cleaned = data.dropna()

#     # 結果を表示
#     # print("NaNを削除した後のデータ:")
#     # print(df_cleaned)

#     # データの前処理
#     data = preprocess_data(data)
#     if data.empty:
#         print("Data is empty after preprocessing.")
#         return

#     # 第一モデルによる予測確率の取得
#     features = ['会場','艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
#                 '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
#                 '展示タイム', '天候', '風向', '風量', '波']
#     X = data[features]
    
#     # 第一モデルの予測
#     y_pred = gbm.predict(X)
#     data['prob_0'] = y_pred[:, 0]
#     data['prob_1'] = y_pred[:, 1]
#     data['prob_2'] = y_pred[:, 2]

#     # 第二モデル用の特徴量作成
#     confidence_df = create_confidence_features(data)

#     # ターゲットと特徴量の分離
#     if 'target' not in confidence_df.columns:
#         print("ターゲット列が存在しません。")
#         return

#     X_conf = confidence_df.drop(['レースID', 'target'], axis=1)
#     y_conf = confidence_df['target']

#     # 第二モデルによる予測
#     y_conf_pred_prob = gbm_conf.predict(X_conf, num_iteration=gbm_conf.best_iteration)
#     # 確信度として確率を使用
#     confidence_df['confidence'] = y_conf_pred_prob

#     # 投資戦略:
#     # 確信度が一定以上のレースにのみ単勝を投資
#     threshold = 0.64  # 例として70%以上の確信度を閾値とする

#     betting_races = confidence_df[confidence_df['confidence'] >= threshold]

#     print(f"投資対象となるレース数: {len(betting_races)}")

#     # 回収率計算のための変数
#     total_bet = len(betting_races) * 100  # 1レースあたり100円
#     total_return = 0
#     payout_list = []

#     for idx, row in betting_races.iterrows():
#         race_id = row['レースID']
#         predicted_first = data[data['レースID'] == race_id].sort_values('prob_0', ascending=False).iloc[0]['艇番']
#         actual_first = data[data['レースID'] == race_id][data['着'] == 1]['艇番'].values

#         if len(actual_first) == 0:
#             print(f"レースID {race_id} に実際の1着情報がありません。スキップします。")
#             payout_list.append(0)
#             continue

#         actual_first = actual_first[0]

#         if predicted_first == actual_first:
#             # 単勝オッズを取得
#             odds_row = odds_list[odds_list['レースID'] == race_id]
#             if odds_row.empty:
#                 print(f"オッズ情報が見つかりません: レースID {race_id}")
#                 payout_list.append(0)
#                 continue

#             winning_odds = odds_row["単勝オッズ"].values[0]
#             if pd.isna(winning_odds):
#                 print(f"単勝オッズがNaNです: レースID {race_id}")
#                 payout_list.append(0)
#                 continue

#             payout = 100 * winning_odds
#             total_return += payout
#             payout_list.append(payout)
#         else:
#             payout_list.append(0)

#     # 回収率の計算
#     return_rate = (total_return / total_bet) * 100 if total_bet > 0 else 0
    
#     # 的中率の計算
#     correct_predictions = sum(payout > 0 for payout in payout_list)
#     hit_rate = (correct_predictions / len(betting_races)) * 100 if len(betting_races) > 0 else 0
#     print(f"的中率: {hit_rate:.2f}%")

#     print(f"\n投資戦略による総賭け金: {total_bet}円")
#     print(f"投資戦略による総回収金額: {total_return}円")
#     print(f"投資戦略による回収率: {return_rate:.2f}%")

#     # 配当金の統計情報
#     payout_series = pd.Series(payout_list)
#     print(f"\n投資戦略の配当金の統計情報:")
#     print(f"最大値: {payout_series.max()}円")
#     print(f"最小値: {payout_series.min()}円")
#     print(f"平均: {payout_series.mean()}円")
#     print(f"標準偏差: {payout_series.std()}円")

#     # 配当金のヒストグラムをプロット
#     plt.figure(figsize=(10, 6))
#     sns.histplot(payout_series, bins=20, kde=False)
#     plt.title("投資戦略の配当金のヒストグラム")
#     plt.xlabel("配当金額 (円)")
#     plt.ylabel("度数")
#     plt.grid(True)
#     plt.savefig('confidence_odds_histogram.png')
#     plt.show()
#     print("ヒストグラムが 'confidence_odds_histogram.png' として保存されました。")


def inspect_model():
    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model.txt')
        # gbm = lgb.LGBMClassifier(model_file='boatrace_model.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    # モデルと前処理器の読み込み
    model = keras.models.load_model('boatrace_dnn_model.h5')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
    # 評価期間のデータ日付リスト作成
    month_folders = ['2408','2409', '2410']
    date_files = {
        '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240701 - 240731
        '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(1, 9)],   # 241001 - 241008
    }

    # データを結合
    b_data_list = []
    k_data_list = []
    odds_list1 = []

    for month in month_folders:
        for date in date_files[month]:
            try:
                b_file = f'b_data/{month}/B{date}.TXT'
                k_file = f'k_data/{month}/K{date}.TXT'
                b_data = load_b_file(b_file)
                k_data, odds_list_part = load_k_file(k_file)

                if not b_data.empty and not k_data.empty:
                    b_data_list.append(b_data)
                    k_data_list.append(k_data)
                if not odds_list_part.empty:
                    odds_list1.append(odds_list_part)

            except FileNotFoundError:
                print(f"ファイルが見つかりません: {date}")

    
    # b_dataとk_dataの結合
    b_data = pd.concat(b_data_list, ignore_index=True)
    k_data = pd.concat(k_data_list, ignore_index=True)
    odds_list = pd.concat(odds_list1, ignore_index=True)

    if b_data.empty or k_data.empty:
        print("データが正しく読み込めませんでした。")
        return
    if odds_list.empty:
        print("データが正しく読み込めませんでした。")

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
            else:
                print(f"Warning: 選手名_x と 選手名_y の最初の4文字が一致しません。手動で確認してください。\n{row[['選手名_x', '選手名_y']]}")
        else:
            # 一致している場合は、どちらかを選んで選手名列に
            data.at[idx, '選手名'] = row['選手名_x']

    # 重複列を削除
    data.drop(['選手名_x', '選手名_y'], axis=1, inplace=True)

    # 結果のデバッグ表示
    print("選手名の統一が完了しました。")
    # data = pd.merge(data, odds_list, on='レースID')  # オッズデータも結合
    
    # print(data.columns)
    print(data)
    print(odds_list)
    print(odds_list.isnull().sum())
    odds_list.info()

    # # データのマージ
    # data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
    # if data.empty:
    #     print("Merged data is empty after merging B and K data.")
    #     return

    # データの前処理
    data = preprocess_data(data)
    if data.empty:
        print("Data is empty after preprocessing.")
        return

    # 特徴量の指定
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                '展示タイム', '天候', '風向', '風量', '波']
    X = data[features]

    # 予測
    y_pred = gbm.predict(X)
    print(y_pred)
    data['prob_0'] = y_pred[:, 0]  # 1着or2着の確率
    data['prob_1'] = y_pred[:, 1]  
    data['prob_2'] = y_pred[:, 2]
    
    # y_pred = gbm.fit(X)
    # probs = gbm.predict_proba(X)
    correct_predictions = 0
    total_races = 0
    # 各日にちごとの正答率を格納する辞書

    daily_accuracy = {}
    # 回収金額と賭け金を計算するための変数
    total_bet = 0  # 全体の賭け金（1レースごとに100円かける）
    total_return = 0  # 全体の回収金額
    payout_list = []  # 配当金のリスト（ヒストグラム用）

    # レースごとに予測順位を決定
    race_ids = data['レースID'].unique()
    
    
    for race_id in race_ids:
        race_data = data[data['レースID'] == race_id]
        race_data = race_data.sort_values('prob_0', ascending=False)

        # レースデータに少なくとも3艇が含まれていることを確認
        if len(race_data) < 3:
            print(f"レースID {race_id} のデータが不足しています。スキップします。")
            continue

        # 予測された1着、2着、3着を決定
        predicted_first = race_data.iloc[0]['艇番']
        predicted_second = race_data.iloc[1]['艇番']
        predicted_third = race_data.iloc[2]['艇番']

        

        # 実際の1着、2着、3着の選手（'着'カラムがそれぞれ1, 2, 3の選手を取得）
        actual_first = race_data[race_data['着'] == 1]['艇番'].values[0]
        actual_second = race_data[race_data['着'] == 2]['艇番'].values[0]
        actual_third = race_data[race_data['着'] == 3]['艇番'].values[0]

        # 賭け金を追加 (1レースあたり100円)
        total_bet += 100

        # #3連単の場合
        # # 正答率の計算（予測が的中したかを確認）
        # if (predicted_first == actual_first and
        #     predicted_second == actual_second and
        #     predicted_third == actual_third):
        #     correct_predictions += 1

        #     # 三連単オッズ情報の取得
        #     odds_row = odds_list[odds_list['レースID'] == race_id]

        #     # デバッグ: オッズ情報が存在するか確認
        #     if odds_row.empty:
        #         print(f"オッズ情報が見つかりません: レースID {race_id}")
        #         payout_list.append(0)
        #         continue

        #     # 予測が的中した場合、オッズに基づいて回収金額を計算
        #     winning_odds = odds_row["三連単オッズ"].values[0]

        #     # デバッグ: オッズが正常に取得されているか確認
        #     if pd.isna(winning_odds):
        #         print(f"オッズがNaNです: レースID {race_id}")
        #         payout_list.append(0)
        #         continue

        #     payout = 100 * winning_odds  # 当たった場合は100円×オッズ
        #     total_return += payout
        #     payout_list.append(payout)  # 配当金をリストに追加

        # else:
        #     payout_list.append(0)  # 外れた場合は0円

        # total_races += 1

        # # 三連単予測結果と実際の結果を表示
        # print(f"レースID: {race_id}")
        # print(f"予測 - 1着: {predicted_first}, 2着: {predicted_second}, 3着: {predicted_third}")
        # print(f"実際 - 1着: {actual_first}, 2着: {actual_second}, 3着: {actual_third}")


        # #複勝の場合
        # if predicted_first == actual_first:
        #     correct_predictions+=1
            
        #     #複勝オッズ情報の取得
        #     odds_row = odds_list[odds_list['レースID'] == race_id]
            
        #     # デバッグ: オッズ情報が存在するか確認
        #     if odds_row.empty:
        #         print(f"オッズ情報が見つかりません: レースID {race_id}")
        #         payout_list.append(0)
        #         continue
            
        #     #予測が的中した場合、オッズに基づいて回収金額を計算
        #     winning_odds1 = odds_row["複勝オッズ1"].values[0]

        #     #デバッグ:　オッズ情報が存在するか確認
        #     if pd.isna(winning_odds1):
        #         print(f"オッズがNaNです: レースID {race_id}")
        #         payout_list.append(100)
        #         continue

        #     payout = 100 * winning_odds1  # 当たった場合は100円×オッズ
        #     total_return += payout
        #     payout_list.append(payout)  # 配当金をリストに追加

        # elif predicted_first == actual_second:
        #     correct_predictions+=1
            
        #     #複勝オッズ情報の取得
        #     odds_row = odds_list[odds_list['レースID'] == race_id]
            
        #     # デバッグ: オッズ情報が存在するか確認
        #     if odds_row.empty:
        #         print(f"オッズ情報が見つかりません: レースID {race_id}")
        #         payout_list.append(0)
        #         continue
            
        #     #予測が的中した場合、オッズに基づいて回収金額を計算
        #     winning_odds2 = odds_row["複勝オッズ2"].values[0]

        #     #デバッグ:　オッズ情報が存在するか確認
        #     if pd.isna(winning_odds2):
        #         print(f"オッズがNaNです: レースID {race_id}")
        #         payout_list.append(100)
        #         continue

        #     payout = 100 * winning_odds2  # 当たった場合は100円×オッズ
        #     total_return += payout
        #     payout_list.append(payout)  # 配当金をリストに追加

        # else:
        #     payout_list.append(0)  # 外れた場合は0円

        # total_races += 1
        
        # # 複勝予測結果と実際の結果を表示
        # print(f"レースID: {race_id}")
        # print(f"予測 - 1着: {predicted_first}, 2着: {predicted_second}, 3着: {predicted_third}")
        # print(f"実際 - 1着: {actual_first}, 2着: {actual_second}, 3着: {actual_third}")

        #単勝の場合
        # 実際の1着の選手（'着'カラムが1の選手を取得）
        actual_first = race_data[race_data['着'] == 1]['艇番'].values[0]
        
        # 賭け金を追加 (1レースあたり100円)
        total_bet += 100

        # 正答率の計算
        if predicted_first == actual_first:
            correct_predictions += 1
            # オッズ情報の取得
            odds_row = odds_list[odds_list['レースID'] == race_id]
            # 予測が的中した場合、オッズに基づいて回収金額を計算
            winning_odds =  odds_row["単勝オッズ"].values[0]
            payout = 100 * winning_odds  # 当たった場合は100円×オッズ
            total_return += payout
            payout_list.append(payout)  # 配当金をリストに追加
        else:
            payout_list.append(0)  # 外れた場合は0円



    # 正答率を表示
    accuracy = correct_predictions / total_races if total_races > 0 else 0
    print(f"\n三連単の正答率: {accuracy * 100:.2f}%")

    # 的中率の計算


    # 回収率の計算
    if total_bet > 0:
        return_rate = (total_return / total_bet) * 100
    else:
        return_rate = 0
    
    # 結果の表示
    print(f"\n総賭け金: {total_bet}円")
    print(f"総回収金額: {total_return}円")
    print(f"回収率: {return_rate:.2f}%")

    # 配当金の統計情報
    payout_series = pd.Series(payout_list)
    print(f"\n配当金の統計情報:")
    print(f"最大値: {payout_series.max()}円")
    print(f"最小値: {payout_series.min()}円")
    print(f"平均: {payout_series.mean()}円")
    print(f"標準偏差: {payout_series.std()}円")

    # 配当金のヒストグラムをプロット
    plt.figure(figsize=(10, 6))
    sns.histplot(payout_series, bins=20, kde=False)
    plt.title("配当金のヒストグラム")
    plt.xlabel("配当金額 (円)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('odds_histogram.png')
    plt.show()
    print("ヒストグラムが 'odds_histogram.png' として保存されました。")

if __name__ == '__main__':
    inspect_model()
    # create_confidence_model()
    # create_confidence_model()

    # inspect_with_confidence_model()