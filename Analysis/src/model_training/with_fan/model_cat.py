import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, load_before_data, merge_before_data, load_before1min_data, preprocess_data1min
from get_data import load_fan_data, add_course_data

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'

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
    # 使用するデータ期間を指定
    month_folders = ['2311', '2312', '2401', '2402', '2403', '2404']
    date_files = {
        '2311': [f'2311{day:02d}' for day in range(1, 31)],
        '2312': [f'2312{day:02d}' for day in range(1, 32)],
        '2401': [f'2401{day:02d}' for day in range(1, 31)],
        '2402': [f'2402{day:02d}' for day in range(1, 29)],
        '2403': [f'2403{day:02d}' for day in range(1, 32)],
        '2404': [f'2404{day:02d}' for day in range(1, 31)],
    }

    # データのロードと前処理
    data_list = []
    for month in month_folders:
        for date in date_files[month]:
            try:
                b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'
                k_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_data/{month}/K{date}.TXT'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data2/{month}/beforeinfo_{date}.txt'

                b_data = load_b_file(b_file)
                k_data, _ = load_k_file(k_file)
                before_data = load_before_data(before_file)

                if not b_data.empty and not k_data.empty and not before_data.empty:
                    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
                    before_data = remove_common_columns(data, before_data, on_columns=['選手登番', 'レースID', '艇番'])
                    data = merge_before_data(data, before_data)
                    data_list.append(data)
                else:
                    print(f"データが不足しています: {date}")
            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file}, {k_file}, または {before_file}")

    if not data_list:
        print("データが正しく読み込めませんでした。")
        return
    
    data = pd.concat(data_list, ignore_index=True)
    fan_file = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\fan\fan2404.txt'
    df_fan = load_fan_data(fan_file)
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')
    data = preprocess_data(data)

    # 目的変数の設定
    # def target_label(x):
    #     x = int(x)
    #     if x in [1, 2, 3]:
    #         return 0
    #     else:
    #         return 1

    # def target_label(x):
    #     x = int(x)
    #     if x in [1, 2]:
    #         return 0
    #     else:
    #         return 1
        
    def target_label(x):
        x = int(x)
        if x == 1:
            return 0
        else:
            return 1

    data['target'] = data['着'].apply(target_label)

    # 特徴量と目的変数
    features = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                'ET', 'tilt', 'EST','ESC',
                'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h','性別','前期能力指数', '今期能力指数', '平均スタートタイミング', '勝率', '複勝率', '優勝回数', '優出回数']
    X = data[features]
    y = data['target']

    # カテゴリカル変数の指定
    categorical_features = ['会場', 'weather', 'wind_d', 'ESC', '性別', '級別', '支部', '艇番']
    for col in categorical_features:
        X[col] = X[col].astype(str)

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # CatBoostのモデル設定
    model = CatBoostClassifier(
        iterations=10000,
        learning_rate=0.05,
        depth=6,
        eval_metric='Accuracy',
        random_seed=42,
        early_stopping_rounds=10,
        verbose=100
    )

    # CatBoostでのトレーニング
    model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=categorical_features)

    # テストデータでの予測
    y_pred = model.predict(X_test)

    # 精度の計算
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy on test data: {accuracy:.4f}')

    # 特徴量の重要度を取得してプロット
    importance = model.get_feature_importance()
    feature_names = model.feature_names_
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    importance_df = importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig('feature_importance_catboost.png')
    plt.show()

    # モデルの保存
    model.save_model('boatrace_model_catboost.cbm')
    pd.Series(feature_names).to_csv('feature_names_catboost.csv', index=False)
    print("Feature names saved to 'feature_names_catboost.csv'.")

    # クロスバリデーション
    print("\nCross-validation with early stopping:")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[test_idx]

        model_cv = CatBoostClassifier(
            iterations=10000,
            learning_rate=0.05,
            depth=6,
            eval_metric='Accuracy',
            random_seed=42,
            early_stopping_rounds=10,
            verbose=False
        )
        
        model_cv.fit(X_train_cv, y_train_cv, eval_set=(X_valid_cv, y_valid_cv), cat_features=categorical_features)
        
        y_pred_cv = model_cv.predict(X_valid_cv)
        accuracy_cv = accuracy_score(y_valid_cv, y_pred_cv)
        scores.append(accuracy_cv)
        print(f'Fold {i+1} Accuracy: {accuracy_cv:.4f}')

    print(f'\nMean Accuracy: {np.mean(scores):.4f}')

if __name__ == '__main__':
    create_model()
