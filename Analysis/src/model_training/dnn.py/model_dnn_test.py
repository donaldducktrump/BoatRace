import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

import pandas as pd
import numpy as np
import os

# 追加インポート
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from ...utils.mymodule import load_b_file, load_k_file, preprocess_data

def create_model():
    # 学習データ期間の設定
    month_folders = ['2403', '2404', '2405', '2406', '2407']
    date_files = {
        '2403': [f'2403{day:02d}' for day in range(1, 32)],
        '2404': [f'2404{day:02d}' for day in range(1, 31)],
        '2405': [f'2405{day:02d}' for day in range(1, 32)],
        '2406': [f'2406{day:02d}' for day in range(1, 31)],
        '2407': [f'2407{day:02d}' for day in range(1, 32)],
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

    # b_dataとk_dataの結合
    b_data = pd.concat(b_data_list, ignore_index=True)
    k_data = pd.concat(k_data_list, ignore_index=True)

    if b_data.empty or k_data.empty:
        print("データが正しく読み込めませんでした。")
        return

    # データのマージ
    data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
    if data.empty:
        print("データのマージ後に結果が空です。")
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
        print("前処理後のデータが空です。")
        return

    # 使用する特徴量の指定
    features = ['艇番', '年齢', '体重', '級別', '全国勝率', '全国2連率',
                '当地勝率', '当地2連率', 'モーターNo', 'モーター2連率',
                'ボートNo', 'ボート2連率']
    X = data[features]
    y = data['着']

    # 目的変数の作成（1着なら1、それ以外は0）
    y = y.apply(lambda x: 1 if int(x) == 1 else 0)

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = ['艇番', '級別', 'モーターNo', 'ボートNo']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_categorical = encoder.fit_transform(X[categorical_features])

    # エンコーダーを保存
    joblib.dump(encoder, 'encoder.pkl')

    # エンコーダーの出力する列名を取得
    categorical_feature_names = encoder.get_feature_names_out(categorical_features)

    # 数値データの抽出と標準化
    numeric_features = ['年齢', '体重', '全国勝率', '全国2連率',
                        '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率']
    X_numeric = X[numeric_features]
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)

    # スケーラーを保存
    joblib.dump(scaler, 'scaler.pkl')

    # 特徴量の結合
    X_processed = np.hstack([X_categorical, X_numeric_scaled])

    # 特徴量名の設定と保存
    feature_names = list(categorical_feature_names) + numeric_features
    joblib.dump(feature_names, 'feature_names.pkl')

    # 特徴量データフレームの作成（必要に応じて）
    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42)

    # モデルの構築
    model = keras.Sequential()
    model.add(layers.Dense(128, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001),
                           input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(0.49))
    model.add(layers.Dense(512, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.49))
    model.add(layers.Dense(512, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.49))
    model.add(layers.Dense(1, activation='sigmoid'))

    # モデルのコンパイル
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # コールバックの設定（早期停止）
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    # モデルの学習
    EPOCHS = 1000
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=512,
        callbacks=[early_stopping],
        verbose=1
    )

    # テストデータでの評価
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'テストデータでのAccuracy: {test_accuracy:.4f}')

    # モデルの保存
    model.save('boatrace_dnn_model.h5')
    print("モデルを 'boatrace_dnn_model.h5' として保存しました。")

    # 学習過程の可視化
    history_dict = history.history

    # 損失値のプロット
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, loss_values, 'bo', label='訓練データの損失')
    plt.plot(epochs, val_loss_values, 'b', label='検証データの損失')
    plt.title('訓練データと検証データの損失値')
    plt.xlabel('エポック数')
    plt.ylabel('損失値')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

    # 正解率のプロット
    plt.clf()
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    plt.plot(epochs, acc, 'bo', label='訓練データの正解率')
    plt.plot(epochs, val_acc, 'b', label='検証データの正解率')
    plt.title('訓練データと検証データの正解率')
    plt.xlabel('エポック数')
    plt.ylabel('正解率')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()

if __name__ == '__main__':
    create_model()
