# investment_system.py
from bs4 import BeautifulSoup
import pickle 
import joblib
import time
from datetime import datetime, timedelta, timezone
# Selenium関連のインポートは後で実装
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
from apscheduler.schedulers.background import BackgroundScheduler
# inspection.py
from save_test4 import get_url, requests_session, get_beforeinfo, save_beforeinfo_data
import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data,preprocess_data3,preprocess_data1min, load_before_data,load_before1min_data, merge_before_data, load_dataframe
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import traceback
import os
from datetime import datetime, timedelta
import pytz
import re
import threading
import logging
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.alert import Alert

# from msedge.selenium_tools import Edge, EdgeOptions
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from os.path import join

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1302815087446196255/gCxgLMe33h-LC9ojW9nzkCIQFvcf4urPnNQ3PE8BNYI08R4SUn1222RwlX1OyWwUY4jZ"

def send_discord_message(message, webhook_url):
    import requests
    data = {
        "content": message,
        "username": "BoatRaceBot"  # 任意のユーザー名
    }
    response = requests.post(webhook_url, data=data)
    if response.status_code != 204:
        print(f"Failed to send message to Discord: {response.status_code}")

# 日本語対応フォントを指定
plt.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

# 1. 場所コードと名称のマッピング
place_code_mapping = {
    '01': '桐生', '02': '戸田', '03': '江戸川', '04': '平和島', '05': '多摩川',
    '06': '浜名湖', '07': '蒲郡', '08': '常滑', '09': '津', '10': '三国',
    '11': 'びわこ', '12': '住之江', '13': '尼崎', '14': '鳴門', '15': '丸亀',
    '16': '児島', '17': '宮島', '18': '徳山', '19': '下関', '20': '若松',
    '21': '芦屋', '22': '福岡', '23': '唐津', '24': '大村'
}

# マッピングを逆転させる（場所名称からコードを取得）
place_name_to_cd = {v: k for k, v in place_code_mapping.items()}
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

def remove_common_columns(df_left, df_right, on_columns):
    """
    df_leftとdf_rightのマージ時に、on_columnsを除く共通の列をdf_rightから削除する。
    """
    common_cols = set(df_left.columns).intersection(set(df_right.columns)) - set(on_columns)
    if common_cols:
        print(f"共通列: {common_cols}。df_rightからこれらの列を削除します。")
        df_right = df_right.drop(columns=common_cols)
    return df_right

def construct_race_id(date_str, place_cd, race_no):
    """
    race_idを構築する。
    フォーマット: 'YYYYMMDDPPRR'
    """
    return f"{date_str}{place_cd}{race_no:02d}"

# 4. セッションとリトライ戦略の設定
def requests_session():
    """
    リトライ戦略を備えたrequestsセッションを作成する。
    
    Returns:
        requests.Session: 設定されたセッションオブジェクト。
    """
    session = requests.Session()
    retry = Retry(
        total=5,  # 最大リトライ回数
        backoff_factor=0.3,  # リトライ間隔の増加率
        status_forcelist=(500, 502, 504),  # リトライ対象のステータスコード
        allowed_methods=["GET", "POST"]  # リトライ対象のメソッド
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

# 5. 処理済みのrace_idをファイルから読み込む
def load_processed_race_ids(file_path='processed_race_ids.json'):
    """処理済みのrace_idをファイルから読み込む"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                print(f"[{get_current_time_jst().strftime('%Y-%m-%d %H:%M:%S')}] JSONデコードエラー: {file_path} を初期化します。")
                return set()
    return set()

# 6. 処理済みのrace_idをファイルに保存する
def save_processed_race_ids(processed_race_ids, file_path='processed_race_ids3_trifecta.json'):
    """処理済みのrace_idをファイルに保存する"""

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(list(processed_race_ids), f, ensure_ascii=False, indent=4)

# def fetch_race_ids_before_deadline(session):
#     """
#     指定されたURLからレース情報を取得し、
#     現在時刻の1分後に開始するレースのrace_idを返す。
    
#     Returns:
#         list: 該当するrace_idのリスト
#     """

#     url = 'https://www.boatrace.jp/owpc/pc/race/index'
#     headers = {'User-Agent': 'Mozilla/5.0'}

#     try:
#         response = session.get(url, headers=headers, timeout=10)
#     except requests.RequestException as e:
#         print(f"[{get_current_time_jst().strftime('%Y-%m-%d %H:%M:%S')}] ページの取得中にエラーが発生しました: {e}")
#         return []
    
#     if response.status_code != 200:
#         print(f"ページの取得に失敗しました。ステータスコード: {response.status_code}")
#         return []
    
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     race_ids = []
    
#     # 現在時刻 + 1分 を取得
#     current_time = get_current_time_jst()
#     target_time = current_time + timedelta(minutes=1)
#     target_time_str = target_time.strftime('%H:%M')
    
#     print(f"現在時刻 (JST): {current_time.strftime('%H:%M')}, ターゲット時刻: {target_time_str}")
    
#     # 全ての<tbody>タグを取得
#     for tbody in soup.find_all('tbody'):
#         trs = tbody.find_all('tr')
#         if len(trs) < 2:
#             continue  # レース情報が不完全な場合はスキップ
        
#         # 1つ目の<tr>: 場所名、レース番号、日付を取得
#         first_tr = trs[0]
        
#         # <img>タグから場所名を取得
#         img_tag = first_tr.find('img', alt=True)
#         if not img_tag:
#             continue
#         place_name = img_tag['alt']
#         if place_name not in place_name_to_cd:
#             print(f"不明な場所名: {place_name}")
#             continue
#         place_cd = place_name_to_cd[place_name]
        
#         # レース番号を取得 (例: '10R' -> 10)
#         race_no_td = first_tr.find('td', text=re.compile(r'\d+R'))
#         if not race_no_td:
#             print("レース番号が見つかりません")
#             continue
#         race_no_text = race_no_td.get_text(strip=True)
#         match = re.match(r'(\d+)R', race_no_text)
#         if not match:
#             print(f"レース番号の形式が不正です: {race_no_text}")
#             continue
#         race_no = int(match.group(1))
        
#         # # レースの日付を取得 (hrefのパラメータから 'hd=YYYYMMDD')
#         # race_link = first_tr.find('a', href=True)
#         # if not race_link:
#         #     print("レースリンクが見つかりません")
#         #     continue
#         # href = race_link['href']
#         # match = re.search(r'hd=(\d{8})', href)
#         # if not match:
#         #     print(f"日付がリンクから取得できませんでした: {href}")
#         #     continue
#         # date_str = match.group(1)  # 'YYYYMMDD'
        
#         # 2つ目の<tr>: レース開始時刻を取得
#         second_tr = trs[1]
#         time_td = second_tr.find('td', text=re.compile(r'\d{1,2}:\d{2}'))
#         if not time_td:
#             print("レース時刻が見つかりません")
#             continue
#         race_time = time_td.get_text(strip=True)  # 例: '15:38'
        
#         # ターゲット時刻と比較
#         if race_time == target_time_str:
#             date_str = current_time.strftime('%Y%m%d')  # 'YYYYMMDD'
#             race_id = construct_race_id(date_str, place_cd, race_no)
#             race_ids.append(race_id)
#             print(f"該当レースIDを検出: {race_id} (時刻: {race_time})")
    
#     return race_ids

# 7. 投票締め切り時刻とrace_idを取得する

def fetch_race_deadlines(session):
    """
    指定されたURLからレース情報を取得し、
    各レースの投票締め切り時刻とrace_idを返す。
    
    Args:
        session (requests.Session): 設定されたセッションオブジェクト。
    
    Returns:
        list of dict: 該当するレースの情報 [{'race_id': ..., 'deadline_time': ...}, ...]
    """
    url = 'https://www.boatrace.jp/owpc/pc/race/index'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = session.get(url, headers=headers, timeout=10)
    except requests.RequestException as e:
        print(f"[{get_current_time_jst().strftime('%Y-%m-%d %H:%M:%S')}] ページの取得中にエラーが発生しました: {e}")
        return []
    
    if response.status_code != 200:
        print(f"[{get_current_time_jst().strftime('%Y-%m-%d %H:%M:%S')}] ページの取得に失敗しました。ステータスコード: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    races = []
    
    
    # 現在時刻を取得
    current_time = get_current_time_jst()
    today_str = current_time.strftime('%Y%m%d')  # 'YYYYMMDD'
    table_num=-1
    # 全ての<tbody>タグを取得
    for tbody in soup.find_all('tbody'):
        table_num+=1
        trs = tbody.find_all('tr')
        if len(trs) < 2:
            continue  # レース情報が不完全な場合はスキップ
        
        # 1つ目の<tr>: 場所名、レース番号を取得
        first_tr = trs[0]
        # <img>タグから場所名を取得
        img_tag = first_tr.find('img', alt=True)
        if not img_tag:
            continue
        place_name = img_tag['alt']
        if place_name not in place_name_to_cd:
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] 不明な場所名: {place_name}")
            continue
        place_cd = place_name_to_cd[place_name]
        
        # レース番号を取得 (例: '10R' -> 10)
        race_no_td = first_tr.find('td', string=re.compile(r'\d+R'))
        if not race_no_td:
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] レース番号が見つかりません")
            continue
        race_no_text = race_no_td.get_text(strip=True)
        match = re.match(r'(\d+)R', race_no_text)
        if not match:
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] レース番号の形式が不正です: {race_no_text}")
            continue
        race_no = int(match.group(1))
        
        # レース開始時刻を取得 (2つ目の<tr>の<td>タグから)
        second_tr = trs[1]
        time_td = second_tr.find('td', string=re.compile(r'\d{1,2}:\d{2}'))
        if not time_td:
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] レース時刻が見つかりません")
            continue
        race_time_str = time_td.get_text(strip=True)  # 例: '15:38'
        try:
            race_time = datetime.strptime(f"{today_str} {race_time_str}", '%Y%m%d %H:%M')
            race_time = pytz.timezone('Asia/Tokyo').localize(race_time)
        except ValueError as e:
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] レース時刻の解析に失敗しました: {e}")
            continue
        
        # 投票締め切り時刻を計算（ここでは仮にレース開始時刻の1分前とします）
        # 実際の締め切り時刻の計算方法に応じて変更してください
        deadline_time = race_time
        
        # race_idを構築
        race_id = construct_race_id(today_str, place_cd, race_no)
        
        races.append({
            'race_id': race_id,
            'deadline_time': deadline_time,
            'table_num': table_num # <tbody>タグのインデックス
        })
        
        print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] レースID: {race_id}, 投票締め切り時刻: {deadline_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return races

def get_current_time_jst():
    """現在の日本標準時を取得"""
    jst = pytz.timezone('Asia/Tokyo')
    return datetime.now(jst)

# ログを記録する関数（日付ごとにファイルを作成）
def log_vote_result(message):
    # 現在の日付を取得してファイル名を生成（例: vote_results_2024-10-18.log）
    current_date = datetime.now().strftime('%Y-%m-%d')
    log_file = f"vote_results_{current_date}.log"
    
    # ログファイルに追記モードで書き込み
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
# 1レース分のtrifecta_dataをテキストファイルに追加保存

def save_trifecta_odds(trifecta_data, date, place_cd, race_no):
    # フォルダとファイル名の作成
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')  # 例: '2023-11-01' 形式の文字列を変換
    month = date.strftime('%y%m')
    date_str = date.strftime('%y%m%d')
    folder_path = os.path.join(r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\trifecta_odds1min', month)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'trifecta1min_{date_str}.txt')
    
    # ファイルに追記
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"Place: {place_cd}, Race: {race_no}\n")
        for entry in trifecta_data:
            line = f"Boat1: {entry['Boat1']}, Boat2: {entry['Boat2']}, Boat3: {entry['Boat3']}, Odds: {entry['Odds']}\n"
            f.write(line)
        f.write("\n")  # 区切り行を追加

    print(f"Saved trifecta odds data to {file_path}")

def inspect_model(race_id,table_num,deadline_time):

    logging.info(f"inspect_modelを呼び出しました。race_id: {race_id}")

    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_first.txt')
        gbm_odds = lgb.Booster(model_file='boatrace_odds_model_inv.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_first.csv').squeeze().tolist()
    feature_names_odds = pd.read_csv('feature_names_odds_inv.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2411']
    date_files = {
        # '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        # '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        # '2410': [f'2410{day:02d}' for day in range(31, 32)],   # 241001 - 241008
        '2411': [f'2411{day:02d}' for day in range(5,6)],  # 241101 - 241130
    }

    # データを結合
    b_data_list = []
    k_data_list = []
    odds_list1 = []
    data_list = []

    for month in month_folders:
        for date in date_files[month]:
            try:
                # b_df_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_dataframe/{month}/B{date}.pkl'
                b_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/b_data/{month}/B{date}.TXT'

                # k_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_data/{month}/K{date}.TXT'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data_1min/{month}/beforeinfo1min_{date}.txt'
                b_data = load_b_file(b_file)

                # b_data = load_dataframe(b_df_file)
                # k_data, odds_list_part = load_k_file(k_file)
                before_data, trifecta_odds=fetch_race_info_by_id(race_id)
                # print(before_data)
                save_race_data(before_data)
                date = pd.to_datetime(race_id[:8], format='%Y%m%d').strftime('%Y-%m-%d')
                place_cd = int(race_id[8:10])  # 例: '202410140404' -> place_cd=04
                race_no = int(race_id[10:12])  # 例: '202410140404' -> race_no=04
                # save_trifecta_odds(trifecta_odds, date, place_cd,race_no)
                before_data=process_race_series(before_data)

                if not b_data.empty and not before_data.empty:
                    # データのマージ
                    # data = pd.merge(b_data, k_data, on=['選手登番', 'レースID'])
                    data = b_data
                    data = merge_before_data(data, before_data)
                    data_list.append(data)
                    # if not odds_list_part.empty:
                    #     odds_list1.append(odds_list_part)
                else:
                    print(f"データが不足しています: {date}")

            except FileNotFoundError:
                print(f"ファイルが見つかりません: {b_file},または {before_file}")
    if not data_list:
        print("データが正しく読み込めませんでした。")
        message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] raceid:{race_id} のデータが正しく読み込めませんでした。"
        log_vote_result(message)
        return
    data_odds = load_processed_odds_data()
    print(data_odds[["win_odds_mean","選手登番","艇番","win_odds"]])

    data = pd.concat(data_list, ignore_index=True)

    df_fan = load_fan_data_from_csv()
    columns_to_add = ['前期能力指数', '今期能力指数', '平均スタートタイミング', '性別', '勝率', '複勝率', '優勝回数', '優出回数']
    data['選手登番'] = data['選手登番'].astype(int)
    df_fan['選手登番'] = df_fan['選手登番'].astype(int)
    data = pd.merge(data, df_fan[['選手登番'] + columns_to_add], on='選手登番', how='left')
    print(data.columns)

    data = preprocess_data(data)

    # 特徴量の指定
    features = [
        '会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
        '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
        'ET', 'tilt', 'EST', 'ESC',
        'weather', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
        '前期能力指数', '今期能力指数', '平均スタートタイミング', '性別',
        '勝率', '複勝率', '優勝回数', '優出回数'
    ]
    X = data[features]

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features = [
        '会場', 'weather', 'wind_d', 'ESC', '艇番', '性別', '支部', '級別'
    ]
    X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

    # 数値データの選択
    numeric_features = [col for col in X.columns if col not in categorical_features]
    X_numeric = X[numeric_features]

    # 特徴量の結合（インデックスをリセットしない）
    X_processed = pd.concat([X_numeric, X_categorical], axis=1)

    # 天候_雪 カラムが存在しない場合に追加する
    for col in feature_names:
        if col not in X_processed.columns:
            X_processed[col] = 0

    # 学習時の特徴量と順序を合わせる
    X_processed = X_processed[feature_names]

    # 予測
    y_pred = gbm.predict(X_processed)

    data['prob_0'] = y_pred[:, 0]  # 1着の確率

    # 'prob_0' の分布をプロット
    # plt.figure(figsize=(8, 6))
    # plt.hist(data['prob_0'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    # plt.title("予測確率 'prob_0' の分布")
    # plt.xlabel("予測確率 (prob_0)")
    # plt.ylabel("度数")
    # plt.grid(True)
    # plt.savefig('prob_0_distribution.png')
    # # plt.show()
    # print("プロットが 'prob_0_distribution.png' として保存されました。")

    # 'win_odds_mean' を計算してマージ
    data_odds_grouped = data_odds.groupby(['選手登番', '艇番'], as_index=False)['win_odds_mean'].mean()
    data['選手登番'] = data['選手登番'].astype(int)
    data_odds_grouped['選手登番'] = data_odds_grouped['選手登番'].astype(int)
    data = data.merge(data_odds_grouped, on=['選手登番', '艇番'], how='left')

    # 必要な特徴量を選択
    features_odds = ['会場', '艇番', '級別', '支部', '年齢', '体重', '全国勝率', '全国2連率',
                    '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率',
                    'ET', 'tilt', 'EST','ESC',
                    'weather','air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h',
                    'win_odds1min', 'prob_0','place_odds1min','win_odds_mean', '性別', '勝率', '複勝率', '優勝回数', '優出回数','前期能力指数', '今期能力指数','平均スタートタイミング']
    
    data.rename(columns={'win_odds':'win_odds1min','place_odds':'place_odds1min'},inplace=True)
    data = data.dropna()
    data['win_odds1min']=np.where(data['win_odds1min']==0, data['win_odds_mean'], data['win_odds1min'])
    data['place_odds1min']=np.where(data['place_odds1min']==0, data['win_odds_mean'], data['place_odds1min'])



    X_odds = data[features_odds].copy()
    # 'win_odds1min', 'place_odds1min', 'win_odds_mean' 列を数値型に変換
    data['win_odds1min'] = pd.to_numeric(data['win_odds1min'], errors='coerce')
    data['place_odds1min'] = pd.to_numeric(data['place_odds1min'], errors='coerce')
    data['win_odds_mean'] = pd.to_numeric(data['win_odds_mean'], errors='coerce')


    # 欠損値やゼロを処理
    data['win_odds1min'] = np.where(data['win_odds1min']==0, data['win_odds_mean'], data['win_odds1min'])
    data['place_odds1min'] = np.where(data['place_odds1min']==0, data['win_odds_mean'], data['place_odds1min'])
    data['win_odds1min'] = 0.725 / data['win_odds1min']
    data['place_odds1min'] = 0.725 / data['place_odds1min']
    data['win_odds_mean'] = 0.725 / data['win_odds_mean']

    X_odds['win_odds1min'] = data['win_odds1min']
    X_odds['place_odds1min'] = data['place_odds1min']
    X_odds['win_odds_mean'] = data['win_odds_mean']

    # カテゴリカル変数のOne-Hotエンコーディング
    categorical_features_odds = ['会場', 'weather','wind_d', 'ESC', '艇番', '性別', '支部', '級別']
    X_categorical_odds = pd.get_dummies(X_odds[categorical_features_odds], drop_first=True)

    # 数値データの選択
    numeric_features_odds = [col for col in X_odds.columns if col not in categorical_features_odds]
    X_numeric_odds = X_odds[numeric_features_odds]

    # 特徴量の結合
    X_processed_odds = pd.concat([X_numeric_odds.reset_index(drop=True), X_categorical_odds.reset_index(drop=True)], axis=1)

    # 特徴量の順序を合わせる
    for col in feature_names_odds:
        if col not in X_processed_odds.columns:
            X_processed_odds[col] = 0
    X_processed_odds = X_processed_odds[feature_names_odds]

    # オッズの予測
    odds_inv_pred = gbm_odds.predict(X_processed_odds)
    data['odds_inv_pred'] = odds_inv_pred
    # data['odds_pred'] = 0.725 / data['odds_inv_pred']

    
    total_capital = 100000  # 総資金（円）
    unit = 500
    prob_threshold = 0

    # 三連単オッズデータの読み込み
    trifecta_odds_data_df = load_trifecta_odds(date, place_cd, race_no)

    # 各艇の確率データ
    data_boats = data[['レースID', '艇番', 'odds_inv_pred', '会場']].copy()

    trifecta_bet = process_race(race_id,trifecta_odds_data_df,data_boats,total_capital,unit,prob_threshold)
    
    # 日付を取得（race_idから）
    date_str = race_id[:8]  # 'YYYYMMDD'
    
    # フォルダのパスを構築
    base_dir = 'vote'
    year_month = date_str[:6]  # 'YYYYMM'
    target_dir = os.path.join(base_dir, year_month)
    
    # フォルダが存在しない場合は作成
    os.makedirs(target_dir, exist_ok=True)
    
    # ファイル名を構築
    file_name = f"vote_{date_str}.csv"
    file_path = os.path.join(target_dir, file_name)
    
    try:
        # データをCSVファイルに保存（既存のファイルがあればアペンド）
        if not os.path.isfile(file_path):
            # ヘッダー付きで保存
            trifecta_bet.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] データを新規保存しました: {file_path}")
        else:
            # ヘッダーなしでアペンド
            trifecta_bet.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] データをアペンドしました: {file_path}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] データの保存中にエラーが発生しました: {e}")
    
    # '艇番'と'bet_amount'のみを抽出
    betting_data = trifecta_bet[['Boat1','Boat2','Boat3', 'bet_amount']]


    # # Chromeのオプション設定
    # chrome_options = Options()
    # # chrome_options.add_argument("--headless")  # ヘッドレスモード
    # # chrome_options.add_argument("--disable-gpu")
    # # chrome_options.add_argument("--no-sandbox")
    
    # # ChromeDriverのパスを指定
    # # service = Service(executable_path='path_to_chromedriver')  # ChromeDriverのパスに置き換えてください
    
    # # WebDriverの初期化
    # driver = webdriver.Chrome(options=chrome_options)
    # wait = WebDriverWait(driver, 2)
    # total = 0
    # try:
       
       
    # # レースページにアクセ
    #    driver.get('https://www.boatrace.jp/owpc/pc/race/index')
    #    time.sleep(2)  # ページがロードされるまで待機
        
    # #     # race_idに基づいて投票ボタンを探す
    # #     # 例として、特定のrace_idとboat_numberに対応するボタンのIDを構築
    # #     # 実際のIDの構造に合わせて調整してください
    #    vote_button_id = f"TENTP090A{table_num}"
    # #     # 投票ボタンをクリック
    #    vote_button = driver.find_element(By.ID, vote_button_id)
    #    vote_button.click()
    #    time.sleep(2)  # ポップアップが開くまで待機

    # #     # ログイン
    #    login_id = driver.find_element(By.ID, 'in_KanyusyaNo')
    #    login_id.send_keys('06783656')
    #    login_pw = driver.find_element(By.ID, 'in_AnsyoNo')
    #    login_pw.send_keys('3316')
    #    verification_pass = driver.find_element(By.ID, 'in_PassWord')
    #    verification_pass.send_keys('QnbMT7')
    #    login_button = driver.find_element(By.XPATH, '//*[@id="TENT010_TENTPC010PRForm"]/p/button')
    #    login_button.click()
    #    time.sleep(2)
        
    #    alert = driver.switch_to.window(driver.window_handles[1])
    #    alert.accept()

    # #    time.sleep(2)

    # #    # 単勝ボタンをクリック
    #    vote_button = driver.find_element(By.XPATH, '//*[@id="betkati1"]/a')
    #    vote_button.click()
    #    time.sleep(2)  # ポップアップが開くまで


    # #     # 投票を実行
    #    for index, row in betting_data.iterrows():
           
    #        boat_number = row['艇番']
    #        bet_amount = row['bet_amount']

    #        select_boat = driver.find_element(By.XPATH, f'//*[@id="regbtn_{boat_number}_1"]/a')
    #        select_boat.click()

    #        # 賭け金額を入力
    #        bet_input = driver.find_element(By.XPATH, '//*[@id="amount"]')  # 賭け金額入力フィールドのIDに置き換えてください
    #        bet_input.clear()
    #        bet_input.send_keys(str(bet_amount/100))

    #        # ベットリストに追加
    #        add_button = driver.find_element(By.XPATH, '//*[@id="regAmountBtn"]/a')
    #        add_button.click()
    #        total += bet_amount

    # #     # 投票確定ボタンをクリック
    #        confirm_button = driver.find_element(By.XPATH, '//*[@id="betList"]/div[3]/div[3]/a')  # 確定ボタンのIDに置き換えてください
    #        confirm_button.click()
    #        time.sleep(2)  # 投票が完了するまで待機

    # #     # 購入金額を入力
    #        purchase_input = driver.find_element(By.ID, 'amount')  # 購入金額入力フィールドのIDに置き換えてください
    #        purchase_input.clear()
    #        purchase_input.send_keys(str(total))

    # #     # パスワードを入力 
    #        password_input = driver.find_element(By.ID, 'pass')  # パスワード入力フィールドのIDに置き換えてください
    #        password_input.clear()
    #        password_input.send_keys('kazu01')  # パスワードを入力してください

    # #     # 購入確定ボタンをクリック
    #        purchase_button = driver.find_element(By.XPATH, '//*[@id="submitBet"]/a')  # 購入確定ボタンのIDに置き換えてください
    #        purchase_button.click()
    #        time.sleep(2)  # 購入が完了するまで待機

    #        print(f"[{get_current_time_jst().strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id}, 舟番: {boat_number} に {bet_amount} 円を投票しました。")
    
    # except Exception as e:
    #      print(f"[{get_current_time_jst().strftime('%Y-%m-%d %H:%M:%S')}] 投票中にエラーが発生しました: {e}")
    
    # finally:
    #      driver.quit()

    # bet_amounts_per_boat = {i: 0 for i in range(1, 7)}
    # 投票締め切り時刻を過ぎていたら投票しない
    now = datetime.now(timezone.utc)  # 例として UTC タイムゾーンを使用
    if now > deadline_time:
        for _, row in betting_data.iterrows():
            boat1, boat2, boat3 = row['Boat1'], row['Boat2'], row['Boat3']
            bet_amount = row['bet_amount']
            if bet_amount > 0:
                message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id}, 組み合わせ: {boat1}-{boat2}-{boat3} に {bet_amount} 円を投票しようとしました。"
                log_vote_result(message)
                print(message)
                send_discord_message(message, DISCORD_WEBHOOK_URL)
        print("投票締め切り時刻を過ぎているため、投票を行いません。")
        return
    
    # 投票を実行（締め切り時刻前）
    for _, row in betting_data.iterrows():
        boat1, boat2, boat3 = row['Boat1'], row['Boat2'], row['Boat3']
        bet_amount = row['bet_amount']
        if bet_amount > 0:
            # 投票メッセージを作成し、ログに保存
            message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id}, 組み合わせ: {boat1}-{boat2}-{boat3} に {bet_amount} 円を投票する予定です。"
            # log_vote_result(message)
            print(message)


    date_str = pd.to_datetime(race_id[:8], format='%Y%m%d').strftime('%Y-%m-%d')
    place_cd = int(race_id[8:10])  # 例: '202410140404' -> place_cd=04
    race_no = int(race_id[10:12])  # 例: '202410140404' -> race_no=04
    # args = (date_str, place_cd, race_no)
    url_before = get_url(date, place_cd, race_no, 'beforeinfo')

    # Edgeのオプション設定
    path='msedgedriver_129.exe'
    options = Options()
    options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")  # リモートデバッグのアドレスを指定
    # options.page_load_strategy = 'eager'  # ページが読み込まれる前に処理を続ける
    options.use_chromium = True
    # options.add_argument("headless")  # ヘッドレスモード
    service = Service(executable_path=path)

    # 既存のEdgeブラウザに接続
    driver = webdriver.Edge(service=service, options=options)
  
    # WebDriverの初期化
    # driver = webdriver.Edge(options=edge_options)
    wait = WebDriverWait(driver, 10)
    total = 0
    try:
        # 新しいウィンドウが開いたか確認して処理
        original_window = driver.current_window_handle
        
        # レースページにアクセス
        driver.get('https://www.boatrace.jp/owpc/pc/race/index')
        time.sleep(1)  # ページがロードされるまで待機

        # race_idに基づいて投票ボタンを探す
        vote_button_id = f"TENTP090A{table_num}"
        vote_button = driver.find_element(By.ID, vote_button_id)
        vote_button.click()
        time.sleep(1)  # ポップアップが開くまで待機

        # wait.until(EC.number_of_windows_to_be(2))  # ウィンドウが2つになるまで待機

        # print(url_before)
        # driver.get(url_before)
        # vote_button = driver.find_element(By.XPATH, '//*[@id="commonHead"]')
        # vote_button.click()
        # time.sleep(0.5)

        # # ログイン
        # login_id = driver.find_element(By.ID, 'in_KanyusyaNo')
        # login_id.send_keys('06783656')
        # login_pw = driver.find_element(By.ID, 'in_AnsyoNo')
        # login_pw.send_keys('3316')
        # verification_pass = driver.find_element(By.ID, 'in_PassWord')
        # verification_pass.send_keys('QnbMT7')
        # login_button = driver.find_element(By.XPATH, '//*[@id="TENT010_TENTPC010PRForm"]/p/button')
        # login_button.click()
        # time.sleep(2)  # ログイン後の処理

        # 新しいウィンドウに切り替える
        for window_handle in driver.window_handles:
            if window_handle != original_window:
                driver.switch_to.window(window_handle)
                print(f"新しいウィンドウに切り替えました: {window_handle}")
                break
        # time.sleep(1)
        
        # もしiframe内であればiframeに切り替える
        # try:
        #     iframe = driver.find_element(By.TAG_NAME, "iframe")
        #     driver.switch_to.frame(iframe)  # iframeに切り替え
        # except Exception:
        #     print("iframeが見つかりませんでした。")

        # 単勝ボタンが表示されるのを待機
        # vote_button = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="betkati1"]/a')))
        # time.sleep(2)  # ポップアップが開くまで待機

        # 単勝ボタンをクリック
        # vote_button = driver.find_element(By.XPATH, '//*[@id="betkati1"]/a')
        # vote_button.click()
        # time.sleep(2)  

        # 投票を実行
        for index, row in betting_data.iterrows():
            boat1 = row['Boat1']
            boat2 = row['Boat2']
            boat3 = row['Boat3']
            bet_amount = row['bet_amount']

            if bet_amount <= 0:
                continue

            select_boat = driver.find_element(By.XPATH, f'//*[@id="regbtn_{boat1}_1"]/a')
            select_boat.click()
            # time.sleep(0.2)

            select_boat2 = driver.find_element(By.XPATH, f'//*[@id="regbtn_{boat2}_2"]/a')
            select_boat2.click()
            # time.sleep(0.2)

            select_boat3 = driver.find_element(By.XPATH, f'//*[@id="regbtn_{boat3}_3"]/a')
            select_boat3.click()
            # time.sleep(0.2)

            # 賭け金額を入力
            bet_input = driver.find_element(By.XPATH, '//*[@id="amount"]')
            bet_input.clear()
            bet_input.send_keys(str(bet_amount / 1000))

            # ベットリストに追加
            add_button = driver.find_element(By.XPATH, '//*[@id="regAmountBtn"]/a')
            add_button.click()
            total += bet_amount
        
        if total == 0:
            message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id} 投票する金額がありません。"
            log_vote_result(message)
            print(message)
            send_discord_message(message, DISCORD_WEBHOOK_URL)
            driver.close()
            return
        
        # # delete_button = driver.find_element(By.XPATH,'//*[@id="betList"]/div[1]/div/a[3]')
        # if place_cd == 6 or place_cd == 7 or place_cd == 8 or place_cd == 9 or place_cd == 11 or place_cd == 22 or place_cd == 23:
        #     # if place_cd == 5:
        #     #     data_boats['bet_amount'] = 0
        #     #     print(f"会場が多摩川です")
        #     #     # delete_button.click()
        #     #     # confirm_delete = driver.find_element(By.XPATH, '//*[@id="ok"]')
        #     #     # confirm_delete.click()
        #     #     # driver.close()
        #     #     # return
        #     if place_cd == 6:
        #         data_boats['bet_amount'] = 0
        #         print(f"会場が浜名湖です")
        #         # delete_button.click()
        #         # confirm_delete = driver.find_element(By.XPATH, '//*[@id="ok"]')
        #         # confirm_delete.click()
        #         # driver.close()
        #         # return
        #     if place_cd == 7:
        #         data_boats['bet_amount'] = 0
        #         print(f"会場が蒲郡です")
        #         # delete_button.click()
        #         # confirm_delete = driver.find_element(By.XPATH, '//*[@id="ok"]')
        #         # confirm_delete.click()
        #         # driver.close()
        #         # return
        #     if place_cd == 8:
        #         data_boats['bet_amount'] = 0
        #         print(f"会場が常滑です")
        #         # delete_button.click()
        #         # confirm_delete = driver.find_element(By.XPATH, '//*[@id="ok"]')
        #         # confirm_delete.click()
        #         # return
        #     if place_cd == 9:
        #         data_boats['bet_amount'] = 0
        #         print(f"会場が津です")
        #         # delete_button.click()
        #         # confirm_delete = driver.find_element(By.XPATH, '//*[@id="ok"]')
        #         # confirm_delete.click()
        #         # return
        #     if place_cd == 11:
        #         data_boats['bet_amount'] = 0
        #         print(f"会場がびわこです")
        #         # delete_button.click()
        #         # confirm_delete = driver.find_element(By.XPATH, '//*[@id="ok"]')
        #         # confirm_delete.click()
        #         # return
        #     # if place_cd == 12:
        #     #     data_boats['bet_amount'] = 0
        #     #     print(f"会場が住之江です")
        #     #     # delete_button.click()
        #     #     # confirm_delete = driver.find_element(By.XPATH, '//*[@id="ok"]')
        #     #     # confirm_delete.click()
        #     #     # return
        #     # if place_cd == 18:
        #     #     data_boats['bet_amount'] = 0 # 18場のレースは賭けない
        #     #     print(f"会場が徳山です")
        #     #     # delete_button.click()
        #     #     # confirm_delete = driver.find_element(By.XPATH, '//*[@id="ok"]')
        #     #     # confirm_delete.click()
        #     #     # return
        #     if place_cd == 22:
        #         data_boats['bet_amount'] = 0
        #         print(f"会場が福岡です")
        #         # delete_button.click()
        #         # confirm_delete = driver.find_element(By.XPATH, '//*[@id="ok"]')
        #         # confirm_delete.click()
        #         # return
        #     if place_cd == 23:
        #         data_boats['bet_amount'] = 0
        #         print(f"会場が唐津です")
        #         # delete_button.click()
        #         # confirm_delete = driver.find_element(By.XPATH, '//*[@id="ok"]')
        #         # confirm_delete.click()
                
        #     # delete_button.click()
        #     # confirm_delete = driver.find_element(By.XPATH, '//*[@id="ok"]')
        #     # confirm_delete.click()
        #     driver.close()
        #     return
        
        # 投票確定ボタンをクリック
        confirm_button = driver.find_element(By.XPATH, '//*[@id="betList"]/div[3]/div[3]/a')
        confirm_button.click()
        time.sleep(1)  # 投票が完了するまで待機

        # 購入金額を入力
        purchase_input = driver.find_element(By.ID, 'amount')
        purchase_input.clear()
        purchase_input.send_keys(str(total))

        # パスワードを入力 
        password_input = driver.find_element(By.ID, 'pass')
        password_input.clear()
        password_input.send_keys('kazu01')  # パスワードを入力してください

        # 購入確定ボタンをクリック
        purchase_button = driver.find_element(By.XPATH, '//*[@id="submitBet"]/a')
        purchase_button.click()

        # 確認ボタンをクリック
        confirm_button = driver.find_element(By.XPATH, '//*[@id="ok"]')
        confirm_button.click()

        time.sleep(2)  # 購入が完了するまで待機
        
        # 投票結果をログに記録
        for index, row in betting_data.iterrows():
            if row['bet_amount'] > 0:
                message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id}, 組み合わせ: {row['Boat1']}-{row['Boat2']}-{row['Boat3']} に {row['bet_amount']} 円を投票しました。"
                log_vote_result(message)
                print(message)
                send_discord_message(message, DISCORD_WEBHOOK_URL)
        driver.close()
 

    except Exception as e:
        driver.close()
        # エラーが発生した場合、エラーメッセージを記録
        message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id} 投票中にエラーが発生しました: {e}"
        log_vote_result(message)
        print(message)
        send_discord_message(message, DISCORD_WEBHOOK_URL)

        # エラー時の投票予定内容を記録
        for index, row in betting_data.iterrows():
            if row['bet_amount'] > 0:
                message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id}, 組み合わせ: {row['Boat1']}-{row['Boat2']}-{row['Boat3']} に {row['bet_amount']} 円を投票予定でした。"
                log_vote_result(message)
                send_discord_message(message, DISCORD_WEBHOOK_URL)
                print(message)

    finally:
        driver.quit()
        

# betting_dataの作成
def create_betting_data(data):
    betting_data = data[['艇番', 'bet_amount']]
    print("Generated betting_data:")
    print(betting_data.head())
    return betting_data

# 8. inspect_modelを実行するスケジュール関数
def schedule_inspect_model(pending_races, processed_race_ids, session):
    """
    各レースの締め切り時刻の50秒前にinspect_modelを実行するようスケジュールする。
    
    Args:
        pending_races (list of dict): レースの情報 [{'race_id': ..., 'deadline_time': ...}, ...]
        processed_race_ids (set): 既に処理済みのrace_idのセット。
        session (requests.Session): 設定されたセッションオブジェクト。
    
    Returns:
        None
    """
    current_time = get_current_time_jst()
    for race in pending_races:
        race_id = race['race_id']
        table_num = race['table_num'] # <tbody>タグのインデックス
        deadline_time = race['deadline_time']
        race_id_time = race_id + deadline_time.strftime('%H%M%S')

        # deadline_timeの分数の最後の桁を取得
        last_digit_of_minute = int(str(deadline_time.minute)[-1])

        # 奇数か偶数かを判定してsecondsを設定
        if last_digit_of_minute % 2 == 1:
            inspect_time = deadline_time - timedelta(seconds=25)
        else:
            inspect_time = deadline_time - timedelta(seconds=25)
        
        if race_id_time in processed_race_ids :
            continue  # 既に処理済み
        
        delay = (inspect_time - current_time).total_seconds()
        if delay > 0:
            # タイマーを設定してinspect_modelを実行（race_idのみを渡す）
            timer = threading.Timer(delay, inspect_model, args=(race_id,table_num,deadline_time))
            timer.start()
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id} のinspect_modelを{inspect_time.strftime('%Y-%m-%d %H:%M:%S')}に予約しました。")
            processed_race_ids.add(race_id_time)
        else:
            # 締め切り時刻が既に過ぎている場合はスキップ
            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id} のinspect_timeが既に過ぎています。スキップします。")

import itertools

def process_race(race_id, race_odds, data_boats, total_capital=100000, unit=500, prob_threshold=0.0, ev_subtract=1.4):
    """
    各レースごとに三連単のボックス投票（上位4艇）を決定し、回収金額を計算します。
    Args:
        race_id (str): レースID
        race_odds (DataFrame): 該当レースの三連単オッズデータ
        data_boats (DataFrame): 各レースの各艇の確率データ
        total_capital (int): 総資金
        unit (int): 賭け金の単位（円）
        prob_threshold (float): この確率以下の組み合わせは賭けない
        ev_subtract (float): EV計算の減算値
    Returns:
        DataFrame: 各三連単の賭け金額と回収金額
    """
    # 該当レースの艇の確率データを取得
    boats_data = data_boats[data_boats['レースID'] == race_id]
    if boats_data.empty:
        # データがない場合
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'bet_amount']
        return pd.DataFrame(columns=columns)

    boats_probs = boats_data.set_index('艇番')['odds_inv_pred'].to_dict()
    # 正規化
    total_odds_inv_pred = sum(boats_probs.values())
    if total_odds_inv_pred == 0:
        # 合計が0の場合、計算できないのでスキップ
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'bet_amount']
        return pd.DataFrame(columns=columns)
    boats_probs = {k: v / total_odds_inv_pred for k, v in boats_probs.items()}

    # 確率が高い順に上位4艇を選択
    top_boats = boats_data.sort_values(by='odds_inv_pred', ascending=False)['艇番'].head(4).tolist()

    if len(top_boats) < 3:
        # 上位4艇が3艇未満の場合
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'bet_amount']
        return pd.DataFrame(columns=columns)

    # 全ての順列（三連単）
    trifecta_combinations = list(itertools.permutations(top_boats, 3))  # 4P3 = 24 combinations

    # Create a DataFrame of these combinations
    trifecta_df = pd.DataFrame(trifecta_combinations, columns=['Boat1', 'Boat2', 'Boat3'])

    # Merge with race_odds to get odds
    merged_df = pd.merge(trifecta_df, race_odds, on=['Boat1', 'Boat2', 'Boat3'], how='left')

    # Drop combinations that do not have odds (if any)
    merged_df = merged_df.dropna(subset=['Odds'])

    # Calculate probability
    merged_df['Probability'] = merged_df.apply(lambda row: calculate_trifecta_probability(
        boats_probs.get(row['Boat1'], 0),
        boats_probs.get(row['Boat2'], 0),
        boats_probs.get(row['Boat3'], 0)
    ), axis=1)

    # Calculate EV
    merged_df['EV'] = (merged_df['Probability'] * merged_df['Odds']) - 4.5
    # print(merged_df[merged_df['レースID'] == '202411032412'])
    # Filter based on EV
    print(f'merged_df: {merged_df}')
    merged_df = merged_df[merged_df['EV'] > 0]

    if merged_df.empty:
        # 賭けがなかった場合
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'bet_amount']
        return pd.DataFrame(columns=columns)

    # Assign bet amount (fixed unit per bet)
    merged_df['bet_amount'] = unit  # 固定ベット金額

    # # Determine the actual combination
    # boats_actual = boats_data.sort_values(by='着')['艇番'].tolist()
    # if len(boats_actual) < 3:
    #     actual_combo = ()
    # else:
    #     actual_combo = tuple(boats_actual[:3])

    # # Calculate payout
    # merged_df['Payout'] = merged_df.apply(
    #     lambda row: row['Bet_Amount'] * row['Odds'] if (row['Boat1'], row['Boat2'], row['Boat3']) == actual_combo else 0,
    #     axis=1
    # )

    # Assign race_id
    merged_df['レースID'] = race_id

    # Select and reorder columns
    trifecta_bet = merged_df[['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'bet_amount']]

    return trifecta_bet

from tqdm import tqdm

# メイン処理関数
def calculate_trifecta_bets(trifecta_odds_df, data_boats, total_capital=100000, unit=500, prob_threshold=0.0):
    """
    全レースに対して三連単のベット金額を決定し、回収率や資金の推移を計算します。
    Args:
        trifecta_odds_df (DataFrame): 三連単オッズのデータ
        data_boats (DataFrame): 各レースの各艇の確率データ
        total_capital (int): 総資金
        unit (int): 賭け金の単位（円）
        prob_threshold (float): この確率以下の組み合わせは賭けない
    Returns:
        DataFrame: 全ての賭け金と回収金額
        float: 総投資額
        float: 総回収額
        float: 回収率
    """
    # ユニークなレースIDを取得
    race_ids = trifecta_odds_df['レースID'].unique()
    
    all_trifecta_bets = []
    
    print("Processing races and calculating bets...")
    for race_id in tqdm(race_ids, desc="Processing Races"):
        race_odds = trifecta_odds_df[trifecta_odds_df['レースID'] == race_id]
        trifecta_bet = process_race(race_id, race_odds, data_boats, 
                                   total_capital=total_capital, 
                                   unit=unit, 
                                   prob_threshold=prob_threshold)
        if not trifecta_bet.empty:
            all_trifecta_bets.append(trifecta_bet)
            # 最初の数レースの結果を表示（ここでは最初の3レースに限定）
            if len(all_trifecta_bets) <= 3:
                print(f"\nレースID: {race_id} の賭け結果:")
                print(trifecta_bet)
    
    if all_trifecta_bets:
        trifecta_bets = pd.concat(all_trifecta_bets, ignore_index=True)
    else:
        # 全レースで賭けがなかった場合、必要な列を持つ空のDataFrameを作成
        columns = ['レースID', 'Boat1', 'Boat2', 'Boat3', 'Probability', 'Odds', 'EV', 'bet_amount']
        trifecta_bets = pd.DataFrame(columns=columns)
    
    # 日付情報の抽出
    if not trifecta_bets.empty:
        trifecta_bets['Date'] = trifecta_bets['レースID'].apply(lambda x: datetime.strptime(x[:8], '%Y%m%d'))
    else:
        trifecta_bets['Date'] = pd.to_datetime([])  # 空のDatetime列
    
    # 総投資額と総回収額を計算
    if not trifecta_bets.empty:
        total_investment = trifecta_bets['bet_amount'].sum()
        # total_payout = trifecta_bets['Payout'].sum()
        # return_rate = (total_payout / total_investment) * 100 if total_investment > 0 else 0
    else:
        total_investment = 0
        total_payout = 0
        return_rate = 0
    
    return trifecta_bets, total_investment, total_payout, return_rate

# 資金の推移を計算する関数
def calculate_capital_evolution(trifecta_bets, initial_capital=100000):
    """
    資金の推移を計算します。
    Args:
        trifecta_bets (DataFrame): 全ての賭け金と回収金額
        initial_capital (int): 初期資金
    Returns:
        DataFrame: 資金の推移
    """
    if trifecta_bets.empty:
        # 賭けがなかった場合、初期資金のみのDataFrameを返す
        return pd.DataFrame({
            'Cumulative_Investment': [0],
            'Cumulative_Payout': [0],
            'Net': [0],
            'Capital': [initial_capital]
        })
    
    # 日付順にソート
    trifecta_bets_sorted = trifecta_bets.sort_values(by=['Date']).reset_index(drop=True)
    trifecta_bets_sorted['Cumulative_Investment'] = trifecta_bets_sorted['bet_amount'].cumsum()
    trifecta_bets_sorted['Cumulative_Payout'] = trifecta_bets_sorted['Payout'].cumsum()
    trifecta_bets_sorted['Net'] = trifecta_bets_sorted['Cumulative_Payout'] - trifecta_bets_sorted['Cumulative_Investment']
    trifecta_bets_sorted['Capital'] = initial_capital + trifecta_bets_sorted['Net']
    return trifecta_bets_sorted

# 結果の可視化関数
def plot_results(trifecta_bets, capital_evolution):
    """
    回収率や資金の推移をプロットします。
    Args:
        trifecta_bets (DataFrame): 全ての賭け金と回収金額
        capital_evolution (DataFrame): 資金の推移
    """
    # 回収率の表示
    total_investment = trifecta_bets['bet_amount'].sum() if not trifecta_bets.empty else 0
    total_payout = trifecta_bets['Payout'].sum() if not trifecta_bets.empty else 0
    return_rate = (total_payout / total_investment) * 100 if total_investment > 0 else 0
    print(f"\n総投資額: {total_investment:.2f}円")
    print(f"総回収額: {total_payout:.2f}円")
    print(f"回収率: {return_rate:.2f}%")
    
    # 最初の数レースの結果を表示
    if not trifecta_bets.empty:
        print("\n最初の数レースの詳細:")
        print(trifecta_bets.head(5))
    else:
        print("\n賭けがなかったため、レースの詳細はありません。")
    
    # 資金の推移プロット
    if capital_evolution.empty or (capital_evolution.shape[0] == 1 and capital_evolution['Capital'].iloc[0] == 100000):
        print("\n資金の推移をプロットできません。")
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(capital_evolution['Date'], capital_evolution['Capital'], marker='o', linestyle='-', color='blue')
        plt.title("資金の推移")
        plt.xlabel("日付")
        plt.ylabel("資金 (円)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('capital_evolution_trifecta.png')
        plt.show()
        print("資金の推移が 'capital_evolution_trifecta.png' として保存されました。")
    
    # 回収金額のヒストグラム
    if not trifecta_bets.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(trifecta_bets['Payout'], bins=50, kde=True, color='green')
        plt.title("回収金額の分布")
        plt.xlabel("回収金額 (円)")
        plt.ylabel("度数")
        plt.grid(True)
        plt.savefig('payout_distribution_trifecta.png')
        plt.show()
        print("回収金額の分布が 'payout_distribution_trifecta.png' として保存されました。")
    
        # 賭け金額のヒストグラム
        plt.figure(figsize=(10, 6))
        sns.histplot(trifecta_bets['bet_amount'], bins=50, kde=True, color='orange')
        plt.title("賭け金額の分布")
        plt.xlabel("賭け金額 (円)")
        plt.ylabel("度数")
        plt.grid(True)
        plt.savefig('bet_amount_distribution_trifecta.png')
        plt.show()
        print("賭け金額の分布が 'bet_amount_distribution_trifecta.png' として保存されました。")
    
        # 投資金額と回収金額の比較プロット
        plt.figure(figsize=(12, 6))
        trifecta_bets_sorted = trifecta_bets.sort_values(by=['Date'])
        plt.bar(trifecta_bets_sorted['Date'], trifecta_bets_sorted['bet_amount'], label='投資金額', alpha=0.6, color='red')
        plt.bar(trifecta_bets_sorted['Date'], trifecta_bets_sorted['Payout'], label='回収金額', alpha=0.6, color='blue')
        plt.title("投資金額と回収金額の比較")
        plt.xlabel("日付")
        plt.ylabel("金額 (円)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('investment_vs_payout_trifecta.png')
        plt.show()
        print("投資金額と回収金額の比較が 'investment_vs_payout_trifecta.png' として保存されました。")
    
        # 日付を横軸とした累積収益の推移プロット
        if 'Net' in capital_evolution.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(capital_evolution['Date'], capital_evolution['Net'].cumsum(), marker='o', linestyle='-', color='purple')
            plt.title("日付を横軸とした累積収益の推移")
            plt.xlabel("日付")
            plt.ylabel("累積収益 (円)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('cumulative_profit_trifecta.png')
            plt.show()
            print("日付を横軸とした累積収益の推移が 'cumulative_profit_trifecta.png' として保存されました。")
        else:
            print("Net列が存在しないため、累積収益の推移をプロットできません。")

def load_trifecta_odds(date, place_cd, race_no):
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')  # 例: '2023-11-01' 形式の文字列を変換
    month = date.strftime('%y%m')
    date_str = date.strftime('%y%m%d')
    file_path = os.path.join(r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\trifecta_odds1min', month, f'trifecta1min_{date_str}.txt')
    
    # ファイルが存在しない場合はエラー
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    # 読み込みとフィルタリング
    trifecta_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        recording = False
        for line in f:
            if f"Place: {place_cd}, Race: {race_no}" in line:
                recording = True
            elif recording and line.strip() == "":
                break
            elif recording:
                parts = line.strip().split(',')
                boat1 = int(parts[0].split(': ')[1])
                boat2 = int(parts[1].split(': ')[1])
                boat3 = int(parts[2].split(': ')[1])
                odds = float(parts[3].split(': ')[1])
                trifecta_data.append({'Boat1': boat1, 'Boat2': boat2, 'Boat3': boat3, 'Odds': odds})
    
    trifecta_df = pd.DataFrame(trifecta_data)
    print(f"Loaded {len(trifecta_df)} trifecta odds entries for Place {place_cd}, Race {race_no} on {date_str}")
    return trifecta_df

# def load_trifecta_odds(base_dir, start_date, end_date):
#     """
#     指定されたディレクトリから三連単オッズデータを読み込み、1つのDataFrameに統合します。
#     Args:
#         base_dir (str): 三連単オッズデータが保存されているベースディレクトリ
#         start_date (datetime): 開始日
#         end_date (datetime): 終了日
#     Returns:
#         DataFrame: 統合された三連単オッズデータ
#     """
#     all_data = []
#     current_date = start_date
#     total_days = (end_date - start_date).days + 1
#     pbar = tqdm(total=total_days, desc="Loading Trifecta Odds")

#     while current_date <= end_date:
#         date_str = current_date.strftime('%y%m%d')  # 修正箇所
#         yymm = current_date.strftime('%y%m')  # フォルダ名が 'yymm' の場合
#         file_name = f'trifecta_{date_str}.csv'
#         file_path = os.path.join(base_dir, yymm, file_name)
        
#         if os.path.exists(file_path):
#             try:
#                 df = pd.read_csv(file_path)
#                 # 生成されたDataFrameにレースIDを追加
#                 df['レースID'] = df.apply(
#                     lambda row: f"{current_date.strftime('%Y%m%d')}{int(row['JCD']):02d}{int(row['Race']):02d}",
#                     axis=1
#                 )
#                 all_data.append(df)
#                 print(f"Loaded {file_path} with {len(df)} records.")
#             except Exception as e:
#                 print(f"Error loading {file_path}: {e}")
#         else:
#             print(f"File not found: {file_path}")
        
#         current_date += timedelta(days=1)
#         pbar.update(1)
    
#     pbar.close()
    
#     if all_data:
#         trifecta_odds_data_df = pd.concat(all_data, ignore_index=True)
#         print(f"Total trifecta odds records loaded: {len(trifecta_odds_data_df)}")
#         return trifecta_odds_data_df
#     else:
#         print("No trifecta odds data loaded.")
#         return pd.DataFrame()
    
# 9. メインループの定義

def calculate_trifecta_probability(p1, p2, p3):
    """
    三連単の確率を計算します。
    """
    # 各値の合計を計算
    # total = p1 + p2 + p3
    # if total == 0:
    #     return 0
    # p1 = p1 / total
    # p2 = p2 / total
    # p3 = p3 / total
    denominator1 = 1 - p1
    denominator2 = 1 - p1 - p2
    if denominator1 <= 0 or denominator2 <= 0:
        return 0
    return (p1 * p2 * p3/denominator1/denominator2) 

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

def get_today_races():
    """
    当日の各レースの情報と投票締め切り時刻を取得します。
    ここではひな形として固定のデータを返しますが、実際にはウェブスクレイピング等で取得してください。
    Returns:
        list of dict: 各レースの情報（レースID、投票ページのリンク、締め切り時刻）
    """
    # ひな形データ
    races = [
        {
            'race_id': 'race1',
            'link': 'https://www.boatrace.jp/owpc/pc/race/racelist?rno=9&jcd=04&hd=20241014#',
            'deadline': datetime(2024, 10, 14, 14, 30)  # YYYY, MM, DD, HH, MM
        },
        {
            'race_id': 'race2',
            'link': 'https://www.boatrace.jp/owpc/pc/race/racelist?rno=10&jcd=05&hd=20241014#',
            'deadline': datetime(2024, 10, 14, 15, 00)
        },
        # 他のレースを追加
    ]
    return races

def fetch_race_info_by_id(race_id, session=None):
    """
    指定したレースIDに基づいてレース情報を取得する関数。

    Args:
        race_id (str): レースの一意識別子。形式は 'YYYYMMDDPPRR'（例: '202410140404'）。
        session (requests.Session, optional): 使用するセッション。デフォルトはNoneで新規作成。

    Returns:
        pd.Series or None: 取得したレース情報が格納された pandas.Series、取得に失敗した場合は None。
    """
    # race_id の形式を確認
    if not isinstance(race_id, str):
        print("Error: race_id must be a string.")
        return None
    if len(race_id) != 12 or not race_id.isdigit():
        print("Error: race_id format is incorrect. Expected 12 digits 'YYYYMMDDPPRR'.")
        return None

    # race_id から date, place_cd, race_no を抽出
    try:
        date_str = pd.to_datetime(race_id[:8], format='%Y%m%d').strftime('%Y-%m-%d')
        place_cd = int(race_id[8:10])  # 例: '202410140404' -> place_cd=04
        race_no = int(race_id[10:12])  # 例: '202410140404' -> race_no=04
    except Exception as e:
        print(f"Error parsing race_id: {e}")
        return None

    print(f"Parsed race_id '{race_id}' as date: {date_str}, place_cd: {place_cd}, race_no: {race_no}")

    # セッションが指定されていない場合、新たに作成
    if session is None:
        session = requests_session()
        close_session = True
    else:
        close_session = False

    # レース情報を取得
    args = (date_str, place_cd, race_no)

    try:
        data, trifecta_odds = get_beforeinfo(args, session)
        if isinstance(data, pd.Series):
            print(f"Successfully fetched data for race_id {race_id}")
        elif isinstance(data, str):
            print(f"Race ID {race_id} skipped: {data}")
            data = None
        else:
            print(f"Race ID {race_id} returned no data.")
            data = None
        if isinstance(trifecta_odds, list):
            print(f"Successfully fetched trifecta odds for race_id {race_id}")
        elif isinstance(trifecta_odds, str):
            print(f"Race ID {race_id} skipped: {trifecta_odds}")
            trifecta_odds = None
        else:
            print(f"Race ID {race_id} returned no trifecta odds.")
            trifecta_odds = None

    except Exception as e:
        print(f"Exception occurred while fetching data for race_id {race_id}: {e}")
        traceback.print_exc()
        data = None
        trifecta_odds = None

    # セッションを閉じる必要がある場合は閉じる
    if close_session:
        session.close()

    return data, trifecta_odds

def process_race_series(race_series):
    """
    指定されたレースデータ（pandas.Series）をワイド形式からロング形式のDataFrameに変換します。

    Args:
        race_series (pd.Series): 1レース分のデータが格納された pandas.Series。

    Returns:
        pd.DataFrame: ロング形式のデータフレーム。各艇の情報が1行ずつ格納される。
    """
    if not isinstance(race_series, pd.Series):
        print("Error: race_series must be a pandas Series.")
        return pd.DataFrame()

    required_columns = ['date', 'place_cd', 'race_no', 'weather', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h']
    for col in required_columns:
        if col not in race_series:
            print(f"Error: Missing required column '{col}' in race_series.")
            return pd.DataFrame()

    # 'レースID' の作成
    race_series['date'] = pd.to_datetime(race_series['date'])
    ymd = race_series['date'].strftime('%Y%m%d')
    place_cd = f"{int(race_series['place_cd']):02d}"
    race_no = f"{int(race_series['race_no']):02d}"
    race_id = ymd + place_cd + race_no

    # 各艇のデータを抽出
    boats = []
    for i in range(1, 7):  # 1号艇から6号艇
        boat_data = {
            'レースID': race_id,
            '艇番': i,
            'ET': race_series.get(f'ET_{i}', np.nan),
            'tilt': race_series.get(f'tilt_{i}', np.nan),
            'EST': race_series.get(f'EST_{i}', np.nan),
            'ESC': race_series.get(f'ESC_{i}', np.nan),
            'weather': race_series.get('weather', np.nan),
            'air_t': race_series.get('air_t', np.nan),
            'wind_d': race_series.get('wind_d', np.nan),
            'wind_v': race_series.get('wind_v', np.nan),
            'water_t': race_series.get('water_t', np.nan),
            'wave_h': race_series.get('wave_h', np.nan),
            'win_odds': race_series.get(f'win_odds_{i}', np.nan),
            'place_odds': race_series.get(f'place_odds_{i}', np.nan),
        }
        boats.append(boat_data)

    df_long = pd.DataFrame(boats)

    return df_long

def save_race_data(race_series, base_dir=r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\before_data_1min'):
    """
    fetch_race_info_by_idによって取得されたレースデータ（pandas.Series）をテキストファイルに保存します。
    
    Args:
        race_series (pd.Series): 1レース分のデータが格納された pandas.Series。
        base_dir (str): ベースディレクトリのパス。
    
    Returns:
        None
    """
    if not isinstance(race_series, pd.Series):
        print("Error: race_series must be a pandas Series.")
        return
    
    required_fields = ['date', 'place_cd', 'race_no']
    for field in required_fields:
        if field not in race_series:
            print(f"Error: Missing required field '{field}' in race_series.")
            return
    
    # レース情報を取得
    date_str = pd.to_datetime(race_series['date']).strftime('%Y%m%d')  # 'yyyymmdd'
    place_cd = f"{int(race_series['place_cd']):02d}"  # 'PP'
    race_no = f"{int(race_series['race_no']):02d}"  # 'RR'
    
    yy = date_str[2:4]  # 'yy'
    mm = date_str[4:6]  # 'mm'
    dd = date_str[6:8]  # 'dd'
    
    # フォルダ名 'YYMM' の作成
    folder_name = yy + mm
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # テキストファイル名 'beforeinfo1min_yyyymmdd.txt' の作成
    txt_file_name = f'beforeinfo1min_{yy}{mm}{dd}.txt'
    txt_file_path = os.path.join(folder_path, txt_file_name)
    
    try:
        # ファイルが存在するか確認
        file_exists = os.path.isfile(txt_file_path)
        
        with open(txt_file_path, 'a', encoding='utf-8-sig') as f:
            if file_exists:
                # ファイルにデータを追記する前に改行を挿入
                f.write('\n')
            else:
                # ヘッダーを書き込む（キー名）
                headers = race_series.index.tolist()
                # ヘッダーをスペースで整列させるために最大キー長を計算
                max_key_length = max(len(str(key)) for key in headers)
                for key in headers:
                    f.write(f"{key.ljust(max_key_length + 4)}{race_series[key]}\n")
            # データを追記
            headers = race_series.index.tolist()
            max_key_length = max(len(str(key)) for key in headers)
            for key in headers:
                f.write(f"{key.ljust(max_key_length + 4)}{race_series[key]}\n")
        action = "Appended" if file_exists else "Created and saved"
        print(f"{action} race data to {txt_file_path}")
    except Exception as e:
        print(f"Error saving race data to {txt_file_path}: {e}")


def main():
    # """
    # メインループ。15秒おきにレース情報をチェックし、該当するrace_idが取得されたらinspect_model(race_id)を呼び出す。
    # """
    # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] レースID取得スクリプトを開始します。")
    
    # # 処理済みのrace_idを保持するセット
    # processed_race_ids = set()
    # session = requests_session()

    # while True:
    #     race_ids = fetch_race_ids_before_deadline(session)
    #     if race_ids:
    #         for rid in race_ids:
    #             if rid not in processed_race_ids:
    #                 inspect_model(rid)
    #                 processed_race_ids.add(rid)
    #             else:
    #                 print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 既に処理済みのrace_id: {rid}")
    #     else:
    #         print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 該当するレースはありませんでした。")
        
    #     # 次のチェックまで待機 (15秒)
    #     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 5秒後に次のチェックを行います。\n")
    #     time.sleep(5)

    """
    メインループ。1分ごとにレースの締め切り時刻を取得し、
    各レースの締め切り時刻の50秒前にinspect_model(race_id)を呼び出すようスケジュールする。
    """
    print(f"[{get_current_time_jst().strftime('%Y-%m-%d %H:%M:%S')}] レースID取得スクリプトを開始します。")
    
    # 処理済みのrace_idを保持するセット（ファイルから読み込み）
    processed_race_ids = load_processed_race_ids()
    
    # セッションを作成
    session = requests_session()
    
    while True:
        # 1分ごとにレースの締め切り時刻を取得
        pending_races = fetch_race_deadlines(session)
        
        # スケジューリング
        schedule_inspect_model(pending_races, processed_race_ids, session)
        
        # 処理済みrace_idをファイルに保存
        save_processed_race_ids(processed_race_ids)
        
        # 次のチェックまで1分待機
        print(f"[{get_current_time_jst().strftime('%Y-%m-%d %H:%M:%S')}] 1分後に次の締め切り時刻を取得します。\n")
        time.sleep(60)

if __name__ == '__main__':
    main()
