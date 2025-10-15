# investment_system.py
from bs4 import BeautifulSoup
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
from save_test2 import get_url, requests_session, get_beforeinfo, save_beforeinfo_data
import pandas as pd
import lightgbm as lgb
from ...utils.mymodule import load_b_file, load_k_file, preprocess_data, load_before_data, merge_before_data
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
def save_processed_race_ids(processed_race_ids, file_path='processed_race_ids2.json'):
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

def inspect_model(race_id,table_num,deadline_time):

    logging.info(f"inspect_modelを呼び出しました。race_id: {race_id}")

    # 学習済みモデルの読み込み
    try:
        gbm = lgb.Booster(model_file='boatrace_model_first.txt')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 特徴量名の読み込み
    feature_names = pd.read_csv('feature_names_first.csv').squeeze().tolist()

    # 評価期間のデータ日付リスト作成
    month_folders = ['2410']
    date_files = {
        # '2408': [f'2408{day:02d}' for day in range(1, 32)],  # 240801 - 240831
        # '2409': [f'2409{day:02d}' for day in range(1, 31)],  # 240901 - 240930
        '2410': [f'2410{day:02d}' for day in range(23, 24)],   # 241001 - 241008
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
                # k_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/k_data/{month}/K{date}.TXT'
                before_file = f'C:/Users/kazut/OneDrive/Documents/BoatRace/ML_pred/before_data_1min/{month}/beforeinfo1min_{date}.txt'

                b_data = load_b_file(b_file)
                # k_data, odds_list_part = load_k_file(k_file)
                # before_data = load_before_data(before_file)]\
                before_data=fetch_race_info_by_id(race_id)
                print(before_data)
                save_race_data(before_data)
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

    # 全データを結合
    data = pd.concat(data_list, ignore_index=True)
    # odds_list = pd.concat(odds_list1, ignore_index=True)

    # 前処理
    data = preprocess_data(data)

    print("data columns:", data.columns.tolist())
    print(data)

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
    print("X_processed columns:", X_processed.columns.tolist())
    X_processed = X_processed[feature_names]  # 学習時の特徴量と順序を合わせる

    # 予測
    y_pred = gbm.predict(X_processed)
    # data['prob_0'] = y_pred[:, 0]  # 1,2着の確率
    # data['prob_1'] = y_pred[:, 1]
    # data['prob_2'] = y_pred[:, 2]

    data['prob_0'] = y_pred[:, 0]  # 1,着の確率
    data['prob_1'] = y_pred[:, 1]
    # data['prob_2'] = y_pred[:, 2]

    # 'prob_0' の分布をプロット
    plt.figure(figsize=(8, 6))
    plt.hist(data['prob_0'], bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title("予測確率 'prob_0' の分布")
    plt.xlabel("予測確率 (prob_0)")
    plt.ylabel("度数")
    plt.grid(True)
    plt.savefig('prob_0_distribution.png')
    # plt.show()
    print("プロットが 'prob_0_distribution.png' として保存されました。")

    print(data)

    data = data.dropna(subset=['win_odds'])

    # オッズをデシマルオッズに変換
    data['decimal_odds'] = data['win_odds'] 
    print(data['decimal_odds'])
    print(data['prob_0'])

    # ケリー基準に基づいて掛け金割合を計算
    data['bet_fraction'] = data.apply(lambda row: calculate_kelly_criterion(row['prob_0'], row['decimal_odds']), axis=1)

    # 掛け金を計算（総資金を100,000円と仮定）
    total_capital = 1000 # 総資金（円）
    data['bet_amount'] = data['bet_fraction'] * total_capital
    # 【追加部分】bet_amount を100円単位に調整
    # bet_amount を100円の倍数に丸め、bet_fractionが0より大きい場合は最低100円に設定
    data['bet_amount'] = data['bet_amount'].apply(round_bet_amount)
    data['bet_amount'] = np.where(data['prob_0'] < 0.2, 0, data['bet_amount'])


    # 掛け金がNaNの場合は0に置換
    data['bet_amount'] = data['bet_amount'].fillna(0)
    print(data[['レースID','艇番','prob_0','win_odds','bet_amount']].head(500))


    selected_data = data[['レースID', '艇番', 'prob_0', 'win_odds', 'bet_amount']]
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
            selected_data.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] データを新規保存しました: {file_path}")
        else:
            # ヘッダーなしでアペンド
            selected_data.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] データをアペンドしました: {file_path}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] データの保存中にエラーが発生しました: {e}")
    
    # '艇番'と'bet_amount'のみを抽出
    betting_data = data[['艇番', 'bet_amount']]

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

    bet_amounts_per_boat = {i: 0 for i in range(1, 7)}
    # 投票締め切り時刻を過ぎていたら投票しない
    now = datetime.now(timezone.utc)  # 例として UTC タイムゾーンを使用
    if now > deadline_time:
        # message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id}, 舟番: {boat_number} に {bet_amount} 円を投票しました。 "
        # log_vote_result(message)
        # print(message)
        # return
        # 舟番ごとの投票結果をログに記録
        for boat_number, bet_amount in bet_amounts_per_boat.items():
            if bet_amount > 0:
                message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id}, 舟番: {boat_number} に {bet_amount} 円を投票しようとしました。"
                log_vote_result(message)
                print(message)
            message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 投票締め切り時刻を過ぎているため、投票を行いません。"
            log_vote_result(message)
        print("投票締め切り時刻を過ぎているため、投票を行いません。")
        return

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
        # レースページにアクセス
        driver.get('https://www.boatrace.jp/owpc/pc/race/index')
        time.sleep(1)  # ページがロードされるまで待機

        # race_idに基づいて投票ボタンを探す
        vote_button_id = f"TENTP090A{table_num}"
        vote_button = driver.find_element(By.ID, vote_button_id)
        vote_button.click()
        time.sleep(1)  # ポップアップが開くまで待機

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

        # 新しいウィンドウが開いたか確認して処理
        original_window = driver.current_window_handle
        wait.until(EC.number_of_windows_to_be(2))  # ウィンドウが2つになるまで待機

        # 新しいウィンドウに切り替える
        for window_handle in driver.window_handles:
            if window_handle != original_window:
                driver.switch_to.window(window_handle)
                print(f"新しいウィンドウに切り替えました: {window_handle}")
                break

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
        vote_button = driver.find_element(By.XPATH, '//*[@id="betkati1"]/a')
        vote_button.click()
        time.sleep(2)  

        # 投票を実行
        for index, row in betting_data.iterrows():
            boat_number = row['艇番']
            bet_amount = row['bet_amount']
            if bet_amount == 0:
                continue
            bet_amounts_per_boat[boat_number] += bet_amount

            select_boat = driver.find_element(By.XPATH, f'//*[@id="regbtn_{boat_number}_1"]/a')
            select_boat.click()

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
            return
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
        # confirm_button = driver.find_element(By.XPATH, '//*[@id="ok"]')
        # confirm_button.click()

        time.sleep(2)  # 購入が完了するまで待機
        
        # 舟番ごとの投票結果をログに記録
        for boat_number, bet_amount in bet_amounts_per_boat.items():
            if bet_amount > 0:
                message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id}, 舟番: {boat_number} に {bet_amount} 円を投票しました。"
                log_vote_result(message)
                print(message)

    except Exception as e:
        # message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id} 投票中にエラーが発生しました: {e}"
        # log_vote_result(message)
        # print(message)
        for boat_number, bet_amount in bet_amounts_per_boat.items():
            if bet_amount > 0:
                message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] race_id: {race_id}, 舟番: {boat_number} に {bet_amount} 円を投票しようとしました。"
                log_vote_result(message)
                print(message)
        message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 投票中にエラーが発生しました: {e}"
        print(message)

    finally:
        driver.quit()

    # 各艇の勝敗を判定（1着の場合勝利）
    # data['win'] = (data['着'] == 1).astype(int)

    # 回収金額を計算
    # data['payout'] = data['bet_amount'] * data['decimal_odds'] * data['win']



    #     # 'data' における各艇の複勝オッズを取得
    # def get_fukusho_odds(row):
    #     boat_number = int(row['艇番'])
    #     odds_column = f'複勝オッズ{boat_number}'
    #     if odds_column in data.columns:
    #         return row[odds_column]
    #     else:
    #         return np.nan

    # data['複勝オッズ'] = data.apply(get_fukusho_odds, axis=1)

    # # オッズが欠損している行を削除
    # data = data.dropna(subset=['複勝オッズ'])

    # # オッズをデシマルオッズに変換
    # data['fukusho_decimal_odds'] = data['複勝オッズ'] / 100

    # # 複勝の勝率を推定（1着または2着になる確率）
    # data['prob_fukusho'] = data['prob_0'] + data['prob_1']

    # # ケリー基準に基づいて複勝の掛け金割合を計算
    # data['bet_fraction_fukusho'] = data.apply(
    #     lambda row: calculate_kelly_criterion(row['prob_fukusho'], row['fukusho_decimal_odds']),
    #     axis=1
    # )

    # # 掛け金を計算（複勝）
    # data['bet_amount_fukusho'] = data['bet_fraction_fukusho'] * total_capital

    # # 掛け金がNaNの場合は0に置換
    # data['bet_amount_fukusho'] = data['bet_amount_fukusho'].fillna(0)

    # # 各艇の勝敗を判定（複勝：1着または2着の場合勝利）
    # data['win_fukusho'] = data['着'].isin([1, 2]).astype(int)

    # # 回収金額を計算（複勝）
    # data['payout_fukusho'] = data['bet_amount_fukusho'] * data['fukusho_decimal_odds'] * data['win_fukusho']

    # # 複勝の総投資額と総回収額を計算
    # total_investment_fukusho = data['bet_amount_fukusho'].sum()
    # total_return_fukusho = data['payout_fukusho'].sum()

    # # 回収率を計算（複勝）
    # return_rate_fukusho = (total_return_fukusho / total_investment_fukusho) * 100 if total_investment_fukusho > 0 else 0

    # print(f"\n複勝の総投資額: {total_investment_fukusho:.2f}円")
    # print(f"複勝の総回収額: {total_return_fukusho:.2f}円")
    # print(f"複勝の回収率: {return_rate_fukusho:.2f}%")

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
            inspect_time = deadline_time - timedelta(seconds=30)
        else:
            inspect_time = deadline_time - timedelta(seconds=30)
        
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

# 9. メインループの定義

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
        data = get_beforeinfo(args, session)
        if isinstance(data, pd.Series):
            print(f"Successfully fetched data for race_id {race_id}")
        elif isinstance(data, str):
            print(f"Race ID {race_id} skipped: {data}")
            data = None
        else:
            print(f"Race ID {race_id} returned no data.")
            data = None
    except Exception as e:
        print(f"Exception occurred while fetching data for race_id {race_id}: {e}")
        traceback.print_exc()
        data = None

    # セッションを閉じる必要がある場合は閉じる
    if close_session:
        session.close()

    return data

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
