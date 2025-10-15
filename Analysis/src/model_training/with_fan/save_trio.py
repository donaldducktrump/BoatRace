import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import sys
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

# --- 設定部分 ---

# 日付範囲の設定（開始日と終了日を設定）
START_DATE = datetime.strptime('2024-11-01', '%Y-%m-%d')
END_DATE = datetime.strptime('2024-11-05', '%Y-%m-%d')

# 保存先のベースディレクトリ
BASE_DIR = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\trio_odds'

# レース番号と場コードの設定
RACE_NUMBERS = list(range(1, 13))  # レース番号1～12
JCD_CODES = [f"{i:02d}" for i in range(1, 25)]  # JCD01～24

# URLテンプレート
URL_TEMPLATE_TAN = 'https://www.boatrace.jp/owpc/pc/race/odds3f?rno={rno}&jcd={jcd}&hd={date}'

# タイムアウト設定（秒）
TIMEOUT = 15

# スレッド数の設定
MAX_WORKERS = 400  # システムとサーバーに応じて調整してください

# --- リクエストセッションとリトライ戦略の設定 ---

def requests_session():
    session = requests.Session()
    retry = Retry(
        total=5,  # 最大リトライ回数（HTTPエラーに対するリトライ）
        backoff_factor=0.3,  # リトライ間隔の増加率
        status_forcelist=(500, 502, 504),  # リトライ対象のステータスコード
        allowed_methods=["GET", "POST"]  # リトライ対象のメソッド
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

# --- URL生成関数 ---

def get_trio_odds_url(date, place_cd, race_no):
    """
    三連単オッズのURLを生成します。
    
    Args:
        date (str): 日付（'YYYYMMDD'形式）
        place_cd (str): 場コード（'01'～'24'）
        race_no (int): レース番号（1～12）
    
    Returns:
        str: 三連単オッズのURL
    """
    return URL_TEMPLATE_TAN.format(rno=race_no, jcd=place_cd, date=date)
# --- HTML解析関数 ---

def parse_trio_odds(html_content):
    """
    HTMLコンテンツから三連単オッズを解析します。
    指定された20種類の組み合わせのみを抽出します。
    
    Args:
        html_content (str): 取得したHTMLコンテンツ
    
    Returns:
        list of dict: 三連単の組み合わせとオッズ
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    trio_data = []
    
    # テーブルのtbodyを取得
    tbody = soup.find('tbody', class_='is-p3-0')
    if not tbody:
        print("Warning: <tbody class='is-p3-0'> が見つかりませんでした。")
        return trio_data
    
    rows = tbody.find_all('tr')
    
    boat1 = None  # 現在の1着艇番号
    boat2_list = []  # 各行での2着艇番号リスト
    
    for row_index, row in enumerate(rows, start=1):
        cols = row.find_all('td')
        if not cols:
            continue
        
        # 行ごとの処理
        if row_index in [1,5,8,10]:
            # x = 1 mod 4 の<tr>
            # 各セットは3つの<td>で構成される: Boat2, Boat3, Odds
            num_sets = len(cols) // 3
            for set_index in range(num_sets):
                try:
                    base = set_index * 3
                    boat2_num = cols[base].get_text(strip=True)
                    boat3_num = cols[base + 1].get_text(strip=True)
                    odds_text = cols[base + 2].get_text(strip=True)
                    
                    # 2着艇番号をリストに追加
                    boat2_list.append(boat2_num)
                    
                    # Boat1の取得方法: 行ごとの1着艇番号
                    # ここではセットインデックス+1を仮定していますが、実際のHTML構造に基づいて調整してください
                    boat1_num = set_index + 1  # 1着艇番号を取得
                    boat1 = boat1_num
                    
                    # オッズの取得
                    odds = float(odds_text)
                    
                    trio_data.append({
                        'Boat1': boat1,
                        'Boat2': boat2_num,
                        'Boat3': boat3_num,
                        'Odds': odds
                    })
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: セット {set_index +1} のデータ取得中にエラーが発生しました ({e})。スキップします。")
                    continue
        else:
            # x != 1 mod 4 の<tr>
            # 各セットは2つの<td>で構成される: Boat3, Odds
            num_sets = len(cols) // 2
            for set_index in range(num_sets):
                try:
                    base = set_index * 2
                    boat3_num = cols[base].get_text(strip=True)
                    odds_text = cols[base + 1].get_text(strip=True)
                    
                    boat1_num = set_index + 1  # 1着艇番号を取得
                    boat1 = boat1_num
                    
                    if set_index < len(boat2_list):
                        boat2 = boat2_list[set_index]  # 2着艇番号を取得
                    else:
                        print(f"Warning: boat2_listに十分なデータがありません。セット {set_index +1} をスキップします。")
                        continue
                    
                    if not boat2:
                        print("Warning: 2着艇番号が未設定です。スキップします。")
                        continue
                    
                    # オッズの取得
                    odds = float(odds_text)
                    
                    trio_data.append({
                        'Boat1': boat1,
                        'Boat2': boat2,
                        'Boat3': boat3_num,
                        'Odds': odds
                    })
                except (ValueError, IndexError) as e:
                    print(f"Warning: セット {set_index +1} のデータ取得中にエラーが発生しました ({e})。スキップします。")
                    continue
    
    return trio_data

# --- データ取得関数 ---

def fetch_trio_odds(args, session):
    """
    指定されたレース番号、場コード、日付のページから三連単オッズを取得する。
    
    Args:
        args (tuple): (date_str, place_cd, race_no)
        session (requests.Session): リクエストセッション
    
    Returns:
        DataFrame or None: 取得したオッズデータのDataFrameまたはNone
    """
    date_str, place_cd, race_no = args
    url = get_trio_odds_url(date_str, place_cd, race_no)
    print(f"Fetching 三連複 URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    
    try:
        response = session.get(url, headers=headers, timeout=TIMEOUT)
        if response.status_code != 200:
            print(f"Failed to retrieve data. HTTP Status Code: {response.status_code} for URL: {url}")
            return None
        
        trio_data = parse_trio_odds(response.text)
        if not trio_data:
            print(f"No trio data found for date {date_str}, place_cd {place_cd}, race_no {race_no}")
            return None
        
        df = pd.DataFrame(trio_data)
        df['Date'] = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
        df['JCD'] = place_cd
        df['Race'] = race_no
        
        # 列の順序を調整
        df = df[['Date', 'JCD', 'Race', 'Boat1', 'Boat2', 'Boat3', 'Odds']]
        
        print(f"Successfully retrieved 三連複 data for date {date_str}, place_cd {place_cd}, race_no {race_no}")
        return df
    
    except requests.exceptions.ReadTimeout:
        print(f"ReadTimeout occurred while accessing URL: {url}")
        return None
    except Exception as e:
        print(f"Error occurred while fetching data from URL: {url}")
        traceback.print_exc()
        return None

# --- メイン処理関数 ---

def save_trio_data():
    """
    指定された期間の三連単オッズデータを取得して保存します。
    1日の全てのレースを並列で処理します。
    """
    # 日付範囲の生成
    date_range = pd.date_range(START_DATE, END_DATE)
    
    # データを保存するフォルダを作成
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # セッションの作成
    session = requests_session()
    
    for date in date_range:
        date_str = date.strftime('%Y%m%d')
        date_str_name = date.strftime('%y%m%d')
        month_str = date.strftime('%y%m')
        save_dir = os.path.join(BASE_DIR, month_str)
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存パスの設定
        save_path_trio = os.path.join(save_dir, f'trio_{date_str_name}.csv')
        
        # 既に保存済みの場合はスキップ
        if os.path.exists(save_path_trio):
            print(f"Already exists: {save_path_trio}")
            continue
        
        print(f"Processing date {date_str}")
        all_data_trio = []
        
        # 全ての (place_cd, race_no) の組み合わせを作成
        args_list = []
        for place_cd in JCD_CODES:
            for race_no in RACE_NUMBERS:
                args_list.append((date_str, place_cd, race_no))
        
        # 並列処理の開始
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 三連単のフェッチタスクを作成
            future_to_args_trio = {executor.submit(fetch_trio_odds, args, session): args for args in args_list}
            
            # as_completedにFutureオブジェクトのみを渡す
            for future in as_completed(future_to_args_trio):
                args = future_to_args_trio[future]
                date_str, place_cd, race_no = args
                try:
                    result = future.result()
                    if result is not None:
                        all_data_trio.append(result)
                except Exception as exc:
                    print(f"Exception occurred for date {date_str}, place_cd {place_cd}, race_no {race_no}: {exc}")
                    traceback.print_exc()
        
        # 保存処理
        if all_data_trio:
            full_df_trio = pd.concat(all_data_trio, ignore_index=True)
            # 欠損値をNaNで埋める（既に処理済み）
            full_df_trio.fillna(np.nan, inplace=True)
            # 列の順序を調整
            full_df_trio = full_df_trio[['Date', 'JCD', 'Race', 'Boat1', 'Boat2', 'Boat3', 'Odds']]
            # 保存
            full_df_trio.to_csv(save_path_trio, index=False, encoding='utf-8-sig')
            print(f"Saved 三連複 data to {save_path_trio}")
        else:
            print(f"No 三連複 data found for date {date_str}")
    
    # セッションのクローズ
    session.close()

if __name__ == "__main__":
    try:
        start_time = time.time()  # 開始時間を記録
        save_trio_data()
        end_time = time.time()  # 終了時間を記録
        elapsed_time = end_time - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        sys.exit()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit()
