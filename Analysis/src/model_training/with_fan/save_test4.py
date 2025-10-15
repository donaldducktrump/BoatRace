import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import sys
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 任意の日程のレースについて直前情報やオッズの情報が記載されたURLを取得する
def get_url(date, place_cd, race_no, content):
    """
    content (str): ['odds3t', 'odds3f', 'odds2tf', 'beforeinfo', 'racelist', 'oddstf']
    """
    url_t = 'https://www.boatrace.jp/owpc/pc/race/'
    ymd = pd.to_datetime(date).strftime('%Y%m%d')
    jcd = f'0{place_cd}' if place_cd < 10 else str(place_cd)
    url = f'{url_t}{content}?rno={race_no}&jcd={jcd}&hd={ymd}'
    return url

# セッションとリトライ戦略の設定
def requests_session():
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

# 直前情報のサイトからHTMLを取得し解析する
def get_beforeinfo(args, session):
    date, place_cd, race_no = args
    url_before = get_url(date, place_cd, race_no, 'beforeinfo')
    print(f"Accessing URL: {url_before}")  # デバッグ用
    headers = {'User-Agent': 'Mozilla/5.0'}
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    options = Options()
    options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")  # リモートデバッグのアドレスを指定
    options.use_chromium = True
    # options.add_argument('headless')  # ヘッドレスモードを有効にする
    # options.page_load_strategy = 'eager'  # ページが読み込まれる前に処理を続ける

    # Edgeのオプション設定
    path='msedgedriver_129.exe'
    service = Service(executable_path=path)
    # 既存のEdgeブラウザに接続
    driver = webdriver.Edge(service=service, options=options)
    
    try:

        # response_before = session.get(url_before, headers=headers, timeout=10)
        # if response_before.status_code != 200:
        #     print(f"Failed to retrieve beforeinfo data. HTTP Status Code: {response_before.status_code}")
        #     return None
        driver.get(url_before)
        soup_before = driver.page_source
        html_content_before = soup_before

        # html_content_before = response_before.text.replace('\n', '').replace('\t', '')
        soup_before = BeautifulSoup(html_content_before, 'lxml')

        # 既存のbeforeinfo解析ロジック（修正済み）
        boats_data = []

        # 各枠番（ボート）ごとにデータを収集
        for tbody in soup_before.find_all('tbody', class_='is-fs12'):
            trs = tbody.find_all('tr')
            if not trs:
                continue
            first_tr = trs[0]
            tds = first_tr.find_all('td')

            # ボート番号を取得
            if tds:
                boat_num_td = tds[0]
                boat_num = boat_num_td.text.strip()
                if boat_num.isdigit():
                    boat_num = int(boat_num)
                else:
                    continue
            else:
                continue

            # 初期化
            ET = None
            tilt = None
            EST = None
            ESC = None

            # ETとtiltの取得
            if len(tds) >= 6:
                ET_text = tds[4].text.strip()
                tilt_text = tds[5].text.strip()

                ET = ET_text if ET_text != '\xa0' and ET_text != '' else None
                tilt = tilt_text if tilt_text != '\xa0' and tilt_text != '' else None
            else:
                ET = None
                tilt = None

            # ESTとESCの取得
            boat_image_divs = soup_before.find_all(class_='table1_boatImage1')
            for div in boat_image_divs:
                num_tag = div.find(class_='table1_boatImage1Number')
                if num_tag and num_tag.text.strip() == str(boat_num):
                    time_tag = div.find(class_='table1_boatImage1Time')
                    if time_tag:
                        EST_text = time_tag.text.strip().replace('F', '-')
                        EST = EST_text if EST_text != '\xa0' and EST_text != '' else None
                    else:
                        EST = None
                    ESC = boat_num  # ボート番号をそのままESCとします
                    break
            else:
                EST = None
                ESC = None

            # データをリストに追加
            boats_data.append({
                'boat_num': boat_num,
                'ET': ET,
                'tilt': tilt,
                'EST': EST,
                'ESC': ESC
            })

        # DataFrameの作成
        df_before = pd.DataFrame(boats_data).set_index('boat_num').sort_index()

        # 列を適切な型に変換
        for col in ['ET', 'tilt', 'EST', 'ESC']:
            df_before[col] = pd.to_numeric(df_before[col], errors='coerce')

        if len(df_before) < 6:
            print("Dataframe length less than 6.")
            return None

        # 天候データの取得（省略）                                                                  
        weather_elements = soup_before.find_all(class_='weather1_bodyUnitLabelData')
        if len(weather_elements) < 4:
            print("Warning: Not enough weather elements found.")
            return None
        air_t, wind_v, water_t, wave_h = [elem.text.strip() for elem in weather_elements]
        weather_titles = soup_before.find_all(class_='weather1_bodyUnitLabelTitle')
        if len(weather_titles) < 2:
            print("Warning: Not enough weather title elements found.")
            return None
        weather = weather_titles[1].text.strip()
        wind_d_class = soup_before.select_one('p[class*="is-wind"]')
        if not wind_d_class or len(wind_d_class['class']) < 2:
            print("Warning: Wind direction class not found.")
            return None
        wind_d = int(wind_d_class['class'][1][7:])

        # dataシリーズの作成
        data = pd.concat([
            pd.Series({'date': date, 'place_cd': place_cd, 'race_no': race_no}),
            pd.Series(df_before.values.T.reshape(-1),
                    index=[f'{col}_{i}' for i in df_before.index for col in df_before.columns]),
            pd.Series({
                'weather': weather,
                'air_t': float(air_t[:-1]),
                'wind_d': wind_d,
                'wind_v': float(wind_v[:-1]),
                'water_t': float(water_t[:-1]),
                'wave_h': float(wave_h[:-2])})])

        # ESC列を整数に変換（可能なら）
        for i in df_before.index:
            if pd.notnull(data[f'ESC_{i}']):
                data[f'ESC_{i}'] = int(data[f'ESC_{i}'])

        print(f"Retrieved beforeinfo data for date {date}, place_cd {place_cd}, race_no {race_no}")

        # ここからオッズ情報の取得と解析
        url_odds = get_url(date, place_cd, race_no, 'oddstf')  # 'oddstf'を使用
        print(f"Accessing Odds URL: {url_odds}")  # デバッグ用

        try:
            # response_odds = session.get(url_odds, headers=headers, timeout=10)
            # if response_odds.status_code != 200:
            #     print(f"Failed to retrieve odds data. HTTP Status Code: {response_odds.status_code}")
            #     return None
            driver.get(url_odds)
            soup_odds = driver.page_source
            html_content_odds = soup_odds
            # html_content_odds = response_odds.text.replace('\n', '').replace('\t', '')
            soup_odds = BeautifulSoup(html_content_odds, 'lxml')
            # driver.quit()

            # 単勝オッズの解析
            win_odds = {}
            h3_win = soup_odds.find('span', string='単勝オッズ')
            if h3_win:
                table_win = h3_win.find_parent('div', class_='title7').find_next_sibling('div', class_='table1').find('table')
                if table_win:
                    for tr in table_win.find_all('tr'):
                        tds = tr.find_all('td')
                        if len(tds) >= 3:
                            boat_num = tds[0].text.strip()
                            odds = tds[2].text.strip()
                            if boat_num.isdigit():
                                if odds == '欠場':
                                    win_odds[int(boat_num)] = None
                                else:
                                    try:
                                        win_odds[int(boat_num)] = float(odds)
                                    except:
                                        win_odds[int(boat_num)] = 0.0
                else:
                    print("Warning: 単勝オッズのテーブルが見つかりませんでした。")
            else:
                print("Warning: '単勝オッズ' の見出しが見つかりませんでした。")

            # 複勝オッズの解析
            place_odds = {}
            h3_place = soup_odds.find('span', string='複勝オッズ')
            if h3_place:
                table_place = h3_place.find_parent('div', class_='title7').find_next_sibling('div', class_='table1').find('table')
                if table_place:
                    for tr in table_place.find_all('tr'):
                        tds = tr.find_all('td')
                        if len(tds) >= 3:
                            boat_num = tds[0].text.strip()
                            odds = tds[2].text.strip()
                            if boat_num.isdigit():
                                if odds == '欠場':
                                    win_odds[int(boat_num)] = None
                                else:                                
                                    try:
                                        if '-' in odds:
                                            parts = odds.split('-')
                                            place_odds_val = (float(parts[0]) + float(parts[1])) / 2
                                        else:
                                            place_odds_val = float(odds)
                                        place_odds[int(boat_num)] = place_odds_val
                                    except:
                                        place_odds[int(boat_num)] = 0.0
                else:
                    print("Warning: 複勝オッズのテーブルが見つかりませんでした。")
            else:
                print("Warning: '複勝オッズ' の見出しが見つかりませんでした。")

            # 単勝オッズと複勝オッズをデータに追加
            for i in range(1, 7):
                data[f'win_odds_{i}'] = win_odds.get(i, 0.0)
                data[f'place_odds_{i}'] = place_odds.get(i, 0.0)

            print(f"Retrieved odds data for date {date}, place_cd {place_cd}, race_no {race_no}")
                
            # ここから3連単オッズ情報の取得と解析
            url_trifecta_odds = get_url(date, place_cd, race_no, 'odds3t')  # 'oddstf'を使用
            print(f"Accessing Odds URL: {url_trifecta_odds}")  # デバッグ用

            try:
                # response_odds = session.get(url_odds, headers=headers, timeout=10)
                # if response_odds.status_code != 200:
                #     print(f"Failed to retrieve odds data. HTTP Status Code: {response_odds.status_code}")
                #     return None
                driver.get(url_trifecta_odds)
                soup_odds = driver.page_source
                html_content_odds = soup_odds
                # html_content_odds = response_odds.text.replace('\n', '').replace('\t', '')
                soup_odds = BeautifulSoup(html_content_odds, 'lxml')
                driver.quit()

                trifecta_data = []
                
                # テーブルのtbodyを取得
                tbody = soup_odds.find('tbody', class_='is-p3-0')
                if not tbody:
                    print("Warning: <tbody class='is-p3-0'> が見つかりませんでした。")
                    return trifecta_data
                
                rows = tbody.find_all('tr')
                
                boat1 = None  # 現在の1着艇番号
                boat2 = None  # 現在の2着艇番号
                
                for row_index, row in enumerate(rows, start=1):
                    cols = row.find_all('td')
                    if not cols:
                        continue
                    
                    # 行ごとの処理
                    if row_index % 4 == 1:
                        boat2_list = []     
                        # x = 1 mod 4 の<tr>
                        # 各セットは3つの<td>で構成される: Boat2, Boat3, Odds
                        num_sets = len(cols) // 3
                        for set_index in range(num_sets):
                            try:
                                base = set_index * 3
                                boat2_num = cols[base].get_text(strip=True)
                                boat3_num = cols[base + 1].get_text(strip=True)
                                odds_text = cols[base + 2].get_text(strip=True)
                                
                                                            # 2着艇番号を格納するリスト
                                boat2_list.append(boat2_num)                        # 2着艇番号をリストに追加

                                # Boat1の取得方法: 行ごとの1着艇番号
                                # ここではセットインデックス+1を仮定していますが、実際のHTML構造に基づいて調整してください
                                boat1_num =  set_index + 1                         # 1着艇番号を取得
                                boat1 = boat1_num
                                
                                # オッズの取得
                                try:
                                    odds = float(odds_text)
                                except ValueError:
                                    odds = np.nan  # オッズが取得できない場合はNaNとする
                                
                                trifecta_data.append({
                                    'Boat1': boat1,
                                    'Boat2': boat2_num,
                                    'Boat3': boat3_num,
                                    'Odds': odds
                                })
                                
                            except IndexError:
                                print(f"Warning: セット {set_index +1} に十分な<td>がありません。スキップします。")
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
                            

                                boat1_num =  set_index + 1                         # 1着艇番号を取得
                                boat1 = boat1_num

                                if set_index < len(boat2_list):
                                    boat2 = boat2_list[set_index]                      # 2着艇番号を取得
                                else:
                                    print(f"Warning: boat2_listに十分なデータがありません。セット {set_index +1} をスキップします。")
                                    continue

                                if not boat2:
                                    print("Warning: 2着艇番号が未設定です。スキップします。")
                                    continue
                                
                                # オッズの取得
                                try:
                                    odds = float(odds_text)
                                except ValueError:
                                    odds = np.nan  # オッズが取得できない場合はNaNとする
                                
                                trifecta_data.append({
                                    'Boat1': boat1,
                                    'Boat2': boat2,
                                    'Boat3': boat3_num,
                                    'Odds': odds
                                })

                            except IndexError:
                                print(f"Warning: セット {set_index +1} に十分な<td>がありません。スキップします。")
                                continue
                print(f"Retrieved odds data for date {date}, place_cd {place_cd}, race_no {race_no}")
                
                return data, trifecta_data

            except requests.exceptions.ReadTimeout:
                print(f"ReadTimeout occurred while accessing Odds URL: {url_trifecta_odds}")
                return None
            except Exception as e:
                print(f"Error parsing odds data: {e}")
                traceback.print_exc()
                return None
            
        except requests.exceptions.ReadTimeout:
            print(f"ReadTimeout occurred while accessing Odds URL: {url_odds}")
            return None
        except Exception as e:
            print(f"Error parsing odds data: {e}")
            traceback.print_exc()
            return None

    except requests.exceptions.ReadTimeout:
        print(f"ReadTimeout occurred while accessing BeforeInfo URL: {url_before}")
        return None
    except Exception as e:
        print(f"Error parsing beforeinfo or odds data: {e}")
        traceback.print_exc()
        return None

def save_beforeinfo_data():
    """
    指定された期間のbeforeinfoデータとオッズ情報を取得して保存します。
    1日の全てのレースを並列で処理します。
    """
    # 2024年3月1日 ～ 2024年10月8日 の期間
    start_date = pd.to_datetime('2024-10-22')
    end_date = pd.to_datetime('2024-10-22')
    date_range = pd.date_range(start_date, end_date)

    # データを保存するフォルダを作成
    base_dir = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\before_data2'
    os.makedirs(base_dir, exist_ok=True)

    # セッションの作成
    session = requests_session()

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        month_str = date.strftime('%y%m')
        save_dir = os.path.join(base_dir, month_str)
        os.makedirs(save_dir, exist_ok=True)

        # 既に保存済みの場合はスキップ
        save_file = os.path.join(save_dir, f'beforeinfo_{date.strftime("%y%m%d")}.txt')
        if os.path.exists(save_file):
            print(f"Already exists: {save_file}")
            continue

        print(f"Processing date {date_str}")
        all_data = []
        error_place_cds = set()

        # 全ての (place_cd, race_no) の組み合わせを作成
        args_list = []
        for place_cd in range(1, 25):  # 場コードは1から24まで
            for race_no in range(1, 13):  # レース番号は1から12まで
                args_list.append((date_str, place_cd, race_no))

        # 並列処理の開始
        try:
            with ThreadPoolExecutor(max_workers=300) as executor:  # max_workers を100に制限
                # 各タスクにセッションを渡す
                future_to_args = {executor.submit(get_beforeinfo, args, session): args for args in args_list}
                for future in as_completed(future_to_args):
                    args = future_to_args[future]
                    date_str, place_cd, race_no = args
                    try:
                        result = future.result()
                        if isinstance(result, str) and result == 'No data found in arr1':
                            print(f"Skipping place_cd {place_cd} for date {date_str} due to missing data.")
                            error_place_cds.add(place_cd)
                        elif result is None:
                            print(f"No data retrieved for date {date_str}, place_cd {place_cd}, race_no {race_no}")
                            error_place_cds.add(place_cd)
                        else:
                            if place_cd not in error_place_cds:
                                all_data.append(result)
                                print(f"Result obtained for date {date_str}, place_cd {place_cd}, race_no {race_no}")
                    except Exception as exc:
                        print(f"Exception occurred for date {date_str}, place_cd {place_cd}, race_no {race_no}: {exc}")
                        error_place_cds.add(place_cd)
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving current data...")
            executor.shutdown(wait=False, cancel_futures=True)
            # 現在のデータを保存
            if all_data:
                # エラーが発生した place_cd のデータを除外
                all_data = [data for data in all_data if data['place_cd'] not in error_place_cds]
                with open(save_file, 'w', encoding='utf-8') as f:
                    for data in all_data:
                        f.write(data.to_string())
                        f.write('\n\n')
                print(f"Saved partial data: {save_file}")
            print("Exiting.")
            sys.exit()

        # エラーが発生した place_cd のデータを除外
        all_data = [data for data in all_data if data['place_cd'] not in error_place_cds]

        if all_data:
            # 日付ごとにデータを保存
            with open(save_file, 'w', encoding='utf-8') as f:
                for data in all_data:
                    f.write(data.to_string())
                    f.write('\n\n')
            print(f"Saved: {save_file}")
        else:
            print(f"No data for {date_str}")

    # セッションのクローズ
    session.close()

if __name__ == "__main__":
    start_time = time.time()  # 開始時間を記録
    save_beforeinfo_data()
    end_time = time.time()  # 終了時間を記録
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
