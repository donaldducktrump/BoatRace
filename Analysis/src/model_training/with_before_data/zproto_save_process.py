import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import sys
import time

# 任意の日程のレースについて直前情報やオッズの情報が記載されたURLを取得する
def get_url(date, place_cd, race_no, content):
    """
    content (str): ['odds3t', 'odds3f', 'odds2tf', 'beforeinfo', 'racelist']
    """
    url_t = 'https://www.boatrace.jp/owpc/pc/race/'
    ymd = pd.to_datetime(date).strftime('%Y%m%d')
    jcd = f'0{place_cd}' if place_cd < 10 else str(place_cd)
    url = f'{url_t}{content}?rno={race_no}&jcd={jcd}&hd={ymd}'
    return url

# 直前情報のサイトからHTMLを取得し解析する
def get_beforeinfo(args):
    date, place_cd, race_no = args
    url = get_url(date, place_cd, race_no, 'beforeinfo')
    print(f"Accessing URL: {url}")  # デバッグ用
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve beforeinfo data. HTTP Status Code: {response.status_code}")
        return None

    html_content = response.text.replace('\n', '').replace('\t', '')
    soup = BeautifulSoup(html_content, 'lxml')

    # データ抽出のロジックはbeforeinfoページの構造に依存します
    # 以下は例として提供します。実際のページ構造に応じて修正が必要です。

    try:
        arr1 = [[tag.find_all('td')[4].text.strip(), tag.find_all('td')[5].text.strip()]
                for tag in soup.find_all(class_='is-fs12')]
        arr1 = [[v if v != '\xa0' else '' for v in row] for row in arr1]
        if not arr1 or any(not row[0] for row in arr1):
            print("No data found in arr1 (ET and tilt).")
            return 'No data found in arr1'

        arr2 = [[tag.find(class_='table1_boatImage1Number').text.strip(),
                 tag.find(class_='table1_boatImage1Time').text.strip()]
                for tag in soup.find_all(class_='table1_boatImage1')]
        arr2 = [[v.replace('F', '-') for v in row] for row in arr2]
        arr2 = [row + [i] for i, row in enumerate(arr2, 1)]
        arr2 = pd.DataFrame(arr2).sort_values(by=[0]).values[:, 1:]

        weather_elements = soup.find_all(class_='weather1_bodyUnitLabelData')
        if len(weather_elements) < 4:
            print("Warning: Not enough weather elements found.")
            return None
        air_t, wind_v, water_t, wave_h = [elem.text.strip() for elem in weather_elements]
        weather_titles = soup.find_all(class_='weather1_bodyUnitLabelTitle')
        if len(weather_titles) < 2:
            print("Warning: Not enough weather title elements found.")
            return None
        weather = weather_titles[1].text.strip()
        wind_d_class = soup.select_one('p[class*="is-wind"]')
        if not wind_d_class or len(wind_d_class['class']) < 2:
            print("Warning: Wind direction class not found.")
            return None
        wind_d = int(wind_d_class['class'][1][7:])

        df = pd.DataFrame(np.concatenate([arr1, arr2], axis=1),
                          columns=['ET', 'tilt', 'EST', 'ESC'])\
            .replace('L', '1').astype(float)

        if len(df) < 6:
            print("Dataframe length less than 6.")
            return None
        data = pd.concat([
            pd.Series({'date': date, 'place_cd': place_cd, 'race_no': race_no}),
            pd.Series(df.values.T.reshape(-1),
                      index=[f'{col}_{i}' for i in range(1, 7) for col in df.columns]),
            pd.Series({
                'weather': weather,
                'air_t': float(air_t[:-1]),
                'wind_d': wind_d,
                'wind_v': float(wind_v[:-1]),
                'water_t': float(water_t[:-1]),
                'wave_h': float(wave_h[:-2])})])
        for i in range(1, 7):
            data[f'ESC_{i}'] = int(data[f'ESC_{i}'])
        print(f"Retrieved data for date {date}, place_cd {place_cd}, race_no {race_no}")
        return data
    except Exception as e:
        print(f"Error parsing beforeinfo data: {e}")
        traceback.print_exc()
        return None

def save_beforeinfo_data():
    """
    指定された期間のbeforeinfoデータを取得して保存します。
    各place_cdで、データがないレースが一つでもあれば、そのplace_cdのデータはスキップします。
    """
    # 2024年3月1日 ～ 2024年10月8日 の期間
    start_date = pd.to_datetime('2024-03-01')
    end_date = pd.to_datetime('2024-03-01')
    date_range = pd.date_range(start_date, end_date)

    # データを保存するフォルダを作成
    base_dir = r'C:\Users\kazut\OneDrive\Documents\BoatRace\ML_pred\before_data'
    os.makedirs(base_dir, exist_ok=True)

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        month_str = date.strftime('%y%m')
        save_dir = os.path.join(base_dir, month_str)
        os.makedirs(save_dir, exist_ok=True)

        # 既に保存済みの場合はスキップ
        save_file = os.path.join(save_dir, f'beforeinfo_{date.strftime("%Y%m%d")}.txt')
        if os.path.exists(save_file):
            print(f"Already exists: {save_file}")
            continue

        all_data = []
        for place_cd in range(1, 25):  # 場コードは1から24まで
            print(f"Processing date {date_str}, place_cd {place_cd}")
            place_data = []
            error_occurred = False  # エラーが発生したかどうかのフラグ
            args_list = [(date_str, place_cd, race_no) for race_no in range(1, 13)]
            try:
                with ProcessPoolExecutor(max_workers=61) as executor:
                    future_to_args = {executor.submit(get_beforeinfo, args): args for args in args_list}
                    for future in as_completed(future_to_args):
                        args = future_to_args[future]
                        date_str, place_cd, race_no = args
                        try:
                            result = future.result()
                            if isinstance(result, str) and result == 'No data found in arr1':
                                print(f"Skipping place_cd {place_cd} for date {date_str} due to missing data.")
                                error_occurred = True
                                break  # この place_cd の処理を中断
                            elif result is None:
                                print(f"No data retrieved for date {date_str}, place_cd {place_cd}, race_no {race_no}")
                                error_occurred = True
                                break  # この place_cd の処理を中断
                            else:
                                place_data.append(result)
                                print(f"Result obtained for date {date_str}, place_cd {place_cd}, race_no {race_no}")
                        except Exception as exc:
                            print(f"Exception occurred for date {date_str}, place_cd {place_cd}, race_no {race_no}: {exc}")
                            error_occurred = True
                            break  # この place_cd の処理を中断
                if error_occurred:
                    print(f"Skipping place_cd {place_cd} for date {date_str} due to errors.")
                    continue  # 次の place_cd へ
                else:
                    all_data.extend(place_data)
            except KeyboardInterrupt:
                print("\nProcess interrupted by user. Saving current data...")
                executor.shutdown(wait=False, cancel_futures=True)
                # 現在のデータを保存
                if all_data:
                    with open(save_file, 'w', encoding='utf-8') as f:
                        for data in all_data:
                            f.write(data.to_string())
                            f.write('\n\n')
                    print(f"Saved partial data: {save_file}")
                print("Exiting.")
                sys.exit()

        if all_data:
            # 日付ごとにデータを保存
            with open(save_file, 'w', encoding='utf-8') as f:
                for data in all_data:
                    f.write(data.to_string())
                    f.write('\n\n')
            print(f"Saved: {save_file}")
        else:
            print(f"No data for {date_str}")

if __name__ == "__main__":
    # Windows での ProcessPoolExecutor の動作を保証するための記述
    import multiprocessing
    multiprocessing.freeze_support()
    start_time = time.time()  # 開始時間を記録
    save_beforeinfo_data()
    end_time = time.time()  # 終了時間を記録
    elapsed_time = end_time - start_time
    print(f"Total execution time with ThreadPoolExecutor: {elapsed_time:.2f} seconds")
