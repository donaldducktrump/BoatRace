import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# 任意の日程のレースについて直前情報やオッズの情報が記載されたURLを取得する
def get_url(date, place_cd, race_no, content):
    """
    content (str): ['odds3t', 'odds3f', 'odds2tf', 'beforeinfo', 'racelist']
    """
    url_t = 'https://www.boatrace.jp/owpc/pc/race/'
    ymd = str(pd.to_datetime(date)).split()[0].replace('-', '')
    jcd = f'0{place_cd}' if place_cd < 10 else str(place_cd)
    url = f'{url_t}{content}?rno={race_no}&jcd={jcd}&hd={ymd}'
    return url

# 直前情報のサイトからHTMLを取得し解析する
def get_beforeinfo(date, place_cd, race_no):
    url = get_url(date, place_cd, race_no, 'beforeinfo')
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve beforeinfo data. HTTP Status Code: {response.status_code}")
        return None
    
    html_content = response.text.replace('\n', '').replace('\t', '')
    soup = BeautifulSoup(html_content, 'lxml')

    # データ抽出のロジックはbeforeinfoページの構造に依存します
    # 以下は例として提供します。実際のページ構造に応じて修正が必要です。

    try:
        arr1 = [[tag.find_all('td')[4].text, tag.find_all('td')[5].text]
                for tag in soup.find_all(class_='is-fs12')]
        arr1 = [[v if v != '\xa0' else '' for v in row] for row in arr1]
        arr2 = [[tag.find(class_=f'table1_boatImage1Number').text,
                 tag.find(class_=f'table1_boatImage1Time').text]
                for tag in soup.find_all(class_='table1_boatImage1')]
        arr2 = [[v.replace('F', '-') for v in row] for row in arr2]
        arr2 = [row + [i] for i, row in enumerate(arr2, 1)]
        arr2 = pd.DataFrame(arr2).sort_values(by=[0]).values[:, 1:]

        weather_elements = soup.find_all(class_='weather1_bodyUnitLabelData')
        air_t, wind_v, water_t, wave_h = [elem.text for elem in weather_elements]
        weather = soup.find_all(class_='weather1_bodyUnitLabelTitle')[1].text
        wind_d_class = soup.select_one('p[class*="is-wind"]')['class'][1]
        wind_d = int(wind_d_class[7:])

        df = pd.DataFrame(np.concatenate([arr1, arr2], axis=1),
                          columns=['ET', 'tilt', 'EST', 'ESC'])\
            .replace('L', '1').astype(float)

        if len(df) < 6:
            return None
        data = pd.concat([
            pd.Series({'date': date, 'place_cd': place_cd, 'race_no': race_no}),
            pd.Series(df.values.T.reshape(-1),
                      index=[f'{col}_{i}' for col in df.columns for i in range(1, 7)]),
            pd.Series({
                'weather': weather,
                'air_t': float(air_t[:-1]),
                'wind_d': wind_d,
                'wind_v': float(wind_v[:-1]),
                'water_t': float(water_t[:-1]),
                'wave_h': float(wave_h[:-2])})])
        for i in range(1, 7):
            data[f'ESC_{i}'] = int(data[f'ESC_{i}'])
        return data
    except Exception as e:
        print(f"Error parsing beforeinfo data: {e}")
        return None

# racelistのサイトからHTMLを取得し解析する
def get_racelist(date, place_cd, race_no):
    """
    racelist コンテンツからレース情報を取得する関数
    Args:
        date (str): 'YYYY-MM-DD' 形式の日付
        place_cd (int): 場コード
        race_no (int): レース番号
    Returns:
        pd.Series: レース情報のデータ
    """
    url = get_url(date, place_cd, race_no, 'racelist')
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve racelist data. HTTP Status Code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')  # 'lxml' を 'html.parser' に変更

    try:
        # racelistページの構造に基づいてデータを抽出
        # 以下は仮の例です。実際のページ構造に合わせて修正が必要です。

        # 例: レース参加者のテーブルを取得
        table = soup.find('table', class_='race_list_table')
        if not table:
            print("Racelist table not found.")
            return None

        rows = table.find_all('tr')[1:]  # ヘッダー行を除く
        data_list = []
        for row in rows:
            cols = row.find_all('td')
            boat_no = cols[0].text.strip()
            driver = cols[1].text.strip()
            odds = cols[2].text.strip()
            # 他の必要な情報を追加
            data_list.append([boat_no, driver, odds])

        df = pd.DataFrame(data_list, columns=['BoatNo', 'Driver', 'Odds'])

        # 追加情報の抽出（例）
        # ここで必要なデータをさらに抽出し、DataFrameに追加します

        # 基本情報の作成
        basic_info = pd.Series({
            'date': date,
            'place_cd': place_cd,
            'race_no': race_no
        })

        # レース情報をフラットにする
        race_info = df.to_dict(orient='records')

        for i, race in enumerate(race_info, 1):
            basic_info[f'BoatNo_{i}'] = race['BoatNo']
            basic_info[f'Driver_{i}'] = race['Driver']
            basic_info[f'Odds_{i}'] = race['Odds']

        return basic_info
    except Exception as e:
        print(f"Error parsing racelist data: {e}")
        return None

# 実行例
if __name__ == "__main__":
    date = '2024-03-01'
    place_cd = 2
    race_no = 1

    # beforeinfoの取得
    bi = get_beforeinfo(date, place_cd, race_no)
    print("Before Info:")
    print(bi)

    # racelistの取得
    rl = get_racelist(date, place_cd, race_no)
    print("\nRacelist Info:")
    print(rl)
