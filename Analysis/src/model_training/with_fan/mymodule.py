# mymodule.py

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import requests
from bs4 import BeautifulSoup

def load_b_file(file_path):
    """
    Bファイルからデータを読み込み、指定された項目を抽出します。
    """
    with open(file_path, encoding='shift-jis') as f:
        data = f.readlines()
    # print(data)
    # # 会場コードの取得
    # place_code_line = [line for line in data if 'BBGN' in line]
    # if place_code_line:
    #     place_code_match = re.match(r'(\d{2})BBGN', place_code_line[0].strip())
    #     if place_code_match:
    #         place_code = place_code_match.group(1)
    #     else:
    #         print("場コードを取得できませんでした。")
    #         place_code = '00'
    # else:
    #     print("場コードを取得できませんでした。")
    #     place_code = '00'

    # デバッグ: 取得した会場コードを表示
    # print(place_code_line)
    # print(f"会場コード: {place_code}")
    
    # レース情報の取得
    races = []
    race_number = None
    date = None

    # 選手情報を格納するためのリスト
    racer_data = []

    for idx, line in enumerate(data):
        line = line.replace('\u3000', '').strip()
        # 日付の取得
        if '第' in line and '日' in line and 'ボートレース' in line:
            date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', line)
            if date_match:
                year, month, day = date_match.groups()
                date = pd.to_datetime(f"{year}-{month}-{day}")
                # print(f"日付: {date}")
        # 会場コードの取得
        if 'BBGN' in line:
            place_code_match = re.match(r'(\d{2})BBGN', line.strip())
            if place_code_match:
                place_code = place_code_match.group(1)
            else:
                print("場コードを取得できませんでした。")
                place_code = '00'
            # print(f"会場コード: {place_code}")
        # レース番号の取得
        if re.match(r'^(\d{1,2})Ｒ', line):
            race_number_match = re.match(r'^(\d{1,2})Ｒ', line)
            if race_number_match:
                full_width_race_number = race_number_match.group(1)
                race_number = full_width_race_number.translate(str.maketrans('０１２３４５６７８９', '0123456789')).zfill(2)
            # print(f"レース番号取得: {race_number}")  # デバッグ用
        # 選手情報の取得（行が選手情報であるかを確認）
        if re.match(r'^\d\s+\d{4}', line):
            racer_data.append(line.strip())  # 選手情報行を追加
            # print(f"取得された選手情報: {line.strip()}")  # デバッグ用出力
            # print(racer_data)

        # 選手情報が6行たまったら、それを解析してレコードに変換
        if len(racer_data) == 6:
            for racer in racer_data:
                # まず全角スペースと改行を削除し、連続する半角スペースを1つに置き換え
                racer_line = racer.replace('\u3000', '').strip()
                racer_line = re.sub(r'\s+', ' ', racer_line)  # 連続するスペースを1つにする
                # print(f"整形後の選手情報: {racer_line}")  # デバッグ用に出力

# マッチングを段階的に行う

# Step 1: 艇番, 選手登番, 選手名, 年齢, 支部, 体重を個別にマッチング

                # Step 1: 艇番と選手登番
                boat_and_racer_pattern = r'^(\d)\s+(\d{4})'
                boat_and_racer_match = re.match(boat_and_racer_pattern, racer_line)
                if boat_and_racer_match:
                    艇番, 選手登番 = boat_and_racer_match.groups()
                    # print(f"Step 1: 艇番={艇番}, 選手登番={選手登番}")
                else:
                    print(f"Step 1: マッチ失敗 - {racer_line}")
                    continue
                
                # # Step 2: 選手名（全角4文字分を抽出）
                # # 正規表現を使って全角4文字分（選手名）を抽出
                # name_pattern = r'^\d\s+\d{4}([^\x00-\x7F\u3000]{2,4})'
                # name_match = re.search(name_pattern, racer_line)
                # if name_match:
                #     選手名 = name_match.group(1).strip()
                #     # print(f"Step 2: 選手名={選手名}")
                # else:
                #     print(f"Step 2: マッチ失敗 - {racer_line}")
                #     continue

                # Step 3: 年齢, 支部、体重、級別
                age_branch_weight_rank_pattern = r'^\d\s+\d{4}[^\x00-\x7F\u3000]{2,4}(\d{2})([^\x00-\x7F]{2,3})(\d{2})([AB]\d)'
                age_branch_weight_rank_match = re.search(age_branch_weight_rank_pattern, racer_line)
                if age_branch_weight_rank_match:
                    年齢, 支部, 体重, 級別 = age_branch_weight_rank_match.groups()  # 選手名は先に抽出済みのため除外
                    # print(f"Step 3: 年齢={年齢}, 支部={支部}, 体重={体重}, 級別={級別}")
                else:
                    print(f"Step 3: マッチ失敗 - {racer_line}")
                    continue

                # Step 4: 全国勝率, 全国2連率, 当地勝率, 当地2連率をまとめて抽出
                win_rates_pattern = r'^\d\s+\d{4}[^\x00-\x7F\u3000]{2,4}\d{2}[^\x00-\x7F]{2,3}\d{2}[AB]\d\s*(\d{1,3}\.\d{2})\s*(\d{1,3}\.\d{2})\s*(\d{1,3}\.\d{2})\s*(\d{1,3}\.\d{2})'
                win_rates_match = re.search(win_rates_pattern, racer_line)
                if win_rates_match:
                    全国勝率, 全国2連率, 当地勝率, 当地2連率 = win_rates_match.groups()
                    # print(f"Step 4: 全国勝率={全国勝率}, 全国2連率={全国2連率}, 当地勝率={当地勝率}, 当地2連率={当地2連率}")
                else:
                    print(f"Step 4: マッチ失敗 - {racer_line}")
                    continue

                # Step 5: モーターNo, モーター2連率, ボートNo, ボート2連率, 今季成績
                third_pattern = r'^\d\s+\d{4}[^\x00-\x7F\u3000]{2,4}\d{2}[^\x00-\x7F]{2,3}\d{2}[AB]\d\s*\d{1,3}\.\d{2}\s*\d{1,3}\.\d{2}\s*\d{1,3}\.\d{2}\s*\d{1,3}\.\d{2}\s*(\d{1,3})\s*(\d{1,3}\.\d{2})\s*(\d{1,3})\s*(\d{1,3}\.\d{2})'
                third_match = re.search(third_pattern, racer_line)
                if third_match:
                    モーターNo, モーター2連率, ボートNo, ボート2連率 = third_match.groups()
                    # print(f"Step 5: モーターNo={モーターNo}, モーター2連率={モーター2連率}, ボートNo={ボートNo}, ボート2連率={ボート2連率}")
                else:
                    print(f"Step 5: マッチ失敗 - {racer_line}")
                    continue

                # Step 6: 今季成績
                fourth_pattern = r'^\d\s+\d{4}[^\x00-\x7F\u3000]{2,4}\d{2}[^\x00-\x7F]{2,3}\d{2}[AB]\d\s*\d{1,3}\.\d{2}\s*\d{1,3}\.\d{2}\s*\d{1,3}\.\d{2}\s*\d{1,3}\.\d{2}\s*(\d{1,3})\s*(\d{1,3}\.\d{2})\s*(\d{1,3})\s*(\d{1,3}\.\d{2})\s+([0-9\s+]{1,14})'
                fourth_match = re.search(fourth_pattern, racer_line)
                if place_code == 23:
                    if fourth_match:
                        今季成績 = fourth_match.group(1).strip()
                        print(f"Step 6: 今季成績={今季成績}")
                        if not 今季成績:
                            今季成績 = "0"  # 今季成績が空の場合は0を設定
                    else:
                        print(f"Step 6: マッチ失敗 - {racer_line}")

                # レースID作成などの処理
                レースID = f"{date.strftime('%Y%m%d')}{place_code}{race_number}"
                record = {
                    '会場': place_code,
                    'レースID': レースID,
                    '艇番': int(艇番),
                    '選手登番': 選手登番.strip(),
                    # '選手名': 選手名.strip(),
                    '年齢': int(年齢),
                    '支部': 支部.strip(),
                    '体重': float(体重),
                    '級別': 級別.strip(),
                    '全国勝率': float(全国勝率),
                    '全国2連率': float(全国2連率),
                    '当地勝率': float(当地勝率),
                    '当地2連率': float(当地2連率),
                    'モーターNo': int(モーターNo),
                    'モーター2連率': float(モーター2連率),
                    'ボートNo': int(ボートNo),
                    'ボート2連率': float(ボート2連率),
                    # '今季成績': int(今季成績)
                }
                races.append(record)

            # 6行の選手情報を処理した後、リストをクリア
            racer_data = []

    # デバッグ終了後、DataFrameに変換
    b_data = pd.DataFrame(races)
    # print("b_data columns:", b_data.columns.tolist())
    # print(b_data.head())
    return b_data

def load_k_file(file_path):
    """
    Kファイルからデータを読み込み、指定された項目を抽出します。
    """
    with open(file_path, encoding='shift-jis') as f:
        data = f.readlines()

        
    # レース情報の取得
    races = []
    race_number = None
    date = None

    # 選手情報を格納するためのリスト
    racer_data = []

    # レース情報の取得
    results = []
    odds_list = []
    race_number = None
    天候 = None
    風向 = None
    風量 = None
    波 = None

    for idx, line in enumerate(data):
        line = line.replace('\u3000', '').rstrip('\n')
        # 日付の取得
        if '第' in line and '日' in line and 'ボートレース' in line:
            date_match = re.search(r'(\d{4})/\s*(\d{1,2})/\s*(\d{1,2})', line)
            if date_match:
                year, month, day = date_match.groups()
                date = pd.to_datetime(f"{year}-{month}-{day}")
                # print(f"日付: {date}")
        # 会場コードの取得
        elif 'KBGN' in line:
            place_code_match = re.match(r'(\d{2})KBGN', line.strip())
            if place_code_match:
                place_code = place_code_match.group(1)
            else:
                print("場コードを取得できませんでした。")
                place_code = '00'
            # print(f"会場コード: {place_code}")
        # レース番号と天候等の取得
        elif re.match(r'^\s*\d{1,2}R', line) and not re.search(r'\d-\d-\d', line):
            race_number_match = re.match(r'^\s*(\d{1,2})R', line)
            if race_number_match:
                race_number = race_number_match.group(1).zfill(2)
                # print(f"レース番号取得: {race_number}")  # デバッグ用
            # 天候等の取得
            weather_match = re.search(r'H1800m\s+(.+)\s+風\s+(.+)\s+(\d+)m\s+波\s+(\d+)cm', line)
            if weather_match:
                天候 = weather_match.group(1).strip()
                風向 = weather_match.group(2).strip()
                風量 = int(weather_match.group(3))
                波 = int(weather_match.group(4))
        elif '単勝' in line:
            odds_match = re.search(r'単勝\s+(\d)\s+(\d+)', line)
            if odds_match:
                winner_boat = odds_match.group(1)  # 1着艇番
                winner_odds = int(odds_match.group(2)) / 100  # 1着オッズ
                if race_number and date and place_code:
                    レースID = f"{date.strftime('%Y%m%d')}{place_code}{race_number}"
                    
                    # 単勝データを作成
                    new_tansho_data = {
                        'レースID': レースID,
                        '単勝結果': winner_boat,
                        '単勝オッズ': winner_odds
                    }
                    
                    # 既存のレースIDが存在するか確認
                    existing_entry = next((item for item in odds_list if item['レースID'] == レースID), None)
                    
                    if existing_entry:
                        # 既存エントリに単勝データを追加または更新
                        existing_entry['単勝結果'] = new_tansho_data.get('単勝結果', existing_entry.get('単勝結果'))
                        existing_entry['単勝オッズ'] = new_tansho_data.get('単勝オッズ', existing_entry.get('単勝オッズ'))
                    else:
                        # レースIDが存在しない場合、新しいエントリとして追加
                        odds_list.append(new_tansho_data)
        elif '複勝' in line:
            odds_match = re.search(r'複勝\s+(\d)\s+(\d+)\s+(\d)\s+(\d+)', line)
            if odds_match:
                first_boat = odds_match.group(1)  # 1着艇番
                second_boat = odds_match.group(3)
                first_odds = int(odds_match.group(2)) / 100  # 1着オッズ
                second_odds = int(odds_match.group(4)) / 100
                if race_number and date and place_code:
                    レースID = f"{date.strftime('%Y%m%d')}{place_code}{race_number}"
                    
                    # 複勝データを作成
                    new_tansho_data = {
                        'レースID': レースID,
                        '複勝結果1': first_boat,
                        '複勝結果2': second_boat,
                        '複勝オッズ1': first_odds,
                        '複勝オッズ2': second_odds
                    }
                    
                    # 既存のレースIDが存在するか確認
                    existing_entry = next((item for item in odds_list if item['レースID'] == レースID), None)
                    
                    if existing_entry:
                        # 既存エントリに単勝データを追加または更新
                        existing_entry['複勝結果1'] = new_tansho_data.get('複勝結果1', existing_entry.get('複勝結果1'))
                        existing_entry['複勝結果2'] = new_tansho_data.get('複勝結果2', existing_entry.get('複勝結果2'))
                        existing_entry['複勝オッズ1'] = new_tansho_data.get('複勝オッズ1', existing_entry.get('複勝オッズ1'))
                        existing_entry['複勝オッズ2'] = new_tansho_data.get('複勝オッズ2', existing_entry.get('複勝オッズ2'))
                    else:
                        # レースIDが存在しない場合、新しいエントリとして追加
                        odds_list.append(new_tansho_data)    
        elif '拡連複' in line:
            i=1
            quinella_data = data[idx:idx+3]
            # print(quinella_data)
            for line in quinella_data:
                line = line.strip()
                odds_match = re.search(r'(\d-\d)\s+(\d+)\s+人気\s+\d', line)
                # print(odds_match)
                if odds_match:
                    result= odds_match.group(1)
                    odds = int(odds_match.group(2)) / 100
                    if race_number and date and place_code:
                        レースID = f"{date.strftime('%Y%m%d')}{place_code}{race_number}"
                        # 拡連複データを作成
                        new_kaku_data = {
                            'レースID': レースID,
                            f'拡連複結果{i}': result,
                            f'拡連複オッズ{i}': odds
                        }
                        # 既存のレースIDが存在するか確認
                        existing_entry = next((item for item in odds_list if item['レースID'] == レースID), None)
                        if existing_entry:
                            # 既存エントリに単勝データを追加または更新
                            existing_entry[f'拡連複結果{i}'] = new_kaku_data.get(f'拡連複結果{i}', existing_entry.get(f'拡連複結果{i}'))
                            existing_entry[f'拡連複オッズ{i}'] = new_kaku_data.get(f'拡連複オッズ{i}', existing_entry.get(f'拡連複オッズ{i}'))
                        else:
                            # レースIDが存在しない場合、新しいエントリとして追加
                            odds_list.append(new_kaku_data)
                # # デバッグ: odds_listの内容を確認
                # for entry in odds_list:
                #     print(entry)  # 各エントリを出力し、レースIDとオッズデータの有無を確認

                i+=1                               
        # 選手情報の取得（分割してマッチング）
        elif re.match(r'^\s*\d{2}\s', line):

            # 着、艇番、選手登番のマッチング
            step1_pattern = r'^\s*(\d{2})\s+(\d)\s+(\d{4})'
            step1_match = re.match(step1_pattern, line)
            if step1_match:
                着, 艇番, 選手登番 = step1_match.groups()
                # print(f"ステップ1: 着={着}, 艇番={艇番}, 選手登番={選手登番}")
            else:
                print("ステップ1でマッチしませんでした")
                continue  # ステップ1がマッチしない場合は次の行へ

            # 選手名のマッチング
            step2_pattern = r'^\s*\d{2}\s+\d\s+\d{4}\s+([^\x00-\x7F\u3000]{2,5})'
            step2_match = re.search(step2_pattern, line)
            if step2_match:
                選手名 = step2_match.group(1).strip()
                # print(f"ステップ2: 選手名={選手名}")
            else:
                print("ステップ2で選手名がマッチしませんでした")
                continue

            # 展示タイム、進入、スタートタイミング、レースタイムのマッチング
            step3_pattern = r'\s+\d{1,3}\s+\d{1,3}\s+(\d+\.\d{2})\s+(\d)\s+(\d+\.\d{2})\s+'
            step3_match = re.search(step3_pattern, line)
            if step3_match:
                展示タイム, 進入, スタートタイミング = step3_match.groups()
                # print(f"ステップ3: 展示タイム={展示タイム}, 進入={進入}, スタートタイミング={スタートタイミング}")
            else:
                print("ステップ3で展示タイムなどがマッチしませんでした")
                print(f"選手情報行: {line}")
                continue

            # レースタイムの変換パターン
            race_time_pattern = r'\s+\d{1,3}\s+\d{1,3}\s+\d+\.\d{2}\s+\d\s+\d+\.\d{2}\s+(\d)?\.?(\d{0,2})?\.?(\d{0,1})?'
            race_time_match = re.search(race_time_pattern, line)

            if race_time_match:
                # マッチした部分をそれぞれ取得。マッチしなかった部分は None になる
                minutes = race_time_match.group(1) if race_time_match.group(1) else '0'
                seconds = race_time_match.group(2) if race_time_match.group(2) else '00'
                fraction = race_time_match.group(3) if race_time_match.group(3) else '0'
                
                # レースタイムを '1m56s3' の形式に変換
                レースタイム = f"{minutes}m{seconds}s{fraction}"
                # print(f"ステップ3: レースタイム={レースタイム}")
            else:
                レースタイム = None  # マッチしない場合はNoneを代入
                print("レースタイムがマッチしませんでした")

            レースID = f"{date.strftime('%Y%m%d')}{place_code}{race_number}"
            record = {
                'レースID': レースID,
                '着': int(着),
                # '艇番': int(艇番),
                '選手登番': 選手登番.strip(),
                '選手名': 選手名.strip(),
                '展示タイム': float(展示タイム),
                '進入': int(進入),
                'スタートタイミング': float(スタートタイミング),
                'レースタイム': レースタイム,
                '天候': 天候,
                '風向': 風向,
                '風量': 風量,
                '波': 波
            }
            results.append(record)
        # オッズ情報の取得
        elif '[払戻金]' in line:
            odds_data = data[idx+1:idx+13]
            odds_list.extend(extract_odds(odds_data, date, place_code))

    k_data = pd.DataFrame(results)
    odds_data = pd.DataFrame(odds_list)
    # print("k_data columns:", k_data.columns.tolist())
    # print(k_data.head())
    return k_data, odds_data

def extract_odds(odds_data_lines, date, place_code):
    """
    Kファイルからオッズ情報を抽出します。
    """
    odds_list = []
    for line in odds_data_lines:
        line = line.strip()
        match = re.match(r'^(\d{1,2}R)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)', line)
        if match:
            race_number = match.group(1).replace('R', '').zfill(2)
            レースID = f"{date.strftime('%Y%m%d')}{place_code}{race_number}"
            三連単結果 = match.group(2)
            三連単オッズ = int(match.group(3)) / 100
            三連複結果 = match.group(4)
            三連複オッズ = int(match.group(5)) / 100
            二連単結果 = match.group(6)
            二連単オッズ = int(match.group(7)) / 100
            二連複結果 = match.group(8)
            二連複オッズ = int(match.group(9)) / 100
            record = {
                'レースID': レースID,
                '三連単結果': 三連単結果,
                '三連単オッズ': 三連単オッズ,
                '三連複結果': 三連複結果,
                '三連複オッズ': 三連複オッズ,
                '二連単結果': 二連単結果,
                '二連単オッズ': 二連単オッズ,
                '二連複結果': 二連複結果,
                '二連複オッズ': 二連複オッズ
            }
            odds_list.append(record)
    return odds_list
    

def preprocess_data_old(df):
    """
    データの前処理を行います。
    """
    # ラベルエンコード
    # ラベルエンコード
    list_LabelEncode = ['支部']
    for label in list_LabelEncode:
        le = LabelEncoder()
        df[label] = le.fit_transform(df[label].astype(str))

    # '級別' のマッピング
    class_mapping = {'B2': 1, 'B1': 2, 'A2': 3, 'A1': 4}
    df['級別'] = df['級別'].map(class_mapping).astype(int)

    # 数値データに変換
    numeric_columns = ['年齢', '体重', '全国勝率', '全国2連率', '当地勝率', '当地2連率',
                       'モーター2連率', 'ボート2連率',
                       'ET', 'tilt', 'EST', 'ESC', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h', 'win_odds', 'place_odds']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 欠損値の処理（欠損行を削除）
    df = df.dropna()

    # レースごとに偏差値を計算
    # list_std = ['全国勝率', '全国2連率', '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率']
    list_std = ['全国勝率', '全国2連率', '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率']
    for col in list_std:
        df.loc[:,col] = df.groupby('レースID')[col].transform(
            lambda x: ((x - x.mean()) / x.std(ddof=0) * 10 + 50) if x.std(ddof=0) != 0 else 50)

    return df

def preprocess_data(df):
    """
    データの前処理を行います。
    """
    # ラベルエンコード
    # ラベルエンコード
    list_LabelEncode = ['支部']
    for label in list_LabelEncode:
        le = LabelEncoder()
        df[label] = le.fit_transform(df[label].astype(str))

    # '級別' のマッピング
    class_mapping = {'B2': 1, 'B1': 2, 'A2': 3, 'A1': 4}
    df['級別'] = df['級別'].map(class_mapping).astype(int)

    # 数値データに変換
    numeric_columns = ['年齢', '体重', '全国勝率', '全国2連率', '当地勝率', '当地2連率',
                       'モーター2連率', 'ボート2連率',
                       'ET', 'tilt', 'EST', 'ESC', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h', 'win_odds', 'place_odds']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 欠損値の処理（欠損行を削除）
    df = df.dropna()

    # レースごとに偏差値を計算
    # list_std = ['全国勝率', '全国2連率', '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率']
    list_std = ['全国勝率', '全国2連率', '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率','今期能力指数','前期能力指数','勝率']
    for col in list_std:
        df.loc[:,col] = df.groupby('レースID')[col].transform(
            lambda x: ((x - x.mean()) / x.std(ddof=0) * 10 + 50) if x.std(ddof=0) != 0 else 50)

    return df
    # list_LabelEncode = ['選手名', '支部']
    # for label in list_LabelEncode:
    #     le = LabelEncoder()
    #     df[label] = le.fit_transform(df[label].astype(str))

    # # map関数でのエンコード
    # place_code_mapping = {'01': '桐生', '02': '戸田', '03': '江戸川', '04': '平和島', '05': '多摩川',
    #                       '06': '浜名湖', '07': '蒲郡', '08': '常滑', '09': '津', '10': '三国',
    #                       '11': 'びわこ', '12': '住之江', '13': '尼崎', '14': '鳴門', '15': '丸亀',
    #                       '16': '児島', '17': '宮島', '18': '徳山', '19': '下関', '20': '若松',
    #                       '21': '芦屋', '22': '福岡', '23': '唐津', '24': '大村'}
    # # place_code = {v: k for k, v in place_code_mapping.items()}
    # class_mapping = {'B2': 1, 'B1': 2, 'A2': 3, 'A1': 4}

    # df['級別'] = df['級別'].map(class_mapping).astype(int)

    # # 数値データに変換
    # numeric_columns = ['年齢', '体重', '全国勝率', '全国2連率', '当地勝率', '当地2連率',
    #                    'モーター2連率', 'ボート2連率', '展示タイム', '風量', '波',
    #                    'ET', 'tilt', 'EST']
    # for col in numeric_columns:
    #     df[col] = pd.to_numeric(df[col], errors='coerce')

    # # 欠損値の処理（欠損行を削除）
    # df = df.dropna()

    # # レースごとに偏差値を計算
    # list_std = ['全国勝率', '全国2連率', '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率']
    # for col in list_std:
    #     df[col] = df.groupby('レースID')[col].transform(
    #         lambda x: ((x - x.mean()) / x.std(ddof=0) * 10 + 50) if x.std(ddof=0) != 0 else 50)

    # return df

def preprocess_data1min(df):
    """
    データの前処理を行います。
    """
    # ラベルエンコード
    # ラベルエンコード
    list_LabelEncode = ['支部']
    for label in list_LabelEncode:
        le = LabelEncoder()
        df[label] = le.fit_transform(df[label].astype(str))

    # '級別' のマッピング
    class_mapping = {'B2': 1, 'B1': 2, 'A2': 3, 'A1': 4}
    df['級別'] = df['級別'].map(class_mapping).astype(int)

    # 数値データに変換
    numeric_columns = ['年齢', '体重', '全国勝率', '全国2連率', '当地勝率', '当地2連率',
                       'モーター2連率', 'ボート2連率',
                       'ET', 'tilt', 'EST', 'ESC', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h', 'win_odds', 'place_odds','win_odds1min', 'place_odds1min']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 欠損値の処理（欠損行を削除）
    # df = df.dropna()

    # レースごとに偏差値を計算
    list_std = ['全国勝率', '全国2連率', '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率','前期能力指数','今期能力指数','勝率']
    for col in list_std:
        df.loc[:,col] = df.groupby('レースID')[col].transform(
            lambda x: ((x - x.mean()) / x.std(ddof=0) * 10 + 50) if x.std(ddof=0) != 0 else 50)

    return df

def preprocess_data3(df):
    """
    データの前処理を行います。
    """
    # ラベルエンコード
    # ラベルエンコード
    list_LabelEncode = ['支部']
    for label in list_LabelEncode:
        le = LabelEncoder()
        df[label] = le.fit_transform(df[label].astype(str))

    # '級別' のマッピング
    class_mapping = {'B2': 1, 'B1': 2, 'A2': 3, 'A1': 4}
    df['級別'] = df['級別'].map(class_mapping).astype(int)

    # 数値データに変換
    numeric_columns = ['年齢', '体重', '全国勝率', '全国2連率', '当地勝率', '当地2連率',
                       'モーター2連率', 'ボート2連率',
                       'ET', 'tilt', 'EST', 'ESC', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h', 'win_odds']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 欠損値の処理（欠損行を削除）
    # df = df.dropna()

    # レースごとに偏差値を計算
    list_std = ['全国勝率', '全国2連率', '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率','前期能力指数','今期能力指数','勝率']
    for col in list_std:
        transformed_values = df.groupby('レースID')[col].transform(
            lambda x: ((x - x.mean()) / x.std(ddof=0) * 10 + 50) if x.std(ddof=0) != 0 else 50
        ).astype(float)
        df[col] = transformed_values  # float型で設定
    return df

def load_dataframe(file_path):
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        print(f"ファイルが見つかりません: {file_path}")
        return None
    
# 追加: beforeinfoデータの読み込み関数
def load_before_data(file_path):
    """
    保存されたbeforeinfoテキストファイルを読み込み、DataFrameに変換します。
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # データを個々のレースごとに分割
    race_entries = content.strip().split('\n\n')

    data_list = []
    for entry in race_entries:
        entry_data = {}
        lines = entry.strip().split('\n')
        for line in lines:
            if line.strip() == '':
                continue
            key_value = line.strip().split()
            if len(key_value) >= 2:
                key = key_value[0]
                value = ' '.join(key_value[1:])
                # 数値に変換可能なものは変換
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # print(f"ValueError: {value}")
                    pass  # 数値に変換できない場合はそのまま
                entry_data[key] = value
        data_list.append(entry_data)

    df = pd.DataFrame(data_list)

    if df.empty:
        return df

    # データをワイド形式からロング形式に変換
    dfs = []
    for idx, row in df.iterrows():
        for i in range(1, 7):  # 1号艇から6号艇まで
            boat_num = i
            new_row = {
                'date': row['date'],
                'place_cd': row['place_cd'],
                'race_no': row['race_no'],
                '艇番': boat_num,
                'ET': row.get(f'ET_{boat_num}', None),
                'tilt': row.get(f'tilt_{boat_num}', None),
                'EST': row.get(f'EST_{boat_num}', None),
                'ESC': row.get(f'ESC_{boat_num}', None),
                'weather': row.get('weather', None),
                'air_t': row.get('air_t', None),
                'wind_d': row.get('wind_d', None),
                'wind_v': row.get('wind_v', None),
                'water_t': row.get('water_t', None),
                'wave_h': row.get('wave_h', None),
                'win_odds':row.get(f'win_odds_{boat_num}', None),
                'place_odds':row.get(f'place_odds_{boat_num}', None),
            }
            dfs.append(new_row)

    df_long = pd.DataFrame(dfs)

    # 'レースID' の作成
    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long['年月日'] = df_long['date'].dt.strftime('%Y%m%d')
    df_long['場コード'] = df_long['place_cd'].astype(int).apply(lambda x: f'{x:02d}')
    df_long['レース番号'] = df_long['race_no'].astype(int).apply(lambda x: f'{x:02d}')
    df_long['レースID'] = df_long['年月日'] + df_long['場コード'] + df_long['レース番号']

    # 不要な列を削除
    df_long = df_long.drop(columns=['date', 'place_cd', 'race_no', '年月日', '場コード', 'レース番号'])
    # print("data columns:", df_long.columns.tolist())
    # print(df_long.head(100))
    return df_long

    # if not os.path.exists(file_path):
    #     print(f"File not found: {file_path}")
    #     return pd.DataFrame()

    # with open(file_path, 'r', encoding='utf-8') as f:
    #     content = f.read()

    # # データを個々のレースごとに分割
    # race_entries = content.strip().split('\n\n')

    # data_list = []
    # for entry in race_entries:
    #     entry_data = {}
    #     lines = entry.strip().split('\n')
    #     for line in lines:
    #         if line.strip() == '':
    #             continue
    #         key_value = line.strip().split()
    #         if len(key_value) >= 2:
    #             key = key_value[0]
    #             value = key_value[1]
    #             # 数値に変換可能なものは変換
    #             try:
    #                 if '.' in value:
    #                     value = float(value)
    #                 else:
    #                     value = int(value)
    #             except ValueError:
    #                 pass  # 数値に変換できない場合はそのまま
    #             entry_data[key] = value
    #     data_list.append(entry_data)

    # df = pd.DataFrame(data_list)
    # return df

    # if not os.path.exists(file_path):
    #     print(f"ファイルが見つかりません: {file_path}")
    #     return pd.DataFrame()
    # before_data = pd.read_csv(file_path)
    # return before_data

# 確定後オッズ推定のためのbefore1minデータの読み込み関数
def load_before1min_data(file_path):
    """
    保存されたbefore1minテキストファイルを読み込み、DataFrameに変換します。
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # データを個々のレースごとに分割
    race_entries = content.strip().split('\n\n')

    data_list = []
    for entry in race_entries:
        entry_data = {}
        lines = entry.strip().split('\n')
        for line in lines:
            if line.strip() == '':
                continue
            key_value = line.strip().split()
            if len(key_value) >= 2:
                key = key_value[0]
                value = ' '.join(key_value[1:])
                # 数値に変換可能なものは変換
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # print(f"ValueError: {value}")
                    pass  # 数値に変換できない場合はそのまま
                entry_data[key] = value
        data_list.append(entry_data)
    
    df = pd.DataFrame(data_list)

    if df.empty:
        return df

    # データをワイド形式からロング形式に変換
    dfs = []
    for idx, row in df.iterrows():
        for i in range(1, 7):
            boat_num = i
            new_row = {
                'date': row['date'],
                'place_cd': row['place_cd'],
                'race_no': row['race_no'],
                '艇番': boat_num,
                'ET': row.get(f'ET_{boat_num}', None),
                'tilt': row.get(f'tilt_{boat_num}', None),
                'EST': row.get(f'EST_{boat_num}', None),
                'ESC': row.get(f'ESC_{boat_num}', None),
                'weather': row.get('weather', None),
                'air_t': row.get('air_t', None),
                'wind_d': row.get('wind_d', None),
                'wind_v': row.get('wind_v', None),
                'water_t': row.get('water_t', None),
                'wave_h': row.get('wave_h', None),
                'win_odds1min': row.get(f'win_odds_{boat_num}', None),
                'place_odds1min': row.get(f'place_odds_{boat_num}', None),
            }
            dfs.append(new_row)
    
    df_long = pd.DataFrame(dfs)

    # 'レースID' の作成
    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long['年月日'] = df_long['date'].dt.strftime('%Y%m%d')
    df_long['場コード'] = df_long['place_cd'].astype(int).apply(lambda x: f'{x:02d}')
    df_long['レース番号'] = df_long['race_no'].astype(int).apply(lambda x: f'{x:02d}')
    df_long['レースID'] = df_long['年月日'] + df_long['場コード'] + df_long['レース番号']

    # 不要な列を削除
    df_long = df_long.drop(columns=['date', 'place_cd', 'race_no', '年月日', '場コード', 'レース番号',
                                     'ET', 'tilt', 'EST', 'ESC',
                                    'weather', 'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h'])
    # print("data columns:", df_long.columns.tolist())
    # print(df_long.head(100))
    return df_long

# 追加: beforeinfoデータの取得関数
def get_beforeinfo(date, place_cd, race_no):
    """
    beforeinfoページからデータを取得します。
    """
    url = get_url(date, place_cd, race_no, 'beforeinfo')
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve beforeinfo data. HTTP Status Code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    try:
        # 展示タイム等のデータを取得
        rows = soup.find_all('tr', class_='is-fs12')
        data_list = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 6:
                ET = cols[4].text.strip()
                tilt = cols[5].text.strip()
                data_list.append([ET, tilt])

        # 天候、風向、風速、波高を取得
        weather_elements = soup.find_all(class_='weather1_bodyUnitLabelData')
        if len(weather_elements) >= 4:
            air_t, wind_v, water_t, wave_h = [elem.text.strip() for elem in weather_elements]
        else:
            air_t = wind_v = water_t = wave_h = None

        weather = soup.find('p', class_='weather1_bodyUnitLabelTitle').text.strip()
        wind_direction_elem = soup.find('p', class_='weather1_bodyUnitImage is-wind')
        wind_d = wind_direction_elem['class'][1] if wind_direction_elem else None

        # データの整形
        df = pd.DataFrame(data_list, columns=['ET', 'tilt'])
        df['艇番'] = df.index + 1
        df['date'] = date
        df['place_cd'] = place_cd
        df['race_no'] = race_no
        df['weather'] = weather
        df['wind_d'] = wind_d
        df['wind_v'] = wind_v
        df['wave_h'] = wave_h

        return df
    except Exception as e:
        print(f"Error parsing beforeinfo data: {e}")
        return None

def get_url(date, place_cd, race_no, content):
    """
    指定された情報をもとにURLを作成します。
    """
    url_t = 'https://www.boatrace.jp/owpc/pc/race/'
    ymd = str(pd.to_datetime(date)).split()[0].replace('-', '')
    jcd = f'0{place_cd}' if place_cd < 10 else str(place_cd)
    url = f'{url_t}{content}?rno={race_no}&jcd={jcd}&hd={ymd}'
    return url

def merge_before_data(data, before_data):
    """
    メインデータとbeforeinfoデータをマージします。
    """
    # '艇番' を整数型に変換
    data['艇番'] = data['艇番'].astype(int)
    before_data['艇番'] = before_data['艇番'].astype(int)

    # 'レースID' が存在しない場合は作成
    if 'レースID' not in before_data.columns:
        # 'before_data' に 'レースID' を作成
        before_data['date'] = pd.to_datetime(before_data['date'])
        before_data['年月日'] = before_data['date'].dt.strftime('%Y%m%d')
        before_data['場コード'] = before_data['place_cd'].astype(int).apply(lambda x: f'{x:02d}')
        before_data['レース番号'] = before_data['race_no'].astype(int).apply(lambda x: f'{x:02d}')
        before_data['レースID'] = before_data['年月日'] + before_data['場コード'] + before_data['レース番号']

    # マージ
    merged_data = pd.merge(data, before_data, on=['レースID', '艇番'], how='left')

    # 不要な列を削除
    merged_data = merged_data.drop(columns=['date', 'place_cd', 'race_no', '年月日', '場コード', 'レース番号'], errors='ignore')

    return merged_data
    # # マージのためにキーを作成
    # data['date'] = pd.to_datetime(data['レースID'].str[:8], format='%Y%m%d')
    # data['place_cd'] = data['レースID'].str[8:10].astype(int)
    # data['race_no'] = data['レースID'].str[10:12].astype(int)
    # data['艇番'] = data['艇番'].astype(int)

    # # before_dataのキーを整形
    # before_data['date'] = pd.to_datetime(before_data['date'])
    # before_data['place_cd'] = before_data['place_cd'].astype(int)
    # before_data['race_no'] = before_data['race_no'].astype(int)
    # before_data['艇番'] = before_data['艇番'].astype(int)

    # # マージ
    # merged_data = pd.merge(data, before_data, on=['date', 'place_cd', 'race_no', '艇番'], how='left')

    # return merged_data

