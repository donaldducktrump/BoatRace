# mymodule.py

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
                
                # Step 2: 選手名（全角4文字分を抽出）
                # 正規表現を使って全角4文字分（選手名）を抽出
                name_pattern = r'^\d\s+\d{4}([^\x00-\x7F\u3000]{2,4})'
                name_match = re.search(name_pattern, racer_line)
                if name_match:
                    選手名 = name_match.group(1).strip()
                    # print(f"Step 2: 選手名={選手名}")
                else:
                    print(f"Step 2: マッチ失敗 - {racer_line}")
                    continue

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
                    '選手名': 選手名.strip(),
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
        if 'KBGN' in line:
            place_code_match = re.match(r'(\d{2})KBGN', line.strip())
            if place_code_match:
                place_code = place_code_match.group(1)
            else:
                print("場コードを取得できませんでした。")
                place_code = '00'
            # print(f"会場コード: {place_code}")
        # レース番号と天候等の取得
        if re.match(r'^\s*\d{1,2}R', line) and not re.search(r'\d-\d-\d', line):
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
        if '単勝' in line:
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
        if '複勝' in line:
            odds_match = re.search(r'複勝\s+(\d)\s+(\d+)\s+(\d)\s+(\d+)', line)
            if odds_match:
                first_boat = odds_match.group(1)  # 1着艇番
                second_boat = odds_match.group(3)
                first_odds = int(odds_match.group(2)) / 100  # 1着オッズ
                second_odds = int(odds_match.group(4)) / 100
                if race_number and date and place_code:
                    レースID = f"{date.strftime('%Y%m%d')}{place_code}{race_number}"
                    
                    # 単勝データを作成
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
                '艇番': int(艇番),
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

def preprocess_data(df):
    """
    データの前処理を行います。
    """
    # ラベルエンコード
    list_LabelEncode = ['選手名', '支部', '天候', '風向','会場']
    for label in list_LabelEncode:
        le = LabelEncoder()
        df[label] = le.fit_transform(df[label].astype(str))

    # map関数でのエンコード
    place_code_mapping = {'01': '桐生', '02': '戸田', '03': '江戸川', '04': '平和島', '05': '多摩川',
                          '06': '浜名湖', '07': '蒲郡', '08': '常滑', '09': '津', '10': '三国',
                          '11': 'びわこ', '12': '住之江', '13': '尼崎', '14': '鳴門', '15': '丸亀',
                          '16': '児島', '17': '宮島', '18': '徳山', '19': '下関', '20': '若松',
                          '21': '芦屋', '22': '福岡', '23': '唐津', '24': '大村'}
    place_code = {v: k for k, v in place_code_mapping.items()}
    class_mapping = {'B2': 1, 'B1': 2, 'A2': 3, 'A1': 4}

    # df['会場'] = df['会場'].map(place_code).astype(int)
    df['級別'] = df['級別'].map(class_mapping).astype(int)

    # 数値データに変換
    numeric_columns = ['年齢', '体重', '全国勝率', '全国2連率', '当地勝率', '当地2連率',
                       'モーター2連率', 'ボート2連率', '展示タイム', '風量', '波']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 欠損値の処理（欠損行を削除）
    df = df.dropna()

    # レースごとに偏差値を計算
    list_std = ['全国勝率', '全国2連率', '当地勝率', '当地2連率', 'モーター2連率', 'ボート2連率']
    for col in list_std:
        df[col] = df.groupby('レースID')[col].transform(
            lambda x: ((x - x.mean()) / x.std(ddof=0) * 10 + 50) if x.std(ddof=0) != 0 else 50)

    return df

