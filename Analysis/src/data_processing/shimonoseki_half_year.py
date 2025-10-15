import requests
from bs4 import BeautifulSoup
import math
import time
import json
import sqlite3
import os

# 競艇場コードと名前のマッピング
RACE_VENUE_MAPPING = {
    '01': '桐生',
    '02': '戸田',
    '03': '江戸川',
    '04': '平和島',
    '05': '多摩川',
    '06': '浜名湖',
    '07': '蒲郡',
    '08': '常滑',
    '09': '津',
    '10': '三国',
    '11': 'びわこ',
    '12': '住之江',
    '13': '尼崎',
    '14': '鳴門',
    '15': '丸亀',
    '16': '児島',
    '17': '宮島',
    '18': '徳山',
    '19': '下関',
    '20': '若松',
    '21': '芦屋',
    '22': '福岡',
    '23': '唐津',
    '24': '大村',
}

# ページからHTMLを取得する関数
def get_html_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # HTTPエラーが発生した場合、例外を発生させる
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve page: {e}")
        return None

# レースページから予想と結果を取得
def parse_race_page(race_url):
    html_content = get_html_from_url(race_url)
    if html_content is None:
        return None, None
    soup = BeautifulSoup(html_content, 'html.parser')

    # 予想データを取得（id='a3ren' の三連単）
    predictions = []
    try:
        a3ren_div = soup.find('div', id='a3ren')
        if not a3ren_div:
            print("No 'a3ren' div found.")
            return None, None

        # テーブルを取得
        table = a3ren_div.find('table')
        if not table:
            print("No table found in 'a3ren' div.")
            return None, None

        # ヘッダーの列名を取得
        headers = [th.text.strip() for th in table.find('thead').find_all('th')]
        # print("Headers:", headers)  # デバッグ用出力

        # 列名に対応するインデックスを取得
        trifecta_idx = headers.index('組') if '組' in headers else None
        probability_idx = headers.index('AI予想確率') if 'AI予想確率' in headers else None
        odds_idx = headers.index('オッズ') if 'オッズ' in headers else None
        poseidon_idx = headers.index('海神指数') if '海神指数' in headers else None

        if None in (trifecta_idx, probability_idx, odds_idx, poseidon_idx):
            print("Required columns not found in the table.")
            return None, None

        # データ行を取得
        rows = table.find('tbody').find_all('tr')
        for row in rows[:21]:  # 上位20位
            cells = row.find_all(['th', 'td'])
            if len(cells) >= max(trifecta_idx, probability_idx, odds_idx, poseidon_idx) + 1:
                # セルの内容を取得
                trifecta = cells[trifecta_idx].text.strip()
                probability_text = cells[probability_idx].text.strip()
                odds_text = cells[odds_idx].text.strip()
                poseidon_text = cells[poseidon_idx].text.strip()

                # オッズの変換
                odds_text_clean = odds_text.replace('pt', '').replace(',', '').strip()
                try:
                    odds = float(odds_text_clean)
                except ValueError:
                    odds = -1.0  # オッズが不明な場合の特別な値

                # 確率の変換
                try:
                    probability = float(probability_text.strip('%'))
                except ValueError:
                    probability = 0.0  # 確率が不明な場合の特別な値

                # 海神指数の変換
                try:
                    poseidon_index = float(poseidon_text)
                except ValueError:
                    poseidon_index = 0.0  # 海神指数が不明な場合の特別な値

                predictions.append({
                    'trifecta': trifecta,
                    'probability': probability,
                    'odds': odds,
                    'poseidon_index': poseidon_index
                })
                # デバッグ用出力をコメントアウト
                # print(f"Found prediction: Trifecta={trifecta}, Probability={probability}%, Odds={odds}, Poseidon Index={poseidon_index}")
            else:
                print("Insufficient number of cells in the row.")
    except Exception as e:
        print(f"Failed to parse predictions: {e}")
        return None, None

    # レース結果を取得
    race_result = None
    try:
        detail_result_div = soup.find('div', id='detail_result', class_='tab-pane active row')
        if detail_result_div:
            p_tag = detail_result_div.find('p', class_='h4')
            if p_tag:
                race_result = p_tag.text.strip()
                # print(f"Race result: {race_result}")  # デバッグ用出力
        else:
            print("No 'detail_result' div found.")
    except Exception as e:
        print(f"Failed to parse race result: {e}")
        race_result = None  # 結果が未確定の場合はNone

    return predictions, race_result

# レースデータをJSONファイルに保存する関数
def save_race_data_json(all_race_data, filename="race_data.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_race_data, f, ensure_ascii=False, indent=4)

# レースデータをSQLiteデータベースに保存する関数
def save_race_data_sqlite(all_race_data, db_filename="race_data.sqlite"):
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()

    # テーブルを作成
    c.execute('''
        CREATE TABLE IF NOT EXISTS race_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_date TEXT,
            race_number TEXT,
            trifecta TEXT,
            probability REAL,
            odds REAL,
            poseidon_index REAL,
            race_result TEXT
        )
    ''')

    # データを挿入
    for race in all_race_data:
        race_date = race['race_date']
        race_number = race['race_number']
        race_result = race.get('race_result', 'Not Available')
        for prediction in race['predictions']:
            c.execute('''
                INSERT INTO race_predictions (race_date, race_number, trifecta, probability, odds, poseidon_index, race_result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_date,
                race_number,
                prediction['trifecta'],
                prediction['probability'],
                prediction['odds'],
                prediction['poseidon_index'],
                race_result
            ))
    conn.commit()
    conn.close()

def calculate_bet_allocation(predictions, total_funds, minimum_bet):
    top_predictions = predictions[:21]  # 上位11個の予想を使用
    inverse_odds_sum = sum([1 / p['odds'] for p in top_predictions if p['odds'] > 0])

    if inverse_odds_sum == 0:
        # print("Invalid odds. Cannot calculate bet allocation.")
        return []

    bet_allocations = []
    expected_payout = total_funds / inverse_odds_sum  # 期待払戻金

    for prediction in top_predictions:
        odds = prediction['odds']
        trifecta = prediction['trifecta']
        probability = prediction['probability']

        if odds <= 0:
            # オッズが無効な場合、スキップ
            continue

        # 購入金額を計算
        bet_amount = (expected_payout / odds)
        # 最小賭け金の倍数に調整
        bet_amount = max(math.floor(bet_amount / minimum_bet) * minimum_bet, minimum_bet)

        bet_allocations.append({
            'trifecta': trifecta,
            'bet_amount': bet_amount,
            'odds': odds,
            'probability': probability,
            'expected_payout': bet_amount * odds
        })

    # 購入金額の合計を計算
    total_bet_amount = sum([bet['bet_amount'] for bet in bet_allocations])

    # 購入金額の合計が総資金を超える場合、比率で調整
    if total_bet_amount > total_funds:
        scaling_factor = total_funds / total_bet_amount
        for bet in bet_allocations:
            bet['bet_amount'] = max(math.floor(bet['bet_amount'] * scaling_factor / minimum_bet) * minimum_bet, minimum_bet)
        # 再度合計を計算
        total_bet_amount = sum([bet['bet_amount'] for bet in bet_allocations])

    # 残りの資金を配分（端数調整）
    remaining_funds = total_funds - total_bet_amount
    idx = 0
    while remaining_funds >= minimum_bet:
        bet_allocations[idx % len(bet_allocations)]['bet_amount'] += minimum_bet
        remaining_funds -= minimum_bet
        idx += 1

    # 各ベットの期待払戻金を再計算
    for bet in bet_allocations:
        bet['expected_payout'] = bet['bet_amount'] * bet['odds']

    return bet_allocations

def calculate_return_rate(bet_allocations, race_result):
    total_investment = sum([bet['bet_amount'] for bet in bet_allocations])
    payout = 0

    for bet in bet_allocations:
        if bet['trifecta'] == race_result:
            payout += bet['bet_amount'] * bet['odds']
            break

    return_rate = payout / total_investment if total_investment > 0 else 0
    return return_rate, payout, total_investment

def main():
    race_venue_code = '19'  # 下関
    race_name = RACE_VENUE_MAPPING.get(race_venue_code, f"Unknown Venue ({race_venue_code})")

    # 指定された日程のリストを作成
    race_dates = []
    # 4月
    race_dates += [f"2024{'04'}{str(day).zfill(2)}" for day in [4,5,6,7,8,17,18,19,20,21,22,23,25,26,27,28,29]]
    # 5月
    race_dates += [f"2024{'05'}{str(day).zfill(2)}" for day in [1,2,3,4,14,15,16,17,18,19,20,26,27,28,29,31]]
    # 6月
    race_dates += [f"2024{'06'}{str(day).zfill(2)}" for day in [1,2,3,14,15,16,17,18,20,21,22,23,24,25]]
    # 7月
    race_dates += [f"2024{'07'}{str(day).zfill(2)}" for day in [3,4,5,6,7,15,16,17,18,19,27,28,29,30,31]]
    # 8月
    race_dates += [f"2024{'08'}{str(day).zfill(2)}" for day in [1,2,5,6,7,8,9,13,14,15,16,17,20,21,22,23,24,25,26]]
    # 9月
    race_dates += [f"2024{'09'}{str(day).zfill(2)}" for day in [3,4,5,6,9,10,11,12,13,14,20,21,22,23,24,25]]
    # 10月
    race_dates += [f"2024{'10'}{str(day).zfill(2)}" for day in [3,4]]

    # 投資しない確率の閾値リスト
    threshold_list = [0, 6, 8, 10, 11, 12]
    threshold_results = {}

    for threshold in threshold_list:
        print(f"\n=== Processing with probability threshold: {threshold}% ===")
        all_race_data = []
        total_investment_all = 0
        total_payout_all = 0
        hit_races = []
        invested_races = []

        for race_date in race_dates:
            print(f"Processing races on {race_date} at Venue {race_name}...")
            for race_number in range(1, 13):  # 1Rから12Rまで
                race_number_str = f"{race_number}R"
                race_url = f"https://poseidon-boatrace.net/race/{race_date}/{race_venue_code}/{race_number_str}"
                # サーバーに負荷をかけないように待機
                # time.sleep(1)

                try:
                    predictions, race_result = parse_race_page(race_url)
                except Exception as e:
                    print(f"    Exception occurred while parsing race page: {e}")
                    continue

                if predictions is None or len(predictions) == 0:
                    print(f"    No predictions available for Race {race_number_str} on {race_date}.")
                    continue

                # 取得した予想データを保存
                race_data = {
                    'race_date': race_date,
                    'race_number': race_number_str,
                    'race_result': race_result if race_result else 'Not Available',
                    'predictions': predictions
                }
                all_race_data.append(race_data)

                # 一番上の予想のprobabilityを確認
                top_probability = predictions[0]['probability']
                if top_probability < threshold:
                    # print(f"    Skipping Race {race_number_str} on {race_date} because top probability is {top_probability}% which is less than {threshold}%")
                    continue

                total_funds = 10000
                minimum_bet = 100

                bet_allocations = calculate_bet_allocation(predictions, total_funds, minimum_bet)
                if not bet_allocations:
                    # print(f"    Failed to calculate bet allocations for Race {race_number_str} on {race_date}.")
                    continue

                return_rate, payout, total_investment = calculate_return_rate(bet_allocations, race_result)

                race_result_str = race_result if race_result else "Not Available"

                # 投資結果を記録
                total_investment_all += total_investment
                total_payout_all += payout

                # 当たったレースを記録
                if payout > 0:
                    hit_races.append(f"{race_date}-{race_number_str}")

                # 投資したレースを記録
                invested_races.append(f"{race_date}-{race_number_str}")

        # レースデータを保存（ファイル名に閾値を含める）
        save_race_data_json(all_race_data, filename=f"race_data_threshold_{threshold}.json")
        save_race_data_sqlite(all_race_data, db_filename=f"race_data_threshold_{threshold}.sqlite")

        # 全体の回収率を計算
        overall_return_rate = (total_payout_all / total_investment_all) * 100 if total_investment_all > 0 else 0
        threshold_results[threshold] = overall_return_rate

        print("=" * 60)
        print(f"Threshold: {threshold}%")
        print(f"Total Investment: {total_investment_all} 円")
        print(f"Total Payout: {total_payout_all} 円")
        print(f"Overall Return Rate: {overall_return_rate:.2f}%")
        print(f"Number of races invested: {len(invested_races)}")
        # print(f"Invested Races: {', '.join(invested_races) if invested_races else 'None'}")  # コメントアウト
        print(f"Number of Hit Races: {len(hit_races)}")
        # print(f"Hit Races: {', '.join(hit_races) if hit_races else 'None'}")
        print("=" * 60)

    # 最適な閾値を特定
    optimal_threshold = max(threshold_results, key=threshold_results.get)
    max_return_rate = threshold_results[optimal_threshold]

    print("\n=== Optimal Threshold Result ===")
    print(f"Optimal Probability Threshold: {optimal_threshold}%")
    print(f"Maximum Overall Return Rate: {max_return_rate:.2f}%")
    print("===============================")

if __name__ == "__main__":
    main()
