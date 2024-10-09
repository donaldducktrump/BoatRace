import requests
from bs4 import BeautifulSoup
import math
import time

# レース場コードのマッピング
RACE_VENUE_MAPPING_REVERSE = {
    '桐生': '1',
    '戸田': '2',
    '江戸川': '3',
    '平和島': '4',
    '多摩川': '5',
    '浜名湖': '6',
    '蒲郡': '7',
    '常滑': '8',
    '津': '9',
    '三国': '10',
    'びわこ': '11',
    '住之江': '12',
    '尼崎': '13',
    '鳴門': '14',
    '丸亀': '15',
    '児島': '16',
    '宮島': '17',
    '徳山': '18',
    '下関': '19',
    '若松': '20',
    '芦屋': '21',
    '福岡': '22',
    '唐津': '23',
    '大村': '24',
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

# レースページから予想と結果を取得する関数
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

        # 列名に対応するインデックスを取得
        trifecta_idx = headers.index('組') if '組' in headers else None
        probability_idx = headers.index('AI予想確率') if 'AI予想確率' in headers else None
        odds_idx = headers.index('オッズ') if 'オッズ' in headers else None

        if None in (trifecta_idx, probability_idx, odds_idx):
            print("Required columns not found in the table.")
            return None, None

        # データ行を取得
        rows = table.find('tbody').find_all('tr')
        for row in rows:
            cells = row.find_all(['th', 'td'])
            if len(cells) >= max(trifecta_idx, probability_idx, odds_idx) + 1:
                trifecta = cells[trifecta_idx].text.strip()
                probability_text = cells[probability_idx].text.strip()
                odds_text = cells[odds_idx].text.strip()

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

                predictions.append({
                    'trifecta': trifecta,
                    'probability': probability,
                    'odds': odds
                })
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
        else:
            print("No 'detail_result' div found.")
    except Exception as e:
        print(f"Failed to parse race result: {e}")
        race_result = None  # 結果が未確定の場合はNone

    return predictions, race_result

# 資金配分を計算する関数
def calculate_bet_allocation(predictions, total_funds, minimum_bet, top_n):
    top_predictions = predictions[:top_n+1]
    inverse_odds_sum = sum([1 / p['odds'] for p in top_predictions if p['odds'] > 0])

    if inverse_odds_sum == 0:
        print("Invalid odds. Cannot calculate bet allocation.")
        return []

    bet_allocations = []
    expected_payout = total_funds / inverse_odds_sum  # 期待払戻金

    for prediction in top_predictions:
        odds = prediction['odds']
        trifecta = prediction['trifecta']
        probability = prediction['probability']

        if odds <= 0:
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
        total_bet_amount = sum([bet['bet_amount'] for bet in bet_allocations])

    # 残りの資金を配分（端数調整）
    remaining_funds = total_funds - total_bet_amount
    idx = 0
    while remaining_funds >= minimum_bet and bet_allocations:
        bet_allocations[idx % len(bet_allocations)]['bet_amount'] += minimum_bet
        remaining_funds -= minimum_bet
        idx += 1

    # 各ベットの期待払戻金を再計算
    for bet in bet_allocations:
        bet['expected_payout'] = bet['bet_amount'] * bet['odds']

    return bet_allocations

# 回収率を計算する関数
def calculate_return_rate(bet_allocations, race_result):
    total_investment = sum([bet['bet_amount'] for bet in bet_allocations])
    payout = 0

    for bet in bet_allocations:
        if bet['trifecta'] == race_result:
            payout += bet['bet_amount'] * bet['odds']
            break

    return payout, total_investment

def main():
    # 投資する予想数のリスト
    top_n_list = [1, 2, 3, 5, 10, 20]

    # 指定されたレースリスト
    race_list = [
        {'venue': '下関', 'race_number': '8R'},
        {'venue': '大村', 'race_number': '1R'},
        {'venue': '浜名湖', 'race_number': '7R'},
        {'venue': '大村', 'race_number': '10R'},
        {'venue': '大村', 'race_number': '11R'},
        {'venue': '唐津', 'race_number': '1R'},
        {'venue': '児島', 'race_number': '2R'},
        {'venue': '三国', 'race_number': '1R'},
        {'venue': '唐津', 'race_number': '5R'},
        {'venue': '下関', 'race_number': '12R'},
        {'venue': '児島', 'race_number': '1R'},
        {'venue': '唐津', 'race_number': '12R'},
        {'venue': '浜名湖', 'race_number': '9R'},
        {'venue': '津', 'race_number': '12R'},
        {'venue': '下関', 'race_number': '10R'},
        {'venue': '唐津', 'race_number': '2R'},
        {'venue': '浜名湖', 'race_number': '12R'},
        {'venue': '津', 'race_number': '1R'},
        {'venue': '下関', 'race_number': '9R'},
        {'venue': '児島', 'race_number': '12R'},
    ]

    race_date = '20241004'  # レース日付を指定

    # レースデータを保存するリスト
    all_race_data = []

    for race_info in race_list:
        venue_name = race_info['venue']
        race_number = race_info['race_number']
        race_venue_code = RACE_VENUE_MAPPING_REVERSE.get(venue_name)
        if not race_venue_code:
            print(f"Unknown venue: {venue_name}")
            continue

        race_url = f"https://poseidon-boatrace.net/race/{race_date}/{race_venue_code}/{race_number}"
        print(f"Processing {venue_name} {race_number} on {race_date}...")

        time.sleep(1)  # サーバーへの負荷軽減

        try:
            predictions, race_result = parse_race_page(race_url)
        except Exception as e:
            print(f"    Exception occurred while parsing race page: {e}")
            continue

        if predictions is None or len(predictions) == 0:
            print("    Failed to get predictions.")
            continue

        # レースデータを保存
        race_data = {
            'venue': venue_name,
            'race_number': race_number,
            'race_date': race_date,
            'predictions': predictions,
            'race_result': race_result
        }
        all_race_data.append(race_data)

    # 投資する予想数ごとにシミュレーションを行う
    for top_n in top_n_list:
        print(f"\nAnalyzing with Top {top_n} Predictions...")
        total_investment_all = 0
        total_payout_all = 0
        invested_races = 0
        hit_races = []

        for race_data in all_race_data:
            predictions = race_data['predictions']
            race_result = race_data['race_result']
            venue_name = race_data['venue']
            race_number = race_data['race_number']

            if race_result is None:
                print(f"    Race result not available for {venue_name} {race_number}. Skipping.")
                continue

            total_funds = 10000
            minimum_bet = 100

            bet_allocations = calculate_bet_allocation(predictions, total_funds, minimum_bet, top_n)
            if not bet_allocations:
                print(f"    Failed to calculate bet allocations for {venue_name} {race_number}.")
                continue

            payout, total_investment = calculate_return_rate(bet_allocations, race_result)

            # 結果を表示
            # print(f"    Total Investment: {total_investment} 円")
            # print(f"    Payout: {payout} 円")
            return_rate = (payout / total_investment) * 100 if total_investment > 0 else 0
            # print(f"    Return Rate: {return_rate:.2f}%")
            # print(f"    Result: {race_result}")
            # print("-" * 40)

            total_investment_all += total_investment
            total_payout_all += payout
            invested_races += 1

            if payout > 0:
                hit_races.append(f"{venue_name} {race_number}")

        overall_return_rate = (total_payout_all / total_investment_all) * 100 if total_investment_all > 0 else 0
        print("=" * 60)
        print(f"Top {top_n} Predictions:")
        print(f"Total Investment: {total_investment_all} 円")
        print(f"Total Payout: {total_payout_all} 円")
        print(f"Overall Return Rate: {overall_return_rate:.2f}%")
        print(f"Number of races invested: {invested_races}")
        print(f"Hit Races: {len(hit_races)}")
        print("=" * 60)

if __name__ == "__main__":
    main()
