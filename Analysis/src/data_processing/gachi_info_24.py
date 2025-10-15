import requests
from bs4 import BeautifulSoup, Comment
import math
import time

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
        print("Headers:", headers)  # デバッグ用出力

        # 列名に対応するインデックスを取得
        trifecta_idx = headers.index('組') if '組' in headers else None
        probability_idx = headers.index('AI予想確率') if 'AI予想確率' in headers else None
        odds_idx = headers.index('オッズ') if 'オッズ' in headers else None

        if None in (trifecta_idx, probability_idx, odds_idx):
            print("Required columns not found in the table.")
            return None, None

        # データ行を取得
        rows = table.find('tbody').find_all('tr')
        for row in rows[:11]:  # 上位10位
            cells = row.find_all(['th', 'td'])
            if len(cells) >= max(trifecta_idx, probability_idx, odds_idx) + 1:
                # デバッグ用にセルの内容を表示
                cell_texts = [cell.text.strip() for cell in cells]
                # print("Row cells:", cell_texts)  # 必要に応じてコメント解除

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
                # print(f"Found prediction: Trifecta={trifecta}, Probability={probability}%, Odds={odds}")  # デバッグ用出力
            else:
                print("Insufficient number of cells in the row.")
    except Exception as e:
        print(f"Failed to parse predictions: {e}")
        return None, None

    # レース結果を取得（直接指定したdivから取得）
    race_result = None
    try:
        detail_result_div = soup.find('div', id='detail_result', class_='tab-pane active row')
        if detail_result_div:
            p_tag = detail_result_div.find('p', class_='h4')
            if p_tag:
                race_result = p_tag.text.strip()
                print(f"Race result: {race_result}")  # デバッグ用出力
        else:
            print("No 'detail_result' div found.")
    except Exception as e:
        print(f"Failed to parse race result: {e}")
        race_result = None  # 結果が未確定の場合はNone

    return predictions, race_result

# 資金配分を計算する関数（省略なし）
def calculate_bet_allocation(predictions, total_funds, minimum_bet):
    top_predictions = predictions[:11]
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

# 回収率を計算する関数（省略なし）
def calculate_return_rate(bet_allocations, race_result):
    total_investment = sum([bet['bet_amount'] for bet in bet_allocations])
    payout = 0

    for bet in bet_allocations:
        if bet['trifecta'] == race_result:
            payout += bet['bet_amount'] * bet['odds']
            break

    return_rate = payout / total_investment if total_investment > 0 else 0
    return return_rate, payout, total_investment

# 結果をファイルに保存する関数（probabilityを追加）
def save_results_to_file(results, filename="shimonoseki_race_results.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"Race Venue: {result['race_name']}\n")
            f.write(f"Race Number: {result['race_number']}\n")
            f.write(f"Total Investment: {result['total_investment']} 円\n")
            f.write(f"Payout: {result['payout']} 円\n")
            f.write(f"Return Rate: {result['return_rate'] * 100:.2f}%\n")
            f.write(f"Result: {result['race_result']}\n")
            f.write("Bet Allocations:\n")
            for bet in result['bet_allocations']:
                f.write(f"  Trifecta: {bet['trifecta']}, Probability: {bet['probability']}%, Bet Amount: {bet['bet_amount']} 円, Odds: {bet['odds']}, Expected Payout: {bet['expected_payout']:.2f} 円\n")
            f.write("\n")

# 結果をコンソールに出力する関数（probabilityを追加）
def print_results(results):
    for result in results:
        print(f"Race Venue: {result['race_name']}")
        print(f"Race Number: {result['race_number']}")
        print(f"Total Investment: {result['total_investment']} 円")
        print(f"Payout: {result['payout']} 円")
        print(f"Return Rate: {result['return_rate'] * 100:.2f}%")
        print(f"Result: {result['race_result']}")
        print("Bet Allocations:")
        for bet in result['bet_allocations']:
            print(f"  Trifecta: {bet['trifecta']}, Probability: {bet['probability']}%, Bet Amount: {bet['bet_amount']} 円, Odds: {bet['odds']}, Expected Payout: {bet['expected_payout']:.2f} 円")
        print("-" * 40)

def main():
    race_date = '20241003'  # 2024年10月3日
    race_venue_code = '24'  # 大村
    race_name = RACE_VENUE_MAPPING.get(race_venue_code, f"Unknown Venue ({race_venue_code})")
    all_results = []
    total_investment_all = 0
    total_payout_all = 0
    hit_races = []

    for race_number in range(1, 13):  # 1Rから12Rまで
        race_number_str = f"{race_number}R"
        race_url = f"https://poseidon-boatrace.net/race/{race_date}/{race_venue_code}/{race_number_str}"
        print(f"Processing Race {race_number_str} at Venue {race_name} on {race_date}...")

        try:
            predictions, race_result = parse_race_page(race_url)
        except Exception as e:
            print(f"    Exception occurred while parsing race page: {e}")
            continue

        if predictions is None or len(predictions) == 0:
            print("    Failed to get predictions.")
            continue

        total_funds = 10000
        minimum_bet = 100

        bet_allocations = calculate_bet_allocation(predictions, total_funds, minimum_bet)
        if not bet_allocations:
            print("    Failed to calculate bet allocations.")
            continue

        return_rate, payout, total_investment = calculate_return_rate(bet_allocations, race_result)

        race_result_str = race_result if race_result else "Not Available"

        # 結果を表示
        print(f"    Total Investment: {total_investment} 円")
        print(f"    Payout: {payout} 円")
        print(f"    Return Rate: {return_rate * 100:.2f}%")
        print(f"    Result: {race_result_str}")
        print("    Bet Allocations:")
        for bet in bet_allocations:
            print(f"      Trifecta: {bet['trifecta']}, Probability: {bet['probability']}%, Bet Amount: {bet['bet_amount']} 円, Odds: {bet['odds']}, Expected Payout: {bet['expected_payout']:.2f} 円")
        print("-" * 40)

        result = {
            'race_name': race_name,
            'race_number': race_number_str,
            'bet_allocations': bet_allocations,
            'return_rate': return_rate,
            'payout': payout,
            'total_investment': total_investment,
            'race_result': race_result_str
        }

        all_results.append(result)

        # 総合計の計算
        total_investment_all += total_investment
        total_payout_all += payout

        # 当たったレースを記録
        if payout > 0:
            hit_races.append(race_number_str)

        # サーバーに負荷をかけないように待機
        time.sleep(1)

    
    # 結果をファイルに保存
    save_results_to_file(all_results, filename="shimonoseki_race_results.txt")
    # 結果をコンソールに出力
    print_results(all_results)


    # 全体の回収率を計算
    overall_return_rate = (total_payout_all / total_investment_all) * 100 if total_investment_all > 0 else 0
    print("=" * 40)
    print(f"Total Investment for all races: {total_investment_all} 円")
    print(f"Total Payout for all races: {total_payout_all} 円")
    print(f"Overall Return Rate: {overall_return_rate:.2f}%")
    print(f"Hit Races: {', '.join(hit_races) if hit_races else 'None'}")
    print("=" * 40)

if __name__ == "__main__":
    main()
