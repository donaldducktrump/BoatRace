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

# ガチガチのレースリンクを取得
def get_gachigachi_race_links():
    url = 'https://poseidon-boatrace.net/pickup'
    html_content = get_html_from_url(url)
    if html_content is None:
        return []
    soup = BeautifulSoup(html_content, 'html.parser')
    race_links = []

    # ガチガチのタブ内のレースリンクを抽出
    nigeru_div = soup.find('div', id='nigeru')
    if not nigeru_div:
        print("No 'nigeru' div found.")
        return []

    # テーブル内のリンクを取得
    table = nigeru_div.find('table')
    if not table:
        print("No table found in 'nigeru' div.")
        return []

    rows = table.find_all('tr')
    for row in rows:
        link_tag = row.find('a')
        if link_tag:
            race_url = 'https://poseidon-boatrace.net' + link_tag.get('href')
            print("Race URL:", race_url)  # デバッグ用出力
            race_links.append(race_url)

    return race_links

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
        # 列名に対応するインデックスを取得
        trifecta_idx = headers.index('組') if '組' in headers else None
        probability_idx = headers.index('AI予想確率') if 'AI予想確率' in headers else None
        odds_idx = headers.index('オッズ') if 'オッズ' in headers else None

        if None in (trifecta_idx, probability_idx, odds_idx):
            print("Required columns not found in the table.")
            return None, None

        # データ行を取得
        rows = table.find('tbody').find_all('tr')
        for row in rows[:10]:  # 上位10位
            cells = row.find_all('td')
            if len(cells) >= max(trifecta_idx, probability_idx, odds_idx):
                trifecta = cells[trifecta_idx].text.strip()
                probability_text = cells[probability_idx].text.strip()
                odds_text = cells[odds_idx].text.strip()

                # オッズの変換
                try:
                    odds = float(odds_text.replace(',', ''))
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
                print(f"Found prediction: Trifecta={trifecta}, Probability={probability_text}, Odds={odds_text}")  # デバッグ用出力
    except Exception as e:
        print(f"Failed to parse predictions: {e}")
        return None, None

    # レース結果を取得（コメント <!-- 結果 --> から）
    race_result = None
    try:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        result_comment_index = None
        for idx, comment in enumerate(comments):
            if '結果' in comment:
                result_comment_index = idx
                break

        if result_comment_index is not None and result_comment_index + 2 < len(comments):
            # コメントの次の次の行を取得
            result_soup = BeautifulSoup(comments[result_comment_index + 1], 'html.parser')
            p_tag = result_soup.find('p', class_='h4')
            if p_tag:
                race_result = p_tag.text.strip()
                print(f"Race result: {race_result}")  # デバッグ用出力
        else:
            # コメントの後のタグを直接探索
            comment = comments[result_comment_index]
            parent = comment.find_parent()
            next_sibling = parent.find_next_sibling()
            if next_sibling and next_sibling.name == 'div' and 'id' in next_sibling.attrs and next_sibling['id'] == 'detail_result':
                p_tag = next_sibling.find('p', class_='h4')
                if p_tag:
                    race_result = p_tag.text.strip()
                    print(f"Race result: {race_result}")  # デバッグ用出力
    except Exception as e:
        print(f"Failed to parse race result: {e}")
        race_result = None  # 結果が未確定の場合はNone

    return predictions, race_result

# 資金配分を計算する関数（変更なし）
def calculate_bet_allocation(predictions, total_funds, minimum_bet):
    top_predictions = predictions[:10]
    total_inverse_odds = sum([1/p['odds'] for p in top_predictions if p['odds'] > 0])
    bet_allocations = []
    total_bet_amount = 0

    for prediction in top_predictions:
        odds = prediction['odds']
        trifecta = prediction['trifecta']

        if odds <= 0:
            # オッズが無効な場合、全額返金
            bet_amount = total_funds  # 全額返金
            refund_amount = total_funds
            bet_allocations.append({
                'trifecta': trifecta,
                'bet_amount': bet_amount,
                'odds': odds,
                'refund_amount': refund_amount
            })
            return bet_allocations  # 返金がある場合、処理を終了

        # 資金配分: 的中時に同じ配当になるように資金を配分
        allocated_fund = (total_funds / odds) / total_inverse_odds
        bet_amount = max(math.floor(allocated_fund / minimum_bet) * minimum_bet, minimum_bet)
        total_bet_amount += bet_amount

        bet_allocations.append({
            'trifecta': trifecta,
            'bet_amount': bet_amount,
            'odds': odds,
            'refund_amount': 0
        })

    remaining_funds = total_funds - total_bet_amount

    # 資金配分の調整
    idx = 0
    while remaining_funds >= minimum_bet:
        bet_allocations[idx % len(bet_allocations)]['bet_amount'] += minimum_bet
        remaining_funds -= minimum_bet
        idx += 1

    return bet_allocations

# 回収率を計算する関数（変更なし）
def calculate_return_rate(bet_allocations, race_result):
    total_investment = sum([bet['bet_amount'] for bet in bet_allocations])
    payout = 0

    for bet in bet_allocations:
        if bet['trifecta'] == race_result:
            payout += bet['bet_amount'] * bet['odds']
            break

    return_rate = payout / total_investment if total_investment > 0 else 1  # 投資が全額返金の場合
    return return_rate, payout, total_investment

# 結果をファイルに保存する関数（変更なし）
def save_results_to_file(results, filename="bet_results.txt"):
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
                f.write(f"  Trifecta: {bet['trifecta']}, Bet Amount: {bet['bet_amount']} 円, Odds: {bet['odds']}\n")
            f.write("\n")

# 結果をコンソールに出力する関数（変更なし）
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
            print(f"  Trifecta: {bet['trifecta']}, Bet Amount: {bet['bet_amount']} 円, Odds: {bet['odds']}")
        print("-" * 40)

def main():
    race_links = get_gachigachi_race_links()
    print("Race links:", race_links)  # デバッグ用出力
    all_results = []

    for race_url in race_links:
        race_parts = race_url.strip('/').split('/')
        race_date = race_parts[-4]  # 日付
        race_venue_code = race_parts[-2]  # 競艇場番号
        race_number = race_parts[-1].replace('R', '')  # レース番号

        # 競艇場コードを名前に変換
        race_name = RACE_VENUE_MAPPING.get(race_venue_code, f"Unknown Venue ({race_venue_code})")

        print(f"Processing Race {race_number} at Venue {race_name} on {race_date}...")

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
        return_rate, payout, total_investment = calculate_return_rate(bet_allocations, race_result)

        race_result_str = race_result if race_result else "Not Available"

        # 結果を表示
        print(f"    Total Investment: {total_investment} 円")
        print(f"    Payout: {payout} 円")
        print(f"    Return Rate: {return_rate * 100:.2f}%")
        print(f"    Result: {race_result_str}")
        print("    Bet Allocations:")
        for bet in bet_allocations:
            print(f"      Trifecta: {bet['trifecta']}, Bet Amount: {bet['bet_amount']} 円, Odds: {bet['odds']}")
        print("-" * 40)

        result = {
            'race_name': race_name,
            'race_number': race_number,
            'bet_allocations': bet_allocations,
            'return_rate': return_rate,
            'payout': payout,
            'total_investment': total_investment,
            'race_result': race_result_str
        }

        all_results.append(result)
        # サーバーに負荷をかけないように待機
        time.sleep(1)

    # 結果をファイルに保存
    save_results_to_file(all_results)
    # 結果をコンソールに出力
    print_results(all_results)

if __name__ == "__main__":
    main()
