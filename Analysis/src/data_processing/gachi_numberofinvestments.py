import json
import math

# 資金配分を計算する関数
def calculate_bet_allocation(predictions, total_funds, minimum_bet, top_n):
    top_predictions = predictions[:top_n]  # 上位N個の予想を使用
    inverse_odds_sum = sum([1 / p['odds'] for p in top_predictions if p['odds'] > 0])
    
    if inverse_odds_sum == 0 or not top_predictions:
        # オッズが無効な場合、または予想がない場合、空のリストを返す
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
    # JSONデータを読み込み
    with open("race_data_threshold_0.json", "r", encoding="utf-8") as f:
        all_race_data = json.load(f)

    top_n_list = [1, 2, 3, 5, 10, 20]
    results = []

    for top_n in top_n_list:
        total_investment_all = 0
        total_payout_all = 0
        invested_races = 0
        hit_races = 0

        for race in all_race_data:
            predictions = race['predictions']
            race_date = race['race_date']
            race_number = race['race_number']
            race_result = race.get('race_result')
            print(race_result)
            # レース結果がない場合はスキップ
            if not race_result or race_result == "Not Available":
                continue

            total_funds = 10000
            minimum_bet = 100

            bet_allocations = calculate_bet_allocation(predictions, total_funds, minimum_bet, top_n)
            if not bet_allocations:
                continue  # 資金配分ができない場合はスキップ

            payout, total_investment = calculate_return_rate(bet_allocations, race_result)

            total_investment_all += total_investment
            total_payout_all += payout
            invested_races += 1
            if payout > 0:
                hit_races += 1

        overall_return_rate = (total_payout_all / total_investment_all) * 100 if total_investment_all > 0 else 0
        results.append({
            'top_n': top_n,
            'total_investment': total_investment_all,
            'total_payout': total_payout_all,
            'overall_return_rate': overall_return_rate,
            'invested_races': invested_races,
            'hit_races': hit_races
        })

    # 結果の表示
    print("Top N Predictions Analysis Results:")
    print(f"{'Top N':>6} | {'Investment':>12} | {'Payout':>10} | {'Return Rate':>12} | {'Invested Races':>15} | {'Hit Races':>10}")
    print("-" * 80)
    max_return_rate = -1
    best_top_n = None
    for result in results:
        top_n = result['top_n']
        total_investment = result['total_investment']
        total_payout = result['total_payout']
        overall_return_rate = result['overall_return_rate']
        invested_races = result['invested_races']
        hit_races = result['hit_races']

        print(f"{top_n:>6} | {total_investment:>12,.0f} | {total_payout:>10,.0f} | {overall_return_rate:>11.2f}% | {invested_races:>15} | {hit_races:>10}")

        if overall_return_rate > max_return_rate:
            max_return_rate = overall_return_rate
            best_top_n = top_n

    print("\nMaximum Return Rate Achieved with Top N = {} Predictions, Return Rate: {:.2f}%".format(best_top_n, max_return_rate))

if __name__ == "__main__":
    main()
