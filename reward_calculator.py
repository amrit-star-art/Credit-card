"""Reward and net-benefit calculations."""
def calculate_annual_rewards(spending, card):
    category_map = {
        "cashback_rate": "shopping",
        "reward_rate": "online",
        "travel_reward": "travel",
        "dining_reward": "dining",
        "fuel_reward": "fuel",
        "shopping_reward": "shopping",
        "online_reward": "online",
    }
    total = 0.0
    for card_col, spend_key in category_map.items():
        rate = float(card.get(card_col, 0) or 0)
        amount = float(spending.get(spend_key, 0) or 0)
        total += amount * 12 * rate / 100
    return round(total, 2)

def calculate_net_benefit(annual_rewards, annual_fee):
    return round(float(annual_rewards) - float(annual_fee), 2)
