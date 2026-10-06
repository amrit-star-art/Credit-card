"""Content-based credit-card recommendation engine using cosine similarity."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from reward_calculator import calculate_annual_rewards, calculate_net_benefit
from preprocessing import BASE_FEATURES

def _safe_scale_user(user_profile, cards):
    values = []
    for feature in BASE_FEATURES:
        values.append(float(user_profile.get(feature, cards[feature].median())))
    return np.array(values, dtype=float).reshape(1, -1)

def build_recommendations(cards, user_profile, predicted_category=None, top_n=3):
    data = cards.copy()
    matrix = data[BASE_FEATURES].astype(float).fillna(0)
    user_vector = _safe_scale_user(user_profile, data)

    # Cosine similarity is used for profile-to-card matching.
    similarity = cosine_similarity(user_vector, matrix)[0]
    data["profile_similarity"] = np.clip((similarity + 1) / 2, 0, 1)

    preferred_fee = float(user_profile.get("annual_fee", 0))
    max_fee = float(user_profile.get("max_annual_fee", preferred_fee))
    data["fee_match"] = np.where(
        data["annual_fee"] <= max_fee, 1.0,
        np.maximum(0.0, 1 - (data["annual_fee"] - max_fee) / np.maximum(max_fee, 1))
    )

    data["category_match"] = (
        data["card_category"].astype(str).str.lower()
        .eq(str(predicted_category).lower() if predicted_category else "")
        .astype(float)
    )

    spending = user_profile.get("spending", {})
    reward_scores = []
    category_scores = []
    reasons = []

    for _, row in data.iterrows():
        annual_rewards = calculate_annual_rewards(spending, row)
        net_benefit = calculate_net_benefit(annual_rewards, row["annual_fee"])
        reward_scores.append((annual_rewards, net_benefit))

        active = []
        checks = [
            ("travel", "travel_reward", "travel"),
            ("dining", "dining_reward", "dining"),
            ("fuel", "fuel_reward", "fuel"),
            ("shopping", "shopping_reward", "shopping"),
            ("online", "online_reward", "online"),
        ]
        for spend_key, rate_col, label in checks:
            if float(spending.get(spend_key, 0)) > 0 and float(row.get(rate_col, 0)) > 0:
                active.append(label)

        category_scores.append(min(1.0, len(active) / 2))
        reason = []
        if predicted_category and str(row["card_category"]).lower() == str(predicted_category).lower():
            reason.append(f"matches the predicted {predicted_category} category")
        if row["annual_fee"] <= max_fee:
            reason.append("fits your annual-fee preference")
        if active:
            reason.append("supports your " + ", ".join(active) + " spending")
        if not reason:
            reason.append("has a relatively close overall profile match")
        reasons.append("; ".join(reason))

    data["estimated_annual_rewards"] = [x[0] for x in reward_scores]
    data["estimated_net_benefit"] = [x[1] for x in reward_scores]
    data["spending_category_match"] = category_scores
    data["reward_suitability"] = np.clip(
        data["estimated_annual_rewards"] / max(data["estimated_annual_rewards"].max(), 1), 0, 1
    )
    data["recommendation_match_score"] = (
        0.35 * data["profile_similarity"]
        + 0.25 * data["category_match"]
        + 0.20 * data["reward_suitability"]
        + 0.10 * data["fee_match"]
        + 0.10 * data["spending_category_match"]
    )
    data["explanation"] = reasons

    return data.sort_values("recommendation_match_score", ascending=False).head(top_n).reset_index(drop=True)
