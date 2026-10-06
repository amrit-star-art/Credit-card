"""SmartCard AI - Personalized Credit Card Recommendation System."""
import os
import joblib
import pandas as pd
import streamlit as st
from preprocessing import load_dataset, BASE_FEATURES
from recommendation_engine import build_recommendations

st.set_page_config(page_title="SmartCard AI", page_icon="💳", layout="wide")

st.title("💳 SmartCard AI")
st.subheader("Personalized Credit Card Recommendation System")
st.caption("Model-based recommendation for demonstration purposes. It is not financial advice or a guarantee of approval.")

REQUIRED = ["credit_cards.csv", "knn_bundle.pkl", "decision_tree.pkl"]
missing = [f for f in REQUIRED if not os.path.exists(f)]
if missing:
    st.error("Missing required files: " + ", ".join(missing))
    st.stop()

cards = load_dataset()
knn_bundle = joblib.load("knn_bundle.pkl")
tree = joblib.load("decision_tree.pkl")

st.sidebar.header("Your Profile")
income = st.sidebar.number_input("Monthly Income (₹)", min_value=0.0, value=50000.0, step=5000.0)
total_spending = st.sidebar.number_input("Monthly Total Spending (₹)", min_value=0.0, value=30000.0, step=1000.0)
shopping = st.sidebar.number_input("Shopping (₹)", min_value=0.0, value=7000.0, step=500.0)
dining = st.sidebar.number_input("Dining (₹)", min_value=0.0, value=4000.0, step=500.0)
travel = st.sidebar.number_input("Travel (₹)", min_value=0.0, value=5000.0, step=500.0)
fuel = st.sidebar.number_input("Fuel (₹)", min_value=0.0, value=3000.0, step=500.0)
online = st.sidebar.number_input("Online Shopping (₹)", min_value=0.0, value=5000.0, step=500.0)
max_fee = st.sidebar.number_input("Maximum Annual Fee (₹)", min_value=0.0, value=2000.0, step=500.0)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 750)
lounge = st.sidebar.selectbox("Lounge Access Importance", ["Low", "Medium", "High"])
preferred = st.sidebar.selectbox(
    "Preferred Benefit",
    ["Cashback", "Travel", "Fuel", "Dining", "Shopping", "Rewards", "Low-Fee / No-Fee"]
)

spending_sum = shopping + dining + travel + fuel + online
if spending_sum > total_spending:
    st.warning("Category spending is greater than total monthly spending. Please adjust the values.")
    st.stop()

# Convert user inputs into the same feature space used during model training.
user_profile = {
    "annual_fee": max_fee,
    "minimum_income": income * 12,
    "credit_score_requirement": credit_score,
    "cashback_rate": 5 if preferred == "Cashback" else 1,
    "reward_rate": 5 if preferred == "Rewards" else 1,
    "travel_reward": 5 if preferred == "Travel" else 1,
    "dining_reward": 5 if preferred == "Dining" else 1,
    "fuel_reward": 5 if preferred == "Fuel" else 1,
    "shopping_reward": 5 if preferred == "Shopping" else 1,
    "online_reward": 5 if preferred == "Shopping" else 1,
    "lounge_access": {"Low": 0, "Medium": 1, "High": 2}[lounge],
    "welcome_bonus": 5000,
    "foreign_transaction_fee": 2,
    "max_annual_fee": max_fee,
    "spending": {
        "shopping": shopping, "dining": dining, "travel": travel,
        "fuel": fuel, "online": online
    },
}

X_user = pd.DataFrame([{feature: user_profile[feature] for feature in BASE_FEATURES}])
knn_pred = knn_bundle["model"].predict(knn_bundle["scaler"].transform(X_user))[0]
tree_pred = tree.predict(X_user)[0]

# User preference is used as the practical recommendation category when it exists.
predicted_category = preferred if preferred in cards["card_category"].unique() else knn_pred

results = build_recommendations(
    cards, user_profile, predicted_category=predicted_category, top_n=3
)

st.success(f"Recommended category: **{predicted_category}**")

col1, col2 = st.columns(2)
with col1:
    st.metric("KNN Predicted Category", str(knn_pred))
with col2:
    st.metric("Decision Tree Predicted Category", str(tree_pred))

st.divider()
st.header("🏆 Top 3 Recommended Cards")

for i, row in results.iterrows():
    with st.container(border=True):
        st.subheader(f"{i + 1}. {row['card_name']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Match Score", f"{row['recommendation_match_score'] * 100:.1f}%")
        c2.metric("Annual Fee", f"₹{row['annual_fee']:,.0f}")
        c3.metric("Est. Annual Rewards", f"₹{row['estimated_annual_rewards']:,.0f}")
        c4.metric("Est. Net Benefit", f"₹{row['estimated_net_benefit']:,.0f}")
        st.write("**Why it matches:**", row["explanation"])

with st.expander("How the system works"):
    st.write("1. KNN predicts a suitable card category from the user profile.")
    st.write("2. Decision Tree provides a second category prediction for comparison.")
    st.write("3. Cosine similarity compares the user profile with each card.")
    st.write("4. The recommendation score combines similarity, category match, rewards, fee suitability, and spending-category match.")
    st.write("5. The top three cards are displayed with estimated rewards and net benefit.")

st.caption("Estimated rewards are based on the demo dataset and the spending entered above.")
