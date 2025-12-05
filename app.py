import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================
# 1. LOAD DATA
# ============================================

@st.cache_data
def load_data():
    df = pd.read_csv("hustler_cross_channel_simulated_v2.csv")
    return df

df = load_data()
channels = df["channel"].unique().tolist()

st.title("AI Agent for Cross-Channel Marketing Optimization")
st.caption("Prototype built for Hustler (simulated data, D2C fitness accessories)")

st.subheader("Historical Data Snapshot")
st.dataframe(df.head(12))

# ============================================
# 2. TRAIN CHANNEL MODELS
# conversions ~ log(spend) + week + seasonality
# ============================================

@st.cache_resource
def train_models(df_input):
    dfm = df_input.copy()
    dfm["log_spend"] = np.log1p(dfm["spend_inr"])

    channel_models = {}
    metrics = []

    for ch in channels:
        df_ch = dfm[dfm["channel"] == ch].copy()
        X = df_ch[["log_spend", "week", "seasonality_factor"]]
        y = df_ch["conversions"]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        channel_models[ch] = model
        metrics.append({"channel": ch, "MAE": mae, "R2": r2})

    metrics_df = pd.DataFrame(metrics)
    return channel_models, metrics_df

channel_models, metrics_df = train_models(df)

st.subheader("Model Performance by Channel")
st.dataframe(metrics_df.style.format({"MAE": "{:.2f}", "R2": "{:.3f}"}))

# ============================================
# 3. PREDICTION ENGINE
# ============================================

def predict_total_conversions(spend_dict, week, seasonality_factor=1.0):
    total = 0.0
    for ch in channels:
        model = channel_models[ch]
        s = spend_dict[ch]
        log_s = np.log1p(s)
        X = np.array([[log_s, week, seasonality_factor]])
        pred = model.predict(X)[0]
        total += max(pred, 0)
    return total


def predict_conversions_by_channel(spend_dict, week, seasonality_factor=1.0):
    conv_dict = {}
    for ch in channels:
        model = channel_models[ch]
        s = spend_dict[ch]
        log_s = np.log1p(s)
        X = np.array([[log_s, week, seasonality_factor]])
        pred = model.predict(X)[0]
        conv_dict[ch] = max(pred, 0)
    return conv_dict

# ============================================
# 4. BASELINE MIX FROM HISTORICAL DATA
# ============================================

# use last 4 weeks as "current plan"
recent_weeks = df["week"].max() - 3
df_recent = df[df["week"] >= recent_weeks]

baseline_spend_hist = df_recent.groupby("channel")["spend_inr"].mean().to_dict()
baseline_total_budget_hist = sum(baseline_spend_hist.values())

baseline_mix = {ch: baseline_spend_hist[ch] / baseline_total_budget_hist for ch in channels}

BASE_TOTAL_BUDGET = 160000  # reference budget used for caps

channel_min = {
    "Google Search": 5000,
    "Instagram Ads": 8000,
    "Facebook Ads": 5000,
    "YouTube Ads": 5000,
    "Influencer Marketing": 8000,
    "Email": 3000
}

channel_max = {
    "Google Search": 60000,
    "Instagram Ads": 60000,
    "Facebook Ads": 60000,
    "YouTube Ads": 50000,
    "Influencer Marketing": 45000,
    "Email": 15000
}

# ============================================
# 5. STREAMLIT SIDEBAR INPUTS
# ============================================

st.sidebar.header("Optimization Settings")

budget = st.sidebar.number_input(
    "Weekly budget (INR)",
    min_value=50000,
    max_value=400000,
    value=160000,
    step=5000
)

weeks_horizon = st.sidebar.selectbox(
    "Plan for how many weeks?",
    [1, 2, 4],
    index=0,
    format_func=lambda x: f"{x} week" if x == 1 else f"{x} weeks"
)

seasonality_label = st.sidebar.selectbox(
    "Seasonality scenario",
    ["Normal week (1.0)", "Promo week (1.25)", "Slow week (0.8)"],
    index=0
)

if "1.25" in seasonality_label:
    seasonality_factor = 1.25
elif "0.8" in seasonality_label:
    seasonality_factor = 0.8
else:
    seasonality_factor = 1.0

st.sidebar.write("---")
run_opt = st.sidebar.button("Run AI Optimization")

# ============================================
# 6. OPTIMIZATION FUNCTION
# ============================================

def optimize_budget(total_budget, week, seasonality_factor):

    def objective(spend_array):
        spend_dict = {channels[i]: spend_array[i] for i in range(len(channels))}
        return -predict_total_conversions(spend_dict, week, seasonality_factor)

    # bounds scaled with budget
    bounds = [
        (channel_min[ch] * total_budget / BASE_TOTAL_BUDGET,
         channel_max[ch] * total_budget / BASE_TOTAL_BUDGET)
        for ch in channels
    ]

    constraints = ({
        "type": "eq",
        "fun": lambda arr: arr.sum() - total_budget
    })

    # start from historical baseline mix
    x0 = np.array([total_budget * baseline_mix[ch] for ch in channels])

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    opt_spend = result.x
    opt_alloc = {channels[i]: float(opt_spend[i]) for i in range(len(channels))}
    opt_conversions = -float(result.fun)

    return opt_alloc, opt_conversions

# ============================================
# 7. RUN OPTIMIZATION AND DISPLAY RESULTS
# ============================================

if run_opt:

    st.subheader("AI-Optimized Budget Allocation")

    # internal "next week" index (user never sees this)
    internal_week = df["week"].max() + 1

    # optimized allocation for one week
    opt_alloc, opt_conv_total_one_week = optimize_budget(
        budget, internal_week, seasonality_factor
    )

    # baseline allocation scaled to same weekly budget
    baseline_spend_scaled = {ch: budget * baseline_mix[ch] for ch in channels}
    base_conv_dict_one_week = predict_conversions_by_channel(
        baseline_spend_scaled, internal_week, seasonality_factor
    )
    opt_conv_dict_one_week = predict_conversions_by_channel(
        opt_alloc, internal_week, seasonality_factor
    )

    # scale conversions by planning horizon
    base_total_conv = sum(base_conv_dict_one_week.values()) * weeks_horizon
    opt_total_conv = opt_conv_total_one_week * weeks_horizon

    uplift_pct = (
        (opt_total_conv - base_total_conv) / base_total_conv * 100
        if base_total_conv > 0 else 0.0
    )

    rows = []
    for ch in channels:
        rows.append({
            "channel": ch,
            "baseline_spend_inr": round(baseline_spend_scaled[ch], 2),
            "optimized_spend_inr": round(opt_alloc[ch], 2),
            "baseline_conversions_per_week": round(base_conv_dict_one_week[ch], 1),
            "optimized_conversions_per_week": round(opt_conv_dict_one_week[ch], 1)
        })

    compare_df = pd.DataFrame(rows)
    compare_df["delta_spend_inr"] = compare_df["optimized_spend_inr"] - compare_df["baseline_spend_inr"]
    compare_df["delta_conversions_per_week"] = (
        compare_df["optimized_conversions_per_week"] - compare_df["baseline_conversions_per_week"]
    )

    st.write(f"**Planning horizon:** {weeks_horizon} week(s)")
    st.write(f"**Predicted baseline conversions over horizon:** {base_total_conv:.1f}")
    st.write(f"**Predicted optimized conversions over horizon:** {opt_total_conv:.1f}")
    st.write(f"**Expected uplift:** {uplift_pct:.1f}%")

    st.dataframe(
        compare_df.style.format({
            "baseline_spend_inr": "₹{:,.0f}",
            "optimized_spend_inr": "₹{:,.0f}",
            "delta_spend_inr": "₹{:+,.0f}",
            "baseline_conversions_per_week": "{:.1f}",
            "optimized_conversions_per_week": "{:.1f}",
            "delta_conversions_per_week": "{:+.1f}"
        })
    )

    # Spend chart
    st.subheader("Spend Comparison")
    spend_chart_df = compare_df.melt(
        id_vars="channel",
        value_vars=["baseline_spend_inr", "optimized_spend_inr"],
        var_name="type", value_name="spend_inr"
    )
    st.bar_chart(
        data=spend_chart_df,
        x="channel",
        y="spend_inr",
        color="type"
    )

    # Conversions chart (per week)
    st.subheader("Conversions Comparison (per week)")
    conv_chart_df = compare_df.melt(
        id_vars="channel",
        value_vars=["baseline_conversions_per_week", "optimized_conversions_per_week"],
        var_name="type", value_name="conversions"
    )
    st.bar_chart(
        data=conv_chart_df,
        x="channel",
        y="conversions",
        color="type"
    )

else:
    st.info("Set your weekly budget and planning horizon, then click 'Run AI Optimization' in the sidebar.")
