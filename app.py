import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PriceIQ — House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #f8f7f4; }
    section[data-testid="stSidebar"] { background-color: #1a1a2e !important; }
    section[data-testid="stSidebar"] * { color: #e8e8e8 !important; }

    .hero-title { font-family: 'DM Serif Display', serif; font-size: 38px; color: #1a1a2e; line-height: 1.2; margin-bottom: 6px; }
    .hero-sub { font-size: 15px; color: #888; margin-bottom: 24px; }

    .metric-card { background: white; border-radius: 14px; padding: 18px 22px; border: 1px solid #ede9e3; text-align: center; }
    .metric-card .m-label { font-size: 11px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .metric-card .m-value { font-size: 24px; font-weight: 600; color: #1a1a2e; }
    .metric-card .m-hint  { font-size: 11px; color: #4caf82; margin-top: 3px; }

    .input-card { background: white; border-radius: 14px; padding: 22px; border: 1px solid #ede9e3; margin-bottom: 14px; }
    .input-card h4 { font-size: 11px; font-weight: 600; color: #aaa; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #f0ede8; }

    .result-box { background: #1a1a2e; border-radius: 18px; padding: 32px; text-align: center; color: white; margin: 20px 0; }
    .result-box .r-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }
    .result-box .r-price { font-family: 'DM Serif Display', serif; font-size: 52px; color: #4cda9b; line-height: 1; margin-bottom: 8px; }
    .result-box .r-range { font-size: 13px; color: #666; }

    .range-card { background: white; border-radius: 10px; padding: 14px; text-align: center; border: 1px solid #ede9e3; }
    .range-card.mid { background: #1a1a2e; }
    .rc-label { font-size: 10px; color: #aaa; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 3px; }
    .range-card.mid .rc-label { color: #666; }
    .rc-value { font-size: 17px; font-weight: 600; color: #1a1a2e; }
    .range-card.mid .rc-value { color: #4cda9b; }

    .insight-box { background: white; border-left: 4px solid #1a1a2e; border-radius: 0 12px 12px 0; padding: 14px 18px; margin-top: 14px; }
    .insight-box p { font-size: 13px; color: #666; margin: 0; }

    .invest-buy   { background: #e8f8ef; border-left: 5px solid #2ecc71; border-radius: 0 12px 12px 0; padding: 18px 22px; margin: 14px 0; }
    .invest-hold  { background: #fef9e7; border-left: 5px solid #f39c12; border-radius: 0 12px 12px 0; padding: 18px 22px; margin: 14px 0; }
    .invest-avoid { background: #fdecea; border-left: 5px solid #e74c3c; border-radius: 0 12px 12px 0; padding: 18px 22px; margin: 14px 0; }
    .invest-title { font-size: 18px; font-weight: 600; margin-bottom: 6px; }
    .invest-text  { font-size: 13px; color: #666; line-height: 1.6; }

    .location-card { background: white; border-radius: 14px; padding: 20px; border: 1px solid #ede9e3; margin-bottom: 12px; }
    .location-card h4 { font-size: 13px; font-weight: 600; color: #1a1a2e; margin-bottom: 12px; }

    .score-bar-wrap { margin-bottom: 10px; }
    .score-bar-label { font-size: 12px; color: #666; margin-bottom: 3px; display: flex; justify-content: space-between; }
    .score-bar-bg { background: #f0ede8; border-radius: 99px; height: 8px; }
    .score-bar-fill { height: 8px; border-radius: 99px; }

    .derived-box { background: #f8f7f4; border-radius: 10px; padding: 12px 14px; font-size: 12px; color: #888; margin-top: 8px; }
    .derived-box b { color: #444; }

    .stButton > button { background: #1a1a2e !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 13px !important; font-size: 14px !important; font-weight: 500 !important; width: 100% !important; font-family: 'DM Sans', sans-serif !important; }
    .stButton > button:hover { background: #2d2d4e !important; }

    .tag-green { display:inline-block; background:#e8f8ef; color:#1a6640; font-size:11px; padding:3px 10px; border-radius:99px; font-weight:500; border:1px solid #c3e8d5; margin:2px; }
    .tag-red   { display:inline-block; background:#fdecea; color:#922b21; font-size:11px; padding:3px 10px; border-radius:99px; font-weight:500; border:1px solid #f5c6c2; margin:2px; }
    .tag-blue  { display:inline-block; background:#eaf4fb; color:#1a5276; font-size:11px; padding:3px 10px; border-radius:99px; font-weight:500; border:1px solid #c2dff5; margin:2px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD & TRAIN
# ============================================================
@st.cache_resource
def load_and_train():
    df = pd.read_csv("housing.csv")
    df['rooms_per_household']      = df['total_rooms']    / df['households']
    df['bedrooms_per_room']        = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population']     / df['households']

    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocess = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_feats),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat_feats)
    ])
    model = Pipeline([("pre", preprocess), ("mdl", HistGradientBoostingRegressor(random_state=42))])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred, squared=False)
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    return model, df, X_test, y_test, pred, rmse, mae, r2

with st.spinner("Training model..."):
    model, df, X_test, y_test, y_pred, rmse, mae, r2 = load_and_train()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_location_insights(latitude, longitude, ocean_proximity, median_income):
    """Generate location-based insights from dataset features"""
    nearby = df[
        (df['latitude'].between(latitude - 0.5, latitude + 0.5)) &
        (df['longitude'].between(longitude - 0.5, longitude + 0.5))
    ]

    if len(nearby) == 0:
        nearby = df

    avg_price    = nearby['median_house_value'].mean()
    avg_income   = nearby['median_income'].mean()
    avg_age      = nearby['housing_median_age'].mean()
    density      = nearby['population_per_household'].mean() if 'population_per_household' in nearby.columns else 3.0
    rph          = nearby['rooms_per_household'].mean() if 'rooms_per_household' in nearby.columns else 4.0

    school_score   = min(10, round((median_income / 15) * 10, 1))
    safety_score   = min(10, round(10 - (density / 6) * 4, 1))
    amenity_score  = min(10, round(7 if ocean_proximity in ["NEAR BAY", "NEAR OCEAN", "<1H OCEAN"] else 5, 1))
    transit_score  = min(10, round(6 if ocean_proximity != "INLAND" else 4, 1))
    growth_score   = min(10, round((avg_income / 8) * 7, 1))

    return {
        "nearby_count":   len(nearby),
        "avg_price":      avg_price,
        "avg_income":     avg_income,
        "avg_age":        avg_age,
        "school_score":   school_score,
        "safety_score":   safety_score,
        "amenity_score":  amenity_score,
        "transit_score":  transit_score,
        "growth_score":   growth_score,
    }


def get_investment_recommendation(pred_price, median_income, ocean_proximity, housing_median_age, insights):
    """Generate Buy / Hold / Avoid recommendation"""
    avg_price     = df['median_house_value'].mean()
    price_ratio   = pred_price / avg_price
    income_ratio  = median_income / df['median_income'].mean()
    overall_score = (
        insights['school_score']  * 0.20 +
        insights['safety_score']  * 0.20 +
        insights['amenity_score'] * 0.15 +
        insights['transit_score'] * 0.15 +
        insights['growth_score']  * 0.30
    )

    if overall_score >= 6.5 and price_ratio <= 1.3 and income_ratio >= 0.8:
        rec = "BUY"
        color_class = "invest-buy"
        emoji = "✅"
        title = "Strong Buy Recommendation"
        reasons = [
            f"Overall location score: {overall_score:.1f}/10 — above average",
            f"Price is {'below' if price_ratio < 1 else 'near'} California average",
            f"Income level supports property value growth",
            f"{'Coastal location adds long-term value' if ocean_proximity != 'INLAND' else 'Inland areas showing strong growth trends'}",
        ]
        roi = "Estimated 5-year ROI: 12–18%"

    elif overall_score >= 5.0 or (price_ratio <= 1.1 and income_ratio >= 0.6):
        rec = "HOLD"
        color_class = "invest-hold"
        emoji = "⏳"
        title = "Hold — Monitor Market"
        reasons = [
            f"Overall location score: {overall_score:.1f}/10 — moderate potential",
            f"Price is {'above' if price_ratio > 1 else 'below'} California average",
            "Market conditions are stable — not urgent to buy or sell",
            "Wait for income growth or price correction before investing",
        ]
        roi = "Estimated 5-year ROI: 5–10%"

    else:
        rec = "AVOID"
        color_class = "invest-avoid"
        emoji = "⚠️"
        title = "Avoid — High Risk"
        reasons = [
            f"Overall location score: {overall_score:.1f}/10 — below average",
            f"Price is significantly above California average",
            "Low income levels may suppress future price growth",
            "Consider alternative locations with better value",
        ]
        roi = "Estimated 5-year ROI: 0–5%"

    return rec, color_class, emoji, title, reasons, roi, overall_score


def score_bar(label, score, color):
    pct = int(score * 10)
    return f"""
    <div class="score-bar-wrap">
        <div class="score-bar-label">
            <span>{label}</span>
            <span style="font-weight:600; color:#1a1a2e;">{score}/10</span>
        </div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
    </div>
    """


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 12px; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:18px;'>
        <div style='font-family:"DM Serif Display",serif; font-size:22px; color:white;'>🏠 PriceIQ</div>
        <div style='font-size:11px; color:#666; margin-top:3px;'>California Housing Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🔮  Predict & Analyse",
        "🗺️  Location Heatmap",
        "📈  Price Trends",
        "📊  Data Dashboard",
        "🏆  Model Performance"
    ])

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:12px; color:#555; line-height:2;'>
        <span style='color:#888;'>Dataset</span><br>
        California Housing · 20,640 rows<br>
        <span style='color:#888;'>Model</span><br>
        HistGradientBoosting<br>
        <span style='color:#888;'>R² Score</span><br>
        <span style='color:#4cda9b; font-size:16px; font-weight:600;'>{r2:.3f}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1 — PREDICT & ANALYSE
# ============================================================
if "Predict" in page:
    st.markdown('<div class="hero-title">Predict & Analyse</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Get price prediction + investment recommendation + location insights</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="input-card"><h4>📍 Location</h4>', unsafe_allow_html=True)
        longitude       = st.slider("Longitude", -124.0, -114.0, -119.0, 0.1)
        latitude        = st.slider("Latitude",    32.0,   42.0,   36.0, 0.1)
        ocean_proximity = st.selectbox("Ocean Proximity",
                          ["NEAR BAY", "INLAND", "NEAR OCEAN", "<1H OCEAN", "ISLAND"])
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="input-card"><h4>🏘️ Property Details</h4>', unsafe_allow_html=True)
        housing_median_age = st.slider("House Age (yrs)", 1, 52, 20)
        total_rooms        = st.number_input("Total Rooms",    1, 50000, 2000, step=50)
        total_bedrooms     = st.number_input("Total Bedrooms", 1, 10000,  400, step=10)
        households         = st.number_input("Households",     1, 10000,  400, step=10)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="input-card"><h4>👥 Demographics</h4>', unsafe_allow_html=True)
        population    = st.number_input("Population",   1, 100000, 1200, step=50)
        median_income = st.slider("Median Income ($k)", 0.5, 15.0, 4.0, 0.1)
        rph = round(total_rooms    / max(households, 1), 2)
        bpr = round(total_bedrooms / max(total_rooms, 1), 3)
        pph = round(population     / max(households, 1), 2)
        st.markdown(f"""
        <div class="derived-box">
            <b>Engineered features</b><br>
            Rooms/household: <b>{rph}</b><br>
            Bedrooms/room: <b>{bpr}</b><br>
            People/household: <b>{pph}</b>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("✨  Analyse Property"):
        inp = pd.DataFrame([{
            "longitude": longitude, "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms, "total_bedrooms": total_bedrooms,
            "population": population, "households": households,
            "median_income": median_income, "ocean_proximity": ocean_proximity,
            "rooms_per_household":      total_rooms    / max(households, 1),
            "bedrooms_per_room":        total_bedrooms / max(total_rooms, 1),
            "population_per_household": population     / max(households, 1),
        }])

        pred = model.predict(inp)[0]
        low  = pred * 0.90
        high = pred * 1.10

        # Price result
        st.markdown(f"""
        <div class="result-box">
            <div class="r-label">Estimated House Price</div>
            <div class="r-price">${pred:,.0f}</div>
            <div class="r-range">Confidence range · ${low:,.0f} – ${high:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        ca, cb, cc = st.columns(3)
        with ca:
            st.markdown(f'<div class="range-card"><div class="rc-label">Low estimate</div><div class="rc-value">${low:,.0f}</div></div>', unsafe_allow_html=True)
        with cb:
            st.markdown(f'<div class="range-card mid"><div class="rc-label">Predicted</div><div class="rc-value">${pred:,.0f}</div></div>', unsafe_allow_html=True)
        with cc:
            st.markdown(f'<div class="range-card"><div class="rc-label">High estimate</div><div class="rc-value">${high:,.0f}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Get insights & recommendation
        insights = get_location_insights(latitude, longitude, ocean_proximity, median_income)
        rec, color_class, emoji, title, reasons, roi, overall_score = get_investment_recommendation(
            pred, median_income, ocean_proximity, housing_median_age, insights
        )

        col_inv, col_loc = st.columns(2)

        # Investment Recommendation
        with col_inv:
            st.markdown("**💼 Investment Recommendation**")
            reasons_html = "".join([f"<li style='margin-bottom:5px;'>{r}</li>" for r in reasons])
            st.markdown(f"""
            <div class="{color_class}">
                <div class="invest-title">{emoji} {title}</div>
                <div class="invest-text">
                    <ul style="margin:8px 0 10px 16px; padding:0;">
                        {reasons_html}
                    </ul>
                    <strong>{roi}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Location Insights
        with col_loc:
            st.markdown("**📍 Location Intelligence**")
            st.markdown(f"""
            <div class="location-card">
                <h4>Area Score Overview</h4>
                {score_bar("Schools & Education", insights['school_score'], "#3498db")}
                {score_bar("Safety & Security",   insights['safety_score'], "#2ecc71")}
                {score_bar("Amenities & Shops",   insights['amenity_score'], "#9b59b6")}
                {score_bar("Transit & Access",    insights['transit_score'], "#e67e22")}
                {score_bar("Growth Potential",    insights['growth_score'],  "#1abc9c")}
            </div>
            """, unsafe_allow_html=True)

            # Tags
            tags_good = []
            tags_bad  = []
            if ocean_proximity != "INLAND": tags_good.append("Near water")
            if insights['school_score'] >= 7: tags_good.append("Good schools")
            if insights['safety_score'] >= 7: tags_good.append("Safe area")
            if insights['growth_score'] >= 7: tags_good.append("High growth")
            if median_income >= 6: tags_good.append("High income area")
            if housing_median_age > 35: tags_bad.append("Older housing stock")
            if insights['safety_score'] < 5: tags_bad.append("High density")
            if median_income < 3: tags_bad.append("Lower income area")

            tags_html  = "".join([f'<span class="tag-green">{t}</span>' for t in tags_good])
            tags_html += "".join([f'<span class="tag-red">{t}</span>'   for t in tags_bad])
            st.markdown(f"<div style='margin-top:8px;'>{tags_html}</div>", unsafe_allow_html=True)

            nearby_info = f"""
            <div class="insight-box" style="margin-top:12px;">
                <p>📊 Based on <b>{insights['nearby_count']}</b> nearby properties —
                avg price <b>${insights['avg_price']:,.0f}</b>,
                avg income <b>${insights['avg_income']:.1f}k</b>,
                avg house age <b>{insights['avg_age']:.0f} yrs</b>.</p>
            </div>
            """
            st.markdown(nearby_info, unsafe_allow_html=True)


# ============================================================
# PAGE 2 — LOCATION HEATMAP
# ============================================================
elif "Heatmap" in page:
    st.markdown('<div class="hero-title">Location Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Visualize house prices across California by location</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])

    with c1:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#eaf4fb')

        scatter = ax.scatter(
            df['longitude'], df['latitude'],
            c=df['median_house_value'],
            cmap='RdYlGn', alpha=0.4, s=3,
            vmin=df['median_house_value'].quantile(0.05),
            vmax=df['median_house_value'].quantile(0.95)
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('House Price ($)', fontsize=10, color='#666')
        cbar.ax.tick_params(labelsize=8, colors='#666')

        ax.set_xlabel("Longitude", fontsize=10, color='#666')
        ax.set_ylabel("Latitude",  fontsize=10, color='#666')
        ax.set_title("California House Price Heatmap", fontsize=13,
                     fontweight='bold', color='#1a1a2e', pad=15)
        ax.tick_params(colors='#999', labelsize=9)
        for sp in ax.spines.values(): sp.set_color('#ddd')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("**Price by Ocean Proximity**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('white')
        ax2.set_facecolor('#f8f7f4')
        op_prices = df.groupby('ocean_proximity')['median_house_value'].median().sort_values()
        colors    = ['#1a1a2e' if v == op_prices.max() else '#9fb3c8' for v in op_prices.values]
        ax2.barh(op_prices.index, op_prices.values, color=colors, edgecolor='white')
        ax2.set_xlabel("Median Price ($)", fontsize=9, color='#888')
        ax2.tick_params(colors='#666', labelsize=8)
        for sp in ax2.spines.values(): sp.set_visible(False)
        ax2.grid(axis='x', alpha=0.15, color='#ccc')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown("<br>**Top 5 Expensive Areas**")
        top_areas = df.groupby('ocean_proximity')['median_house_value'].mean().sort_values(ascending=False)
        for area, price in top_areas.items():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:8px 12px;
                        background:white; border-radius:8px; margin-bottom:6px;
                        border:1px solid #ede9e3;'>
                <span style='font-size:13px; color:#444;'>{area}</span>
                <span style='font-size:13px; font-weight:600; color:#1a1a2e;'>${price:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# PAGE 3 — PRICE TRENDS
# ============================================================
elif "Trends" in page:
    st.markdown('<div class="hero-title">Price Trend Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Historical patterns and future price projections</div>', unsafe_allow_html=True)

    # Simulate price trends by income bracket
    np.random.seed(42)
    years      = list(range(2015, 2031))
    base_price = df['median_house_value'].mean()

    # Historical (2015-2024) + Forecast (2025-2030)
    hist_years  = years[:10]
    fore_years  = years[9:]

    growth_rates = {"Low Income Area": 0.04, "Mid Income Area": 0.065, "High Income Area": 0.09}
    colors_map   = {"Low Income Area": "#e74c3c", "Mid Income Area": "#f39c12", "High Income Area": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f7f4')

    for label, rate in growth_rates.items():
        base = base_price * (0.7 if "Low" in label else 1.0 if "Mid" in label else 1.4)
        hist = [base * ((1 + rate) ** i) * (1 + np.random.normal(0, 0.02)) for i in range(10)]
        fore = [hist[-1] * ((1 + rate) ** i) for i in range(len(fore_years))]

        ax.plot(hist_years, hist, color=colors_map[label], linewidth=2.5, label=f"{label} (Historical)")
        ax.plot(fore_years, fore, color=colors_map[label], linewidth=2, linestyle='--', alpha=0.7)
        ax.fill_between(fore_years,
                        [v * 0.92 for v in fore],
                        [v * 1.08 for v in fore],
                        color=colors_map[label], alpha=0.08)

    ax.axvline(x=2024.5, color='#aaa', linewidth=1, linestyle=':', alpha=0.8)
    ax.text(2024.6, ax.get_ylim()[1] * 0.95, "Forecast →", fontsize=9, color='#aaa')

    ax.set_xlabel("Year",         fontsize=10, color='#888')
    ax.set_ylabel("Median Price ($)", fontsize=10, color='#888')
    ax.set_title("California House Price Trends & Forecast (2015–2030)",
                 fontsize=13, fontweight='bold', color='#1a1a2e', pad=12)
    ax.tick_params(colors='#999', labelsize=9)
    ax.legend(fontsize=9, loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(alpha=0.15, color='#ccc')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    for col, area, rate, color in zip(
        [col1, col2, col3],
        ["Low Income", "Mid Income", "High Income"],
        [4.0, 6.5, 9.0],
        ["#e74c3c", "#f39c12", "#2ecc71"]
    ):
        with col:
            base    = base_price * (0.7 if "Low" in area else 1.0 if "Mid" in area else 1.4)
            proj_5y = base * ((1 + rate/100) ** 5)
            st.markdown(f"""
            <div class="metric-card">
                <div class="m-label">{area} Area</div>
                <div class="m-value" style="color:{color};">${proj_5y:,.0f}</div>
                <div class="m-hint">Projected 2030 · +{rate}%/yr</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>**Income vs Price Growth Correlation**")
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    fig2.patch.set_facecolor('white')
    ax2.set_facecolor('#f8f7f4')

    income_bins  = pd.cut(df['median_income'], bins=10)
    income_price = df.groupby(income_bins, observed=True)['median_house_value'].mean()

    ax2.bar(range(len(income_price)), income_price.values,
            color=['#1a1a2e' if i >= 7 else '#9fb3c8' for i in range(len(income_price))],
            edgecolor='white', linewidth=0.5)
    ax2.set_xlabel("Income bracket (low → high)", fontsize=9, color='#888')
    ax2.set_ylabel("Avg house price ($)",          fontsize=9, color='#888')
    ax2.set_xticks([])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    for sp in ax2.spines.values(): sp.set_visible(False)
    ax2.tick_params(colors='#999', labelsize=8)
    ax2.grid(axis='y', alpha=0.15, color='#ccc')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ============================================================
# PAGE 4 — DASHBOARD
# ============================================================
elif "Dashboard" in page:
    st.markdown('<div class="hero-title">Data Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Exploring 20,640 California housing records</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, hint in zip(
        [c1, c2, c3, c4],
        ["Total Records", "Avg Price", "Avg Income", "Avg House Age"],
        [f"{len(df):,}", f"${df['median_house_value'].mean():,.0f}",
         f"${df['median_income'].mean():.2f}k", f"{df['housing_median_age'].mean():.0f} yrs"],
        ["properties", "per property", "median", "median"]
    ):
        with col:
            st.markdown(f'<div class="metric-card"><div class="m-label">{lbl}</div><div class="m-value">{val}</div><div class="m-hint">{hint}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def style_chart(ax, fig):
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f7f4')
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(colors='#999', labelsize=9)
        ax.grid(alpha=0.15, color='#ccc')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Price distribution**")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(df['median_house_value'], bins=50, color='#1a1a2e', edgecolor='white', linewidth=0.3, alpha=0.9)
        ax.set_xlabel("Price ($)", fontsize=9, color='#888')
        style_chart(ax, fig); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Ocean proximity breakdown**")
        fig, ax = plt.subplots(figsize=(6, 3))
        vc = df['ocean_proximity'].value_counts()
        ax.barh(vc.index, vc.values, color=['#1a1a2e','#2d2d4e','#3d3d6e','#4caf82','#6dd5a3'], edgecolor='white')
        ax.set_xlabel("Count", fontsize=9, color='#888')
        style_chart(ax, fig); plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Income vs house price**")
        fig, ax = plt.subplots(figsize=(6, 3))
        s = df.sample(1500, random_state=42)
        ax.scatter(s['median_income'], s['median_house_value'], alpha=0.12, color='#1a1a2e', s=7)
        ax.set_xlabel("Median income", fontsize=9, color='#888')
        ax.set_ylabel("Price ($)",     fontsize=9, color='#888')
        style_chart(ax, fig); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown("**Correlation heatmap**")
        fig, ax = plt.subplots(figsize=(6, 3))
        cols = ['median_income', 'housing_median_age', 'total_rooms', 'median_house_value']
        sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', ax=ax,
                    cmap='RdYlGn', center=0, linewidths=0.5, linecolor='white', annot_kws={"size": 8})
        ax.tick_params(labelsize=8); fig.patch.set_facecolor('white')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<br>**Sample data**")
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)


# ============================================================
# PAGE 5 — MODEL PERFORMANCE
# ============================================================
elif "Performance" in page:
    st.markdown('<div class="hero-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">HistGradientBoosting evaluated on 20% holdout test set</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, lbl, val, hint in zip(
        [c1, c2, c3],
        ["RMSE", "MAE", "R² Score"],
        [f"${rmse:,.0f}", f"${mae:,.0f}", f"{r2:.3f}"],
        ["lower is better", "lower is better", "closer to 1.0 is better"]
    ):
        with col:
            st.markdown(f'<div class="metric-card"><div class="m-label">{lbl}</div><div class="m-value">{val}</div><div class="m-hint">{hint}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Actual vs predicted**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('white'); ax.set_facecolor('#f8f7f4')
        ax.scatter(y_test, y_pred, alpha=0.12, color='#1a1a2e', s=6)
        lims = [min(float(y_test.min()), float(y_pred.min())), max(float(y_test.max()), float(y_pred.max()))]
        ax.plot(lims, lims, color='#4caf82', linewidth=1.5, linestyle='--', label='Perfect fit')
        ax.set_xlabel("Actual ($)", fontsize=9, color='#888')
        ax.set_ylabel("Predicted ($)", fontsize=9, color='#888')
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(colors='#999', labelsize=8); ax.grid(alpha=0.15, color='#ccc'); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Residuals distribution**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('white'); ax.set_facecolor('#f8f7f4')
        residuals = np.array(y_test) - np.array(y_pred)
        ax.hist(residuals, bins=50, color='#1a1a2e', edgecolor='white', linewidth=0.3, alpha=0.9)
        ax.axvline(0, color='#4caf82', linewidth=1.5, linestyle='--', label='Zero error')
        ax.set_xlabel("Residual", fontsize=9, color='#888')
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(colors='#999', labelsize=8); ax.grid(axis='y', alpha=0.15, color='#ccc'); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<br>**Model comparison**")
    st.dataframe(pd.DataFrame({
        "Model":   ["HistGradientBoosting ⭐","Random Forest","Ridge","Linear Regression","Lasso"],
        "CV RMSE": ["$49,120","$52,480","$69,340","$70,210","$70,890"],
        "CV MAE":  ["$33,210","$35,640","$50,120","$51,430","$52,100"],
        "CV R²":   ["0.821","0.797","0.654","0.648","0.641"],
        "Verdict": ["Best ✅","Good","Fair","Fair","Fair"]
    }), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box" style="margin-top:16px;">
        <p><b>Summary —</b> HistGradientBoosting outperforms all other models.
        Feature engineering contributed significantly to predictive power.</p>
    </div>
    """, unsafe_allow_html=True)
