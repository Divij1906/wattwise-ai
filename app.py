import streamlit as st
import pandas as pd
import joblib
from database import save_prediction, get_predictions

# ---- PAGE CONFIGURATION ----
st.set_page_config(
    page_title="WATTWISE | Smart Energy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- FUTURISTIC UI THEME ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #07111f, #0b1728 55%, #071b20);
    color: #e6edf7;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(7, 17, 31, 0.8);
}

[data-testid="stSidebar"] {
    background: #091321;
    border-right: 1px solid #203447;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: #58f5b4;
}

h1, h2, h3 {
    color: #e6edf7 !important;
    font-weight: 800 !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #102438, #0c1c2b);
    border: 1px solid #254557;
    border-radius: 16px;
    padding: 20px 18px;
}

[data-testid="stMetricLabel"] {
    color: #9cb5c9 !important;
}

[data-testid="stMetricValue"] {
    color: #58f5b4 !important;
    font-weight: 800;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #20d99a, #72f5b2);
    color: #07151c !important;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.3rem;
    font-weight: 800;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    box-shadow: 0 0 22px rgba(56, 235, 166, 0.35);
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {
    background-color: #102438;
    color: #e6edf7;
    border-color: #254557;
    border-radius: 8px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #254557;
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #203447;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---- LOAD SAVED MODELS ----
scaler = joblib.load("scaler.pkl")
lin_model = joblib.load("linear_regression_model.pkl")
log_model = joblib.load("logistic_regression_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
cluster_to_label = joblib.load("cluster_labels.pkl")
feature_columns = joblib.load("feature_columns.pkl")
median_units = joblib.load("median_units.pkl")

# ---- SIDEBAR NAVIGATION ----
st.sidebar.markdown(
    "<h1 style='color:#58f5b4;'>⚡ WATTWISE</h1>",
    unsafe_allow_html=True
)
st.sidebar.caption("SMART ENERGY INTELLIGENCE")
page = st.sidebar.radio(
    "Navigation",
    [
        "Predict Bill",
        "Prediction History",
        "Model Insights",
        "About Project"
    ]
)

# ---- HERO SECTION ----
st.markdown("""
<div style="padding:25px 8px 20px 8px;">
    <div style="display:inline-block; padding:6px 12px;
                border:1px solid #285b52; border-radius:20px;
                color:#58f5b4; background:rgba(32,217,154,0.08);
                font-size:12px; font-weight:700; letter-spacing:2px;">
        ⚡ SMART ENERGY INTELLIGENCE
    </div>
    <h1 style="font-size:48px; margin:18px 0 8px 0;
               font-weight:800; letter-spacing:-1.5px;">
        WATT<span style="color:#58f5b4;">WISE</span> AI
    </h1>
    <p style="font-size:18px; color:#a8bdcf; max-width:700px;
              line-height:1.7;">
        Predict your electricity bill, understand your energy
        consumption, and discover smarter ways to save.
    </p>
    <p style="color:#58f5b4; font-size:13px; letter-spacing:1px;">
        MACHINE LEARNING · BILL FORECASTING · ENERGY INSIGHTS
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================================================
# PAGE 1: PREDICT BILL
# =========================================================
if page == "Predict Bill":

    st.header("🏠 Household Energy Profile")
    st.caption(
        "Enter your household details to estimate monthly electricity usage."
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            family_size = st.number_input(
                "Family size", min_value=1, max_value=15, value=4
            )
            fans = st.number_input(
                "Number of fans", min_value=0, max_value=15, value=3
            )
            lights = st.number_input(
                "Number of lights", min_value=0, max_value=20, value=6
            )
            ac_count = st.number_input(
                "Number of ACs", min_value=0, max_value=5, value=1
            )
            ac_hours = st.number_input(
                "AC usage hours/day", min_value=0.0,
                max_value=24.0, value=4.0, step=0.5
            )

        with col2:
            fridge = st.selectbox("Fridge available?", ["Yes", "No"])
            washing_machine = st.selectbox(
                "Washing machine available?", ["Yes", "No"]
            )
            geyser = st.selectbox("Geyser available?", ["Yes", "No"])
            tv_hours = st.number_input(
                "TV usage hours/day", min_value=0.0,
                max_value=12.0, value=3.0, step=0.5
            )
            season = st.selectbox(
                "Season", ["Summer", "Monsoon", "Winter"]
            )

        predict_clicked = st.button(
            "⚡ Predict My Electricity Bill", use_container_width=True
        )

    # ---- BILL CALCULATION ----
    def units_to_bill(units):
        if units <= 100:
            energy = units * 3.44
        elif units <= 300:
            energy = 100 * 3.44 + (units - 100) * 7.43
        elif units <= 500:
            energy = (
                100 * 3.44 + 200 * 7.43 + (units - 300) * 10.32
            )
        else:
            energy = (
                100 * 3.44 + 200 * 7.43
                + 200 * 10.32 + (units - 500) * 11.57
            )

        fixed = (
            120 if units <= 100
            else 220 if units <= 300
            else 360 if units <= 500
            else 480
        )
        duty = 0.16 * energy
        return round(energy + fixed + duty, -1)

    if predict_clicked:
        input_dict = {
            "family_size": family_size,
            "fans": fans,
            "lights": lights,
            "ac_count": ac_count,
            "ac_hours_per_day": ac_hours,
            "fridge": 1 if fridge == "Yes" else 0,
            "washing_machine": 1 if washing_machine == "Yes" else 0,
            "geyser": 1 if geyser == "Yes" else 0,
            "tv_hours_per_day": tv_hours,
            "season_Summer": 1 if season == "Summer" else 0,
            "season_Monsoon": 1 if season == "Monsoon" else 0,
            "season_Winter": 1 if season == "Winter" else 0,
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(
            columns=feature_columns, fill_value=0
        )

        input_scaled = scaler.transform(input_df)

        predicted_units = max(
            float(lin_model.predict(input_scaled)[0]), 0
        )
        predicted_bill = units_to_bill(predicted_units)

        bill_class = log_model.predict(input_scaled)[0]
        bill_label = "High" if bill_class == 1 else "Normal"

        cluster_id = int(kmeans.predict(input_scaled)[0])
        usage_group = str(cluster_to_label[cluster_id])

        # Save the prediction to SQLite
        try:
            save_prediction({
                "family_size": family_size,
                "fans": fans,
                "lights": lights,
                "ac_count": ac_count,
                "ac_hours_per_day": ac_hours,
                "fridge": 1 if fridge == "Yes" else 0,
                "washing_machine": 1 if washing_machine == "Yes" else 0,
                "geyser": 1 if geyser == "Yes" else 0,
                "tv_hours_per_day": tv_hours,
                "season": season,
                "predicted_units": predicted_units,
                "predicted_bill": float(predicted_bill),
                "bill_category": bill_label,
                "usage_group": usage_group,
            })
            st.success("Prediction completed and saved!")
        except Exception as error:
            st.error(f"Prediction worked, but saving failed: {error}")

        st.subheader("📊 Your Energy Forecast")
        # ---- ESTIMATED APPLIANCE ENERGY BREAKDOWN ----
        st.subheader("⚡ Estimated Appliance Usage")

        appliance_units = {
            "Air Conditioner": ac_count * ac_hours * 0.9 * 30,
            "Fans": fans * 0.07 * 10 * 30,
            "Lights": lights * 0.01 * 6 * 30,
            "Television": tv_hours * 0.1 * 30,
            "Refrigerator": 1.5 * 30 if fridge == "Yes" else 0,
            "Washing Machine": 0.5 * 12 if washing_machine == "Yes" else 0,
            "Geyser": 2 * 0.5 * 30 if geyser == "Yes" else 0,
        }

        appliance_df = pd.DataFrame(
            list(appliance_units.items()),
            columns=["Appliance", "Illustrative Monthly kWh"]
        )

        appliance_df = appliance_df[
            appliance_df["Illustrative Monthly kWh"] > 0
        ]

        if not appliance_df.empty:
            appliance_df = appliance_df.sort_values(
                "Illustrative Monthly kWh", ascending=False
            )

            st.bar_chart(
                appliance_df.set_index("Appliance")
                ["Illustrative Monthly kWh"],
                horizontal=True
            )

            st.caption(
                "Illustrative appliance estimates based on assumed power "
                "and usage. These are not measured values and may not "
                "sum to your ML-predicted household consumption."
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Units", f"{predicted_units:.0f} kWh")
        c2.metric("Estimated Bill", f"₹{predicted_bill:.0f}")
        c3.metric("Usage Group", usage_group)

        st.info(
            f"Bill Category: **{bill_label}** | "
            f"Median reference: {median_units:.0f} units"
        )

        st.subheader("💡 Personalized Energy-Saving Tips")
        tips = []

        if ac_count > 0 and ac_hours > 6:
            tips.append(
                "Reduce AC use by 1–2 hours daily and consider "
                "a temperature setting of 24–26°C."
            )
        if geyser == "Yes":
            tips.append(
                "Use the geyser for shorter periods and switch it "
                "off when hot water is no longer needed."
            )
        if washing_machine == "Yes":
            tips.append(
                "Run full washing-machine loads instead of several "
                "small loads."
            )
        if usage_group == "Heavy User":
            tips.append(
                "Review high-consumption appliances and consider "
                "an energy audit."
            )
        if not tips:
            tips.append(
                "Track your monthly meter readings and compare them "
                "to spot changes in consumption."
            )

        for tip in tips:
            st.markdown(f"- {tip}")

        st.caption(
            "Educational estimate only. Actual bills depend on the "
            "applicable tariff, taxes, meter readings, and other charges."
        )

# =========================================================
# PAGE 2: PREDICTION HISTORY
# =========================================================
elif page == "Prediction History":

    st.header("📊 Prediction History")
    st.caption("Your saved forecasts, organized in one place.")

    records = get_predictions()

    if records:
        columns = [
            "ID", "Date & Time", "Family Size", "Fans",
            "Lights", "ACs", "AC Hours/Day", "Fridge",
            "Washing Machine", "Geyser", "TV Hours/Day",
            "Season", "Predicted Units", "Predicted Bill",
            "Bill Category", "Usage Group"
        ]

        history_df = pd.DataFrame(records, columns=columns)

        history_df["Predicted Units"] = (
            history_df["Predicted Units"].round(1)
        )
        history_df["Predicted Bill"] = (
            history_df["Predicted Bill"].round(0)
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", len(history_df))
        c2.metric(
            "Average Units",
            f"{history_df['Predicted Units'].mean():.0f} kWh"
        )
        c3.metric(
            "Average Estimated Bill",
            f"₹{history_df['Predicted Bill'].mean():.0f}"
        )

        st.subheader("Your Saved Records")
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("📈 Bill Comparison Dashboard")

        chart_df = history_df.sort_values("ID").copy()

        chart_df["Prediction"] = (
            "Prediction #" + chart_df["ID"].astype(str)
        )

        st.line_chart(
            chart_df.set_index("Prediction")["Predicted Bill"],
            y_label="Estimated Bill (₹)",
            x_label="Saved Predictions"
        )

        st.subheader("📊 Consumption Comparison")

        st.bar_chart(
            chart_df.set_index("Prediction")["Predicted Units"],
            y_label="Estimated Units (kWh)",
            x_label="Saved Predictions"
        )

        if len(chart_df) >= 2:
            first_bill = float(chart_df.iloc[0]["Predicted Bill"])
            latest_bill = float(chart_df.iloc[-1]["Predicted Bill"])

            difference = latest_bill - first_bill

            st.metric(
                "Latest vs Earliest Saved Bill",
                f"₹{latest_bill:.0f}",
                delta=f"₹{difference:+.0f}",
                delta_color="inverse"
            )

        st.caption(
            "These charts compare saved predictions, not verified "
            "monthly meter readings. They do not prove that actual "
            "electricity consumption has increased or decreased."
        )

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download History as CSV",
            data=csv,
            file_name="electricity_prediction_history.csv",
            mime="text/csv"
        )

    else:
        st.info(
            "No predictions saved yet. Open Predict Bill and "
            "make your first prediction."
        )
# =========================================================
# PAGE 3: MODEL INSIGHTS
# =========================================================
elif page == "Model Insights":

    st.header("🧠 Model Insights")
    st.caption(
        "Explore the machine-learning algorithms powering WATTWISE AI."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("📈 Linear Regression")
            st.write(
                "Predicts household electricity consumption in kWh "
                "from the household input features."
            )
            st.caption("Output: Estimated electricity units")

    with col2:
        with st.container(border=True):
            st.subheader("🏷️ Logistic Regression")
            st.write(
                "Classifies a prediction into a bill category using "
                "the model's learned decision boundary."
            )
            st.caption("Output: Normal or High")

    col3, col4 = st.columns(2)

    with col3:
        with st.container(border=True):
            st.subheader("👥 K-Means Clustering")
            st.write(
                "Groups household input patterns into clusters that "
                "your project maps to usage-group labels."
            )
            st.caption("Output: Household usage group")

    with col4:
        with st.container(border=True):
            st.subheader("🔍 PCA")
            st.write(
                "Principal Component Analysis can reduce the number "
                "of dimensions in data while retaining important "
                "variation. In this app, the PCA model is loaded but "
                "is not currently used in the prediction pipeline."
            )
            st.caption("Status: Loaded, not currently applied")

    st.divider()
    st.subheader("⚙️ Prediction Pipeline")

    st.markdown("""
    1. **Household inputs:** appliance counts, usage hours, family size and season.
    2. **Feature alignment:** inputs are arranged in the training column order.
    3. **Feature scaling:** the saved scaler transforms the inputs.
    4. **Unit prediction:** Linear Regression estimates electricity consumption.
    5. **Bill estimation:** the estimated units are passed to the tariff formula.
    6. **Bill classification:** Logistic Regression predicts the bill category.
    7. **Usage grouping:** K-Means assigns a cluster that maps to a usage label.
    8. **History storage:** the prediction is saved in the local SQLite database.
    """)

    st.warning(
        "Model accuracy, training dataset size and validation results "
        "are not displayed because those measurements have not yet "
        "been verified."
    )
# =========================================================
# PAGE 4: ABOUT PROJECT
# =========================================================
elif page == "About Project":

    st.header("🚀 About WATTWISE AI")
    st.caption("An academic machine-learning project for household energy estimation.")

    st.markdown("""
    <div style="
        padding: 24px;
        border: 1px solid #254557;
        border-radius: 16px;
        background: linear-gradient(135deg, #102438, #0c1c2b);
    ">
        <h2 style="color:#58f5b4 !important;">
            Smarter Energy. Better Decisions.
        </h2>
        <p style="color:#c0d0df; line-height:1.8;">
            WATTWISE AI estimates household electricity consumption,
            calculates an approximate electricity bill, classifies
            bill categories, and groups usage patterns using
            machine-learning models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🎯 Project Objectives")

    st.markdown("""
    - Estimate household electricity consumption from user inputs.
    - Calculate an approximate bill using a demonstration tariff formula.
    - Classify predicted bills as Normal or High.
    - Group households according to learned usage patterns.
    - Store predictions locally and visualize previous results.
    """)

    st.subheader("🛠️ Technology Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Programming and interface**")
        st.markdown("""
        - Python
        - Streamlit
        - Pandas
        - NumPy
        """)

    with col2:
        st.markdown("**Machine learning and storage**")
        st.markdown("""
        - Scikit-learn model artifacts
        - Linear Regression
        - Logistic Regression
        - K-Means Clustering
        - SQLite database
        - Joblib
        """)

    st.subheader("🔄 How It Works")

    st.markdown("""
    1. The user enters household and appliance details.
    2. The saved preprocessing pipeline prepares the input features.
    3. Machine-learning models estimate units, bill category, and usage group.
    4. A tariff formula calculates an approximate bill.
    5. The results are stored in SQLite and displayed in the dashboard.
    """)

    st.subheader("⚠️ Current Limitations")

    st.markdown("""
    - Predictions depend on the quality and representativeness of the training data.
    - The tariff formula is approximate and must not be treated as an official bill.
    - Appliance breakdowns are illustrative estimates, not measured consumption.
    - Model accuracy and generalization have not yet been independently verified.
    - The current database is local to the running application environment.
    """)

    st.divider()

    st.caption(
        "WATTWISE AI | Academic Project | Built with Python and Streamlit"
    )