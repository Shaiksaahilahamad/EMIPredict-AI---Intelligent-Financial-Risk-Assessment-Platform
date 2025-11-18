import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src.config import (
    PREPROCESSOR_PATH,
    CLASSIFIER_MODEL_PATH,
    REGRESSOR_MODEL_PATH,
    EMI_SCENARIO_RULES,
    DATA_PATH,
)
from src.utils_scenario import validate_emi_scenario
from src.data_quality import data_quality_report
from src.data_preprocessing import basic_cleaning
from src.feature_engineering import engineer_features

# ---------------------------------------------------------
# 🔵 GLOBAL PAGE CONFIG + CUSTOM THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="EMIPredict AI",
    layout="wide",
    page_icon="💸",
)


def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Overall background */
        .stApp {
            background: radial-gradient(circle at top left, #0f172a 0, #020617 60%);
            color: #e5e7eb;
        }

        /* Center block container */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* Card look */
        .card {
            border-radius: 18px;
            padding: 18px 20px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.4);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.65);
        }

        .card-soft {
            border-radius: 16px;
            padding: 16px 18px;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(51, 65, 85, 0.7);
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.80rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .pill-eligible {
            background: rgba(16, 185, 129, 0.1);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }

        .pill-high {
            background: rgba(245, 158, 11, 0.08);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.5);
        }

        .pill-not {
            background: rgba(248, 113, 113, 0.08);
            color: #fca5a5;
            border: 1px solid rgba(248, 113, 113, 0.5);
        }

        .headline {
            font-size: 2.0rem;
            font-weight: 700;
            background: linear-gradient(120deg, #38bdf8, #22c55e);
            -webkit-background-clip: text;
            color: transparent;
        }

        .subtle {
            color: #9ca3af;
            font-size: 0.92rem;
        }

        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
        }

        .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
        }

        .emoji-badge {
            font-size: 1.3rem;
            margin-right: 0.35rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617, #020617);
            border-right: 1px solid rgba(51, 65, 85, 0.9);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()

# ---------------------------------------------------------
# 🔁 MODEL ARTIFACTS
# ---------------------------------------------------------


@st.cache_resource
def load_artifacts():
    preproc_dict = joblib.load(PREPROCESSOR_PATH)
    preprocessor = preproc_dict["preprocessor"]
    feature_cols = preproc_dict["feature_cols"]

    clf_obj = joblib.load(CLASSIFIER_MODEL_PATH)
    # New format: {"model": best_model, "label_encoder": label_encoder}
    if isinstance(clf_obj, dict) and "model" in clf_obj:
        classifier = clf_obj["model"]
        label_encoder = clf_obj.get("label_encoder", None)
    else:
        classifier = clf_obj
        label_encoder = None  # fallback if old file

    regressor = joblib.load(REGRESSOR_MODEL_PATH)

    return preprocessor, feature_cols, classifier, regressor, label_encoder


@st.cache_data
def load_training_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        return df
    return pd.DataFrame()


def run_prediction(input_dict):
    preprocessor, feature_cols, classifier, regressor, label_encoder = load_artifacts()

    # 1) Raw input from Streamlit
    df_input_raw = pd.DataFrame([input_dict])

    # 2) Apply the SAME feature engineering as in training
    df_input = engineer_features(df_input_raw)

    # 3) Align columns with training feature_cols
    missing_cols = [c for c in feature_cols if c not in df_input.columns]
    for c in missing_cols:
        df_input[c] = 0

    df_input = df_input[feature_cols]

    # 4) Transform & predict
    X_trans = preprocessor.transform(df_input)

    raw_pred = classifier.predict(X_trans)[0]
    proba = classifier.predict_proba(X_trans)[0]

    if label_encoder is not None:
        class_pred = label_encoder.inverse_transform([raw_pred])[0]
        class_labels = list(label_encoder.classes_)
    else:
        class_pred = raw_pred
        class_labels = list(getattr(classifier, "classes_", []))

    emi_pred = regressor.predict(X_trans)[0]

    # Optional safety: avoid weird negative EMI
    emi_pred = float(emi_pred)
    if emi_pred < 0:
        emi_pred = 0.0

    return class_pred, proba, emi_pred, class_labels


# ---------------------------------------------------------
# 🧭 SIDEBAR NAVIGATION
# ---------------------------------------------------------


def sidebar_nav():
    st.sidebar.markdown("### 💸 EMIPredict AI")
    st.sidebar.caption("Intelligent EMI & risk assessment")

    st.sidebar.markdown("---")
    return st.sidebar.radio(
        "Navigation",
        ["Overview", "EDA & Model Summary", "EMI Prediction", "Data Explorer / CRUD"],
        index=2,
    )


# ---------------------------------------------------------
# 📄 OVERVIEW PAGE
# ---------------------------------------------------------


def page_overview():
    st.markdown(
        """
        <div class="card">
            <div class="pill pill-eligible" style="margin-bottom: 0.35rem;">
                <span class="emoji-badge">✨</span>FinTech · Risk Assessment
            </div>
            <div class="headline">EMIPredict AI – Intelligent Financial Risk Assessment</div>
            <p class="subtle" style="margin-top:0.5rem;">
                Real-time EMI eligibility, risk scoring, and maximum safe EMI powered by machine learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="card-soft">
                <div class="metric-label">Dual ML Tasks</div>
                <div class="metric-value">Classification + Regression</div>
                <div class="subtle">Eligibility + Max EMI in one shot</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="card-soft">
                <div class="metric-label">Financial Profiles</div>
                <div class="metric-value">400K+</div>
                <div class="subtle">Synthetic but realistic EMI scenarios</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="card-soft">
                <div class="metric-label">EMI Scenarios</div>
                <div class="metric-value">5</div>
                <div class="subtle">Shopping · Home · Vehicle · Personal · Education</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### 🧠 What this platform does")
        st.markdown(
            """
            - ✅ **Classification** – Predict EMI eligibility  
              *(Eligible / High_Risk / Not_Eligible)*
            - 📈 **Regression** – Estimate **maximum safe monthly EMI** (`max_monthly_emi`)
            - 📊 **Feature engineering** – Ratios for income, expenses, EMI burden, savings
            - 🔁 **MLflow integration** – Model tracking, metrics, and experiment history
            - ☁️ **Streamlit-ready** – Built for demo + deployment
            """
        )
    with right:
        st.markdown("#### 🏦 Where it’s useful")
        st.markdown(
            """
            - Banks & NBFCs for faster **loan underwriting**
            - FinTech apps for **instant eligibility checks**
            - Loan officers for a **quick financial health snapshot**
            - Credit teams for **risk-based pricing** decisions  
            """
        )

    st.markdown("---")
    st.subheader("📏 EMI Scenario Rules")

    rules_df = (
        pd.DataFrame(EMI_SCENARIO_RULES)
        .T[["amount_min", "amount_max", "tenure_min", "tenure_max"]]
        .rename(
            columns={
                "amount_min": "Min Amount (₹)",
                "amount_max": "Max Amount (₹)",
                "tenure_min": "Min Tenure (months)",
                "tenure_max": "Max Tenure (months)",
            }
        )
    )
    st.dataframe(
        rules_df.style.format("{:,.0f}"),
        use_container_width=True,
    )


# ---------------------------------------------------------
# 📊 EDA & MODEL SUMMARY
# ---------------------------------------------------------


def page_eda():
    st.markdown(
        """
        <div class="card">
            <div class="pill pill-high" style="margin-bottom: 0.35rem;">
                <span class="emoji-badge">📊</span>EDA Snapshot
            </div>
            <div class="headline" style="font-size:1.6rem;">EDA & Model Summary</div>
            <p class="subtle" style="margin-top:0.5rem;">
                High-level view of your training data distribution and targets.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    df = load_training_data()
    if df.empty:
        st.warning("Training dataset not found at data/EMI_dataset.csv")
        return

    dq = data_quality_report(df)

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown("##### Dataset Overview")
        st.markdown(
            f"""
            - **Rows:** `{dq['row_count']}`  
            - **Columns:** `{dq['column_count']}`  
            - **Duplicate rows:** `{dq['duplicate_rows']}`
            """
        )

        tab1, tab2 = st.tabs(["🎯 EMI Eligibility", "🧺 EMI Scenario"])
        with tab1:
            st.caption("Class distribution for `emi_eligibility`")
            st.bar_chart(df["emi_eligibility"].value_counts())
        with tab2:
            st.caption("Count of records per EMI scenario")
            st.bar_chart(df["emi_scenario"].value_counts())

    with col2:
        st.markdown("##### Numeric Summary (sample)")
        num_cols = df.select_dtypes(include=["number"]).columns
        st.dataframe(df[num_cols].describe().T, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 💰 Salary vs. Max EMI (sample scatter)")
        # simple scatter-style using sample + line chart
        temp = df[["monthly_salary", "max_monthly_emi"]].dropna().sample(
            min(500, len(df)), random_state=42
        )
        temp = temp.sort_values("monthly_salary")
        st.line_chart(temp.set_index("monthly_salary"))

    with col_b:
        st.markdown("##### 🧮 Credit Score Distribution")
        if "credit_score" in df.columns:
            st.histogram = st.bar_chart(
                df["credit_score"].value_counts().sort_index()
            )

    st.markdown(
        """
        <p class="subtle">
        For full EDA (correlations, feature importance, MLflow runs, etc.) you can use a Jupyter Notebook.  
        This page is meant as a **visual summary** for your Streamlit app demo.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# 🔮 PREDICTION PAGE – MAIN EYE CANDY
# ---------------------------------------------------------


def page_prediction():
    st.markdown(
        """
        <div class="card">
            <div class="pill pill-eligible" style="margin-bottom: 0.35rem;">
                <span class="emoji-badge">🔮</span>Real-time Prediction
            </div>
            <div class="headline" style="font-size:1.8rem;">EMI Eligibility & Max EMI Estimator</div>
            <p class="subtle" style="margin-top:0.5rem;">
                Fill the customer profile on the left, get an AI-powered EMI decision on the right.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    with st.form("emi_form"):
        col1, col2, col3 = st.columns(3)

        # ==== COL 1 – Demographics & Job ====
        with col1:
            st.markdown("##### 👤 Customer Profile")
            age = st.number_input("Age", 18, 80, 38)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox(
                "Education",
                ["High School", "Graduate", "Post Graduate", "Professional"],
            )
            employment_type = st.selectbox(
                "Employment Type", ["Private", "Government", "Self-employed"]
            )
            years_of_employment = st.number_input(
                "Years of Employment", 0.0, 50.0, 5.0, step=0.1
            )

        # ==== COL 2 – Housing & Expenses ====
        with col2:
            st.markdown("##### 🏠 Household & Lifestyle")
            monthly_salary = st.number_input(
                "Monthly Salary (₹)", 10000.0, 500000.0, 50000.0, step=1000.0
            )
            company_type = st.selectbox(
                "Company Type", ["Mid-size", "MNC", "Startup", "Large Indian"]
            )
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
            monthly_rent = st.number_input(
                "Monthly Rent (₹)", 0.0, 200000.0, 0.0, step=1000.0
            )
            family_size = st.number_input("Family Size", 1, 20, 3)
            dependents = st.number_input("Dependents", 0, 10, 1)

        # ==== COL 3 – Education & Monthly Costs ====
        with col3:
            st.markdown("##### 💸 Monthly Obligations")
            school_fees = st.number_input("School Fees (₹)", 0.0, 200000.0, 0.0, 1000.0)
            college_fees = st.number_input(
                "College Fees (₹)", 0.0, 200000.0, 0.0, 1000.0
            )
            travel_expenses = st.number_input(
                "Travel Expenses (₹)", 0.0, 100000.0, 3000.0, 500.0
            )
            groceries_utilities = st.number_input(
                "Groceries & Utilities (₹)", 0.0, 200000.0, 15000.0, 1000.0
            )
            other_monthly_expenses = st.number_input(
                "Other Monthly Expenses (₹)", 0.0, 200000.0, 5000.0, 500.0
            )
            existing_loans = st.selectbox("Existing Loans?", ["Yes", "No"])
            current_emi_amount = st.number_input(
                "Current EMI Amount (₹)", 0.0, 200000.0, 0.0, 1000.0
            )

        st.markdown("---")
        col4, col5, col6 = st.columns(3)

        # ==== Additional financial profile ====
        with col4:
            st.markdown("##### 💳 Financial Strength")
            credit_score = st.number_input(
                "Credit Score", 300.0, 900.0, 700.0, step=1.0
            )
            bank_balance = st.number_input(
                "Bank Balance (₹)", 0.0, 2000000.0, 100000.0, 5000.0
            )
            emergency_fund = st.number_input(
                "Emergency Fund (₹)", 0.0, 2000000.0, 50000.0, 5000.0
            )

        with col5:
            st.markdown("##### 🧾 Loan Request")
            emi_scenario = st.selectbox(
                "EMI Scenario",
                [
                    "E-commerce Shopping EMI",
                    "Home Appliances EMI",
                    "Vehicle EMI",
                    "Personal Loan EMI",
                    "Education EMI",
                ],
            )
            requested_amount = st.number_input(
                "Requested Loan Amount (₹)", 10000.0, 2000000.0, 200000.0, 10000.0
            )
            requested_tenure = st.number_input(
                "Requested Tenure (months)", 3, 120, 24, 1
            )

        with col6:
            st.markdown("##### ")
            st.markdown(" ")
            st.markdown(" ")
            submit_btn = st.form_submit_button("⚡ Run EMI Prediction", use_container_width=True)

    if submit_btn:
        input_dict = {
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "emi_scenario": emi_scenario,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
        }

        valid, messages = validate_emi_scenario(
            emi_scenario, requested_amount, requested_tenure
        )

        st.markdown("---")
        st.subheader("🧾 Scenario Validation")
        for msg in messages:
            if valid:
                st.success(msg)
            else:
                st.warning(msg)

        class_pred, proba, emi_pred, class_labels = run_prediction(input_dict)

        st.subheader("📌 Model Output")

        colL, colR = st.columns([1.1, 1])

        # ---------------- Classification result (eye candy) ----------------
        with colL:
            # pick pill color based on prediction
            pred = str(class_pred)
            if pred == "Eligible":
                pill_class = "pill-eligible"
                emoji = "✅"
                subtitle = "Low risk · EMI looks comfortable"
            elif pred == "High_Risk":
                pill_class = "pill-high"
                emoji = "⚠️"
                subtitle = "Borderline case · Needs careful review"
            else:
                pill_class = "pill-not"
                emoji = "⛔"
                subtitle = "High risk · Not recommended"

            st.markdown(
                f"""
                <div class="card-soft">
                    <div class="pill {pill_class}">
                        <span class="emoji-badge">{emoji}</span>Predicted EMI Eligibility
                    </div>
                    <div style="margin-top:0.8rem;" class="metric-value">{pred}</div>
                    <div class="subtle">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("###### 🎯 Class Probabilities")

            labels = class_labels if class_labels else ["Eligible", "High_Risk", "Not_Eligible"]

            # Show as small progress bars instead of plain dataframe
            for label, p in zip(labels[: len(proba)], proba):
                pct = float(p) * 100.0
                st.write(f"**{label}** – {pct:.1f}%")
                st.progress(float(p))

        # ---------------- Regression result & EMI comparison ----------------
        with colR:
            approx_requested_emi = requested_amount / max(1, requested_tenure)

            st.markdown(
                f"""
                <div class="card-soft">
                    <div class="metric-label">Max Safe Monthly EMI (₹)</div>
                    <div class="metric-value">₹{emi_pred:,.0f}</div>
                    <div class="subtle">Estimated from full financial profile</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("###### 💡 EMI Comparison")
            st.write(
                f"Approximate EMI for this request (no interest): **₹{approx_requested_emi:,.0f} / month**"
            )

            # simple visual bar comparison
            max_val = max(emi_pred, approx_requested_emi, 1)
            safe_ratio = min(emi_pred / max_val, 1.0)
            requested_ratio = min(approx_requested_emi / max_val, 1.0)

            st.caption("Safe EMI vs Requested EMI")
            st.write("Safe EMI capacity")
            st.progress(float(safe_ratio))
            st.write("Requested EMI level")
            st.progress(float(requested_ratio))

            if approx_requested_emi <= emi_pred:
                st.success("Requested EMI appears **affordable** based on model estimate.")
            else:
                st.error(
                    "Requested EMI is **above** the estimated safe EMI — this is a **high-risk** case."
                )


# ---------------------------------------------------------
# 📁 CRUD PAGE – DATA EXPLORER
# ---------------------------------------------------------


def user_data_path():
    return Path("data") / "user_records.csv"


@st.cache_data
def load_user_records():
    path = user_data_path()
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_user_records(df: pd.DataFrame):
    path = user_data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def page_crud():
    st.markdown(
        """
        <div class="card">
            <div class="pill pill-high" style="margin-bottom: 0.35rem;">
                <span class="emoji-badge">📁</span>Data Explorer
            </div>
            <div class="headline" style="font-size:1.7rem;">User Records – CRUD</div>
            <p class="subtle" style="margin-top:0.5rem;">
                Save, update, and delete custom EMI records for demos and testing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    df = load_user_records()
    st.subheader("📚 Current User Records")
    if df.empty:
        st.info("No user records yet. Use the form below to add one.")
    else:
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("✏️ Add / Update / Delete Record")

    with st.form("crud_form"):
        record_id = st.text_input("Record ID (for update/delete)", value="")
        json_template = (
            '{"age": 38, "gender": "Female", "marital_status": "Married", '
            '"education": "Professional", "monthly_salary": 82600.0, '
            '"employment_type": "Private", "years_of_employment": 0.9, '
            '"company_type": "Mid-size", "house_type": "Rented", "monthly_rent": 20000.0, '
            '"family_size": 3, "dependents": 2, "school_fees": 0.0, "college_fees": 0.0, '
            '"travel_expenses": 7200.0, "groceries_utilities": 19500.0, '
            '"other_monthly_expenses": 13200.0, "existing_loans": "Yes", '
            '"current_emi_amount": 23700.0, "credit_score": 660.0, "bank_balance": 303200.0, '
            '"emergency_fund": 70200.0, "emi_scenario": "Personal Loan EMI", '
            '"requested_amount": 850000.0, "requested_tenure": 15, '
            '"emi_eligibility": "Not_Eligible", "max_monthly_emi": 500.0}'
        )
        json_text = st.text_area(
            "Record JSON (must follow dataset schema)",
            height=220,
            value=json_template,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            add_btn = st.form_submit_button("➕ Add New", use_container_width=True)
        with col2:
            upd_btn = st.form_submit_button("🛠 Update by ID", use_container_width=True)
        with col3:
            del_btn = st.form_submit_button("🗑 Delete by ID", use_container_width=True)

    record = None
    if json_text.strip():
        try:
            record = pd.read_json(f"[{json_text}]", typ="frame").iloc[0].to_dict()
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    if add_btn and record:
        df_new = df.copy()
        new_id = int(df_new["id"].max() + 1) if ("id" in df_new.columns and not df_new.empty) else 1
        record_row = {"id": new_id}
        record_row.update(record)
        df_new = pd.concat([df_new, pd.DataFrame([record_row])], ignore_index=True)
        save_user_records(df_new)
        st.success(f"✅ Added record with id = {new_id}")

    if upd_btn and record_id and record:
        if "id" not in df.columns:
            st.error("No 'id' column yet. Add at least one record first.")
        else:
            rid = int(record_id)
            if rid in df["id"].values:
                df_new = df.copy()
                for k, v in record.items():
                    df_new.loc[df_new["id"] == rid, k] = v
                save_user_records(df_new)
                st.success(f"🛠 Updated record id = {rid}")
            else:
                st.error(f"No record with id = {rid}")

    if del_btn and record_id:
        if "id" not in df.columns:
            st.error("No 'id' column yet.")
        else:
            rid = int(record_id)
            if rid in df["id"].values:
                df_new = df[df["id"] != rid].copy()
                save_user_records(df_new)
                st.success(f"🗑 Deleted record id = {rid}")
            else:
                st.error(f"No record with id = {rid}")


# ---------------------------------------------------------
# 🏁 MAIN
# ---------------------------------------------------------


def main():
    page = sidebar_nav()
    if page == "Overview":
        page_overview()
    elif page == "EDA & Model Summary":
        page_eda()
    elif page == "EMI Prediction":
        page_prediction()
    else:
        page_crud()


if __name__ == "__main__":
    main()
