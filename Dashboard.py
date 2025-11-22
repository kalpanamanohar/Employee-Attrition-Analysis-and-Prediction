import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

st.set_page_config(page_title="Employee Analytics Dashboard", layout="wide")

# ---------------------------------------------------------------
# LOAD TRAINED MODELS
# ---------------------------------------------------------------
@st.cache_resource
def load_models():
    with open("best_model_final.pkl", "rb") as f:
        attrition_model = pickle.load(f)
    with open("best_promotion_model.pkl", "rb") as f:
        promotion_model = pickle.load(f)
    return attrition_model, promotion_model


attrition_model, promotion_model = load_models()

# ---------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------
st.sidebar.title("🏢 Employee Analytics Dashboard")
option = st.sidebar.radio(
    "Select Prediction Task:",
    ("Predict Employee Attrition", "Predict Promotion Likelihood")
)

# ---------------------------------------------------------------
# ATTRITION PREDICTION
# ---------------------------------------------------------------
if option == "Predict Employee Attrition":
    st.title("🔮 Predict Employee Attrition (Turnover Prediction)")
    st.write("""
    **Goal:** Predict whether an employee will leave the company (Attrition).  
    **Target Variable:** Attrition (Yes/No)
    """)

    with st.form("attrition_form"):
        st.subheader("Enter Employee Details:")

        age = st.number_input("Age", 18, 65, 30)
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000)
        job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
        years_at_company = st.number_input("Years at Company", 0, 40, 3)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        overtime = st.selectbox("OverTime", ["Yes", "No"])

        submit = st.form_submit_button("Predict Attrition")

    if submit:
        # 1️⃣ All columns used during training
        all_cols = [
            'Age','BusinessTravel','DailyRate','Department','DistanceFromHome',
            'Education','EducationField','EnvironmentSatisfaction','Gender','HourlyRate',
            'JobInvolvement','JobLevel','JobRole','MaritalStatus','MonthlyIncome',
            'MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike',
            'PerformanceRating','RelationshipSatisfaction','StockOptionLevel',
            'TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance',
            'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion',
            'YearsWithCurrManager','OvertimeLowPay','NoPromotionRecently','YearsBeforeManager'
        ]

        # 2️⃣ Create default input values
        user_input = {col: 0 for col in all_cols}
        user_input.update({
            'Age': age,
            'Department': department,
            'MonthlyIncome': monthly_income,
            'JobSatisfaction': job_satisfaction,
            'YearsAtCompany': years_at_company,
            'MaritalStatus': marital_status,
            'OverTime': 1 if overtime == "Yes" else 0,
            'OvertimeLowPay': 0,
            'NoPromotionRecently': 0,
            'YearsBeforeManager': 0,
            'BusinessTravel': "Travel_Rarely",
            'Education': 3,
            'EducationField': "Life Sciences",
            'EnvironmentSatisfaction': 3,
            'Gender': "Male",
            'JobRole': "Sales Executive",
            'JobLevel': 2,
            'PerformanceRating': 3,
            'WorkLifeBalance': 3
        })

        user_df = pd.DataFrame([user_input])

        # 3️⃣ Predict
        try:
            prediction = attrition_model.predict(user_df)[0]
            prob = attrition_model.predict_proba(user_df)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.error(f"⚠️ Employee is **likely to leave** (Attrition Probability: {prob:.2f})")
            else:
                st.success(f"✅ Employee is **likely to stay** (Attrition Probability: {prob:.2f})")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------
# PROMOTION PREDICTION
# ---------------------------------------------------------------
elif option == "Predict Promotion Likelihood":
    st.title("🚀 Predict Employee Promotion Likelihood")
    st.write("""
    **Goal:** Predict when an employee is likely to get promoted.  
    **Target Variable:** YearsSinceLastPromotion
    """)

    with st.form("promotion_form"):
        st.subheader("Enter Employee Details:")

        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        total_working_years = st.number_input("Total Working Years", 0, 40, 5)
        years_in_current_role = st.number_input("Years in Current Role", 0, 20, 3)
        performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        years_at_company = st.number_input("Years at Company", 0, 40, 5)
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manager", "Manufacturing Director", "Healthcare Representative",
            "Human Resources", "Sales Representative"
        ])
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        training_times_last_year = st.slider("Training Times Last Year", 0, 10, 3)
        work_life_balance = st.slider("Work Life Balance (1=Poor, 4=Excellent)", 1, 4, 3)

        submit2 = st.form_submit_button("Predict Promotion Likelihood")

    if submit2:
        df_input = pd.DataFrame({
            "JobLevel": [job_level],
            "TotalWorkingYears": [total_working_years],
            "YearsInCurrentRole": [years_in_current_role],
            "PerformanceRating": [performance_rating],
            "Education": [education],
            "YearsAtCompany": [years_at_company],
            "JobRole": [job_role],
            "Department": [department],
            "TrainingTimesLastYear": [training_times_last_year],
            "WorkLifeBalance": [work_life_balance],
            "PromotionGap": [years_at_company - 0],
            "ExperienceRatio": [years_in_current_role / (total_working_years + 1)],
            "HighPerformer": [1 if performance_rating >= 4 else 0],
            "TrainingEffect": [training_times_last_year * performance_rating]
        })

        try:
            predicted_years = promotion_model.predict(df_input)[0]
            st.markdown("---")
            st.success(f"🎯 Estimated Years Until Next Promotion: **{predicted_years:.2f} years**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("""
---
📊 *Developed with ❤️ using Streamlit and Scikit-learn*  
Model Files: `best_model_final.pkl` | `best_promotion_model.pkl`
""")
