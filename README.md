# 🧠 Employee Attrition & Promotion Prediction

## 📊 Overview
This project aims to analyze employee data to **predict attrition (turnover)** and **promotion likelihood** using machine learning.  
It provides two separate pipelines:
1. **Employee Attrition Prediction (Classification)**
2. **Promotion Likelihood Prediction (Regression)**

Both models are trained, tuned, and saved as `.pkl` files for easy deployment in a Streamlit dashboard.

---

## 🎯 Project Objectives

### 🔹 Predict Employee Attrition
- **Goal:** Determine if an employee is likely to leave the company.  
- **Target Variable:** `Attrition` (`Yes`/`No`)  
- **Features:** Age, Department, Monthly Income, Job Satisfaction, Years at Company, Marital Status, Overtime, etc.  

### 🔹 Predict Promotion Likelihood
- **Goal:** Predict `YearsSinceLastPromotion` for an employee.  
- **Target Variable:** `YearsSinceLastPromotion`  
- **Features:** Job Level, Performance Rating, Total Working Years, Training, Work-Life Balance, etc.  

---

## 🧾 Dataset
- **Dataset Name:** `Employee-Attrition.csv`  
- **Source:** IBM HR Analytics Dataset (Kaggle)  
- **Rows:** ~1470  
- **Columns:** 35+ features including demographics, compensation, and performance metrics.  

---

## ⚙️ Key Features Engineered
| Feature | Description |
|----------|--------------|
| `TenureBucket` | Categorized `YearsAtCompany` into experience bins |
| `NoPromotionRecently` | 1 if `YearsSinceLastPromotion` > 5 |
| `YearsBeforeManager` | Difference between `YearsAtCompany` and `YearsWithCurrManager` |
| `OvertimeLowPay` | Employees working overtime with below-median income |
| `PromotionGap` | Years since last promotion |
| `ExperienceRatio` | `YearsInCurrentRole / TotalWorkingYears` |
| `HighPerformer` | Binary flag for high performance ratings |
| `TrainingEffect` | Combined impact of training & performance |

---

## 🧩 Workflow

### 🔸 1. Data Preprocessing
- Dropped unnecessary columns (`EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`)
- Encoded categorical variables using `OneHotEncoder`
- Scaled numerical features using `StandardScaler`
- Handled imbalance using **SMOTE** for classification & **RandomUnderSampler** for regression

### 🔸 2. Model Building
#### Classification Models (Attrition)
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  

#### Regression Models (Promotion)
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  

### 🔸 3. Hyperparameter Tuning
- `RandomizedSearchCV` used for model optimization  

### 🔸 4. Evaluation Metrics
**Attrition (Classification):**
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

**Promotion (Regression):**
- MSE  
- RMSE  
- MAE  
- R² Score  

---

