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
- **Goal:** estimate when an employee is likely to be promoted.  
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

## 📈 Evaluation Metrics

### 🧮 Attrition Prediction (Classification)
| Metric | Description |
|--------|--------------|
| Accuracy | Overall correctness of predictions |
| Precision | True positives out of predicted positives |
| Recall | True positives out of actual positives |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Discriminative power of the model |

---

### 📊 Promotion Likelihood (Regression)
| Metric | Description |
|--------|--------------|
| MSE | Mean squared difference between predicted & actual values |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Variance explained by the model |

---

## 🛠️ Tech Stack
| Component | Tool/Language |
|------------|---------------|
| Data Handling | Python (Pandas, NumPy) |
| Machine Learning | scikit-learn, xgboost |
| Visualizations | seaborn, matplotlib |
| Imbalance Handling | imbalanced-learn |
| Deployment | Streamlit |
| Model Serialization | pickle |

---

## 🗂️ File Structure
├── Employee-Attrition.csv
├── employee_dashboard.py          # Streamlit app
├── employee_attrition_promotion.py # Training script
├── best_model_final.pkl
├── best_promotion_model.pkl
├── requirements.txt
└── README.md


