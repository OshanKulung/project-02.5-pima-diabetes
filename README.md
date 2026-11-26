# Pima Indians Diabetes Prediction

This repository contains an end-to-end Machine Learning assessment to predict whether a patient has diabetes based on diagnostic measurements. 

## The Challenge
The dataset contained a known trap: missing values were encoded as `0` in medical columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI), which is biologically impossible. Additionally, the target variable was imbalanced (~65% non-diabetic, ~35% diabetic).

## Approach
1. **Data Cleaning:** Replaced `0` values with `NaN` and imputed them using the column median.
2. **Preprocessing:** Applied `StandardScaler` to the features and used `stratify=y` during the 80/20 train-test split to maintain the class distribution.
3. **Modeling:**
   - **Logistic Regression:** Used `class_weight='balanced'` to handle the imbalance. Achieved 73.38% accuracy.
   - **XGBoost:** Used `scale_pos_weight` to penalize minority class errors. Achieved 75.97% accuracy.
4. **Tuning:** Used `GridSearchCV` with 5-fold Cross-Validation. The tuned XGBoost model improved to **76.62% accuracy**.

## Key Insights & Business Value
Using `feature_importances_` from the tuned XGBoost model, the top 3 predictors of diabetes are:
1. **Glucose:** The strongest indicator. 
2. **BMI:** Body weight plays a significant role in insulin sensitivity.
3. **Age:** Susceptibility increases with age.

**Medical Advice:** Patients should focus on controlling blood sugar, maintaining a healthy body weight, staying physically active, and undergoing regular medical screenings as they get older.

## Files
- `pima_diabetes_eda_ml.ipynb`: The complete Jupyter Notebook containing EDA, preprocessing, modeling, tuning, and insights.