# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import datetime
import warnings

# Define the data preparation function
def prepare_data(df):
    # Clean column names: lowercase, strip spaces, and replace spaces with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convert percentage fields from string/int to float
    df['int_rate'] = df['int_rate'] / 100
    df['revol_util'] = df['revol_util'] / 100
    df['dti'] = df['dti'] / 100

    # Extract numeric value from term string (e.g., '36 months' → 36)
    df['term'] = df['term'].str.extract(r'(\d+)').astype(float)

    # Map employment length from string to numeric values
    emp_length_map = {
        '10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7,
        '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3,
        '2 years': 2, '1 year': 1, '< 1 year': 0, 'n/a': None
    }
    df['emp_length'] = df['emp_length'].map(emp_length_map)

    # Parse date columns and fix errors due to incorrect century
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%d-%m-%y', errors='coerce')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%d-%m-%y', errors='coerce')
    df.loc[df['issue_d'].dt.year > 2023, 'issue_d'] -= pd.DateOffset(years=100)
    df.loc[df['earliest_cr_line'].dt.year > 2023, 'earliest_cr_line'] -= pd.DateOffset(years=100)

    # Drop irrelevant or high-cardinality text columns
    df.drop(columns=[col for col in ['emp_title', 'title'] if col in df.columns], inplace=True, errors='ignore')

    # Fill missing values
    df['emp_length'] = df['emp_length'].fillna(df['emp_length'].median())
    df['mort_acc'] = df['mort_acc'].fillna(df['mort_acc'].median())
    df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(0)
    df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())

    # Clip extreme values to limit outliers
    df['dti'] = df['dti'].clip(upper=100)
    df['pub_rec'] = df['pub_rec'].clip(upper=10)
    df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].clip(upper=5)

    # Cap numerical outliers at the 99.5th percentile
    for col in ['mort_acc', 'open_acc', 'total_acc', 'revol_util']:
        thresh = df[col].quantile(0.995)
        df[col] = df[col].clip(upper=thresh)

    # Log-transform income, balance, and installment
    df['annual_inc_log'] = np.log1p(df['annual_inc'])
    df['revol_bal_log'] = np.log1p(df['revol_bal'])
    df['installment_log'] = np.log1p(df['installment'])

    # Create derived features
    df['credit_age_months'] = ((df['issue_d'] - df['earliest_cr_line']) / pd.Timedelta(days=30)).round(1)
    df['annual_inc_safe'] = df['annual_inc'].replace(0, 1)  # Avoid division by 0
    df['low_income_flag'] = (df['annual_inc'] < 10000).astype(int)

    # Cap annual income and installment
    annual_inc_cap = df['annual_inc_safe'].quantile(0.995)
    installment_cap = df['installment'].quantile(0.995)
    df['annual_inc_capped'] = df['annual_inc_safe'].clip(upper=annual_inc_cap)
    df['installment_capped'] = df['installment'].clip(upper=installment_cap)

    # Compute ratios related to affordability
    df['installment_to_monthly_income'] = df['installment_capped'] / (df['annual_inc_capped'] / 12 + 1)
    df['loan_to_annual_income'] = df['loan_amnt'] / (df['annual_inc_capped'] + 1)
    df['loan_to_monthly_income'] = df['loan_amnt'] / (df['annual_inc_capped'] / 12 + 1)

    # Cap those ratios
    for col in ['installment_to_monthly_income', 'loan_to_annual_income', 'loan_to_monthly_income']:
        threshold = df[col].quantile(0.995)
        df[col] = df[col].clip(upper=threshold)

    # More feature engineering
    df['loan_amnt_term_income_ratio'] = df['loan_amnt'] / ((df['annual_inc_capped'] + 1) * (df['term'] / 12))
    df['loan_amnt_term_income_ratio'] = df['loan_amnt_term_income_ratio'].clip(
        upper=df['loan_amnt_term_income_ratio'].quantile(0.995))

    df['revol_util_per_acc'] = df['revol_util'] / (df['open_acc'] + 1)
    df['revol_util_per_acc'] = df['revol_util_per_acc'].clip(upper=df['revol_util_per_acc'].quantile(0.995))

    df['credit_age_term_ratio'] = df['credit_age_months'] / (df['term'] + 1)
    df['mortgage_acc_ratio'] = df['mort_acc'] / (df['open_acc'] + 1)
    df['mortgage_acc_ratio'] = df['mortgage_acc_ratio'].clip(upper=df['mortgage_acc_ratio'].quantile(0.995))

    #Encode categorical variables

    # Map grade and sub-grade to numeric values
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df['grade_num'] = df['grade'].apply(lambda x: grade_order.index(x))

    subgrades_ordered = sorted(df['sub_grade'].dropna().unique(), key=lambda x: (x[0], int(x[1])))
    subgrade_map = {k: i for i, k in enumerate(subgrades_ordered)}
    df['sub_grade_num'] = df['sub_grade'].map(subgrade_map)

    # Clean and encode categorical variables
    df['home_ownership_clean'] = df['home_ownership'].replace({
        'OTHER': 'OTHER_RARE',
        'NONE': 'OTHER_RARE',
        'ANY': 'OTHER_RARE'
    })
    df = pd.get_dummies(df, columns=['home_ownership_clean'], prefix='home', drop_first=True)
    df = pd.get_dummies(df, columns=['verification_status'], drop_first=True)

    # Group rare purposes into a single category and encode
    freq = df['purpose'].value_counts(normalize=True)
    rare_purposes = freq[freq < 0.01].index
    df['purpose_clean'] = df['purpose'].apply(lambda x: 'OTHER_RARE' if x in rare_purposes else x)
    df = pd.get_dummies(df, columns=['purpose_clean'], drop_first=True)

    # Binary encoding of list status and application type
    df['initial_list_status_bin'] = (df['initial_list_status'] == 'w').astype(int)
    df['application_type_clean'] = df['application_type'].apply(lambda x: x if x == 'INDIVIDUAL' else 'OTHER')
    df['application_type_bin'] = (df['application_type_clean'] == 'OTHER').astype(int)

    # Define final features to return
    final_features = [
        'loan_amnt', 'term', 'int_rate', 'annual_inc_log', 'emp_length', 'dti', 'revol_bal_log', 'total_acc',
        'pub_rec_bankruptcies', 'mortgage_acc_ratio', 'credit_age_term_ratio', 'loan_amnt_term_income_ratio',
        'initial_list_status_bin', 'verification_status_Source Verified', 'home_RENT', 'home_OWN', 'home_OTHER_RARE',
        'application_type_bin', 'purpose_clean_credit_card', 'purpose_clean_debt_consolidation',
        'purpose_clean_small_business', 'revol_util_per_acc'
    ]
    # Ensure all expected features exist, filling missing ones with 0
    for col in final_features:
        if col not in df.columns:
            df[col] = 0

    return df[final_features]
