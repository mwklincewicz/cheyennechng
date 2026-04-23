import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer

# Load merged dataset
dir_path = 'C:/thesis_dss'
absolute_path = dir_path + '/Output files/'

merged_df = pd.read_csv(absolute_path + 'Merged_dataset.csv')

# --------------------------------------------------------------------------------
# LONELINESS CALCULATION

# Calculate Loneliness Score with normalization for different scales
loneliness_cols = ['Loneliness_1', 'Loneliness_2', 'Loneliness_3', 'Loneliness_4', 'Loneliness_5', 'Loneliness_6']
for col in loneliness_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

def calculate_loneliness_score(data, loneliness_cols):
    reverse_cols = ['Loneliness_2', 'Loneliness_3', 'Loneliness_4']
    # reverse coding
    for col in reverse_cols:
        data[col]= data[col].apply(lambda x: 4-x if pd.notna(x) else x)
    # code 1=yes or 2=more or less -> 1, 3=no -> 0
    dich_matrix = pd.DataFrame({col: data[col].apply(lambda x: 1 if x in [1,2] else (0 if x == 3 else np.nan))
                                for col in loneliness_cols})
    # sum of scores
    data['Loneliness_Score'] = dich_matrix.sum(axis=1, min_count=6)
    return data

# Calculate loneliness score
merged_df = calculate_loneliness_score(merged_df, loneliness_cols)

# Keep only rows with Loneliness scores
merged_df = merged_df.dropna(subset=['Loneliness_Score'])

# --------------------------------------------------------------------------------
# MAPPING

# 1. Health Perception
health_mapping = {
    'poor': 1,
    'moderate': 2,
    'good': 3,
    'very good': 4,
    'excellent': 5
}
merged_df['Health_Perception'] = merged_df['Health_Perception'].map(health_mapping).fillna(merged_df['Health_Perception'])

# Leisure Satisfaction
satisfaction_mapping = {
    '0 not at all satisfied': 0,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
    '6': 6, '7': 7, '8': 8, '9': 9,
    '10 very satisfied': 10,
    999: np.nan,
    -9: np.nan
}
merged_df['Leisure_Satisfaction'] = merged_df['Leisure_Satisfaction'].replace(satisfaction_mapping)

# Family_Evening, Neighborhood_Evening, Others_Evening
contact_mapping = {
    'almost every day': 1,
    'once or twice a week': 2,
    'a few times per month': 3,
    'about once a month': 4,
    'a number of times per year': 5,
    'about once a year': 6,
    'never': 7,
    8: np.nan,
    9: np.nan
}

for col in ['Family_Evening', 'Neighborhood_Evening', 'Others_Evening']:
    merged_df[col] = merged_df[col].replace(contact_mapping)

# Create combined feature (mean contact frequency)
merged_df['Social_Contact_Score'] = merged_df[
    ['Family_Evening', 'Neighborhood_Evening', 'Others_Evening']
].mean(axis=1, skipna=True)

# Inclusion of others (IOS)
IOS_mapping = {
    'not connected': 1,
    '2': 2, '3': 3, '4': 4, '5': 5,
    '6': 6, 'very connected': 7
}
merged_df['IOS'] = merged_df['IOS'].replace(IOS_mapping)

# Age category mapping
age_mapping = {
    '15 - 24 years': '15-24',
    '25 - 34 years': '25-34',
    '35 - 44 years': '35-44',
    '45 - 54 years': '45-54',
    '55 - 64 years': '55-64',
    '65 years and older': '65+'
}
merged_df['Age category'] = merged_df['Age category'].replace(age_mapping)

# Nr household members mapping
household_mapping = {
    'One person': 1,
    'Two persons': 2,
    'Three persons': 3,
    'Four persons': 4,
    'Five persons': 5,
    'Six persons': 6,
    'Seven persons': 7,
    'Eight persons': 8,
    'Nine persons or more': 9
}
merged_df['Nr household members'] = merged_df['Nr household members'].replace(household_mapping)

# Gender
gender_mapping = {
    'Male': 0,
    'Female': 1,
    'Other': 2
}
merged_df['Gender_Code'] = merged_df['Gender'].replace(gender_mapping)

# Partner
partner_mapping = {
    'Yes': 1,
    'No': 0
}
merged_df['Partner'] = merged_df['Partner'].replace(partner_mapping)

# Urbanity
merged_df['Urbanity'] = merged_df['Urbanity'].replace('-99.0', np.nan)  # Replace -99.0 with NaN
urbanity_mapping = {
    'Not urban': 1,
    'Slightly urban': 2,
    'Moderately urban': 3,
    'Very urban': 4,
    'Extremely urban': 5,
    -99: np.nan
}
merged_df['Urbanity'] = merged_df['Urbanity'].replace(urbanity_mapping)

# Education Level
education_mapping = {
    'primary school': 1,
    'vmbo (intermediate secondary education, US: junior high school)': 2,
    'havo/vwo (higher secondary education/preparatory university education, US: senio': 3,
    'mbo (intermediate vocational education, US: junior college)': 4,
    'hbo (higher vocational education, US: college)': 5,
    'wo (university)': 6
}
merged_df['Education Level'] = merged_df['Education Level'].replace(education_mapping)

# Children living-at-home
children_mapping = {
    'One child': 1,
    'Two children': 2,
    'Three children': 3,
    'Four children': 4,
    'Five children': 5,
    'Six children': 6
}
merged_df['Children living-at-home'] = merged_df['Children living-at-home'].replace(children_mapping)

# Occupation
def simplify_occupation(occupation):  # missing values
    if pd.isna(occupation):
        return np.nan

occupation_mapping_numeric = {
    1: 'employed',  # Paid employment
    2: 'employed',  # Works or assists in family business
    3: 'employed',  # Autonomous professional, freelancer, self-employed
    4: 'unemployed',  # Jobseeker following job loss
    5: 'unemployed',  # First-time jobseeker
    6: 'unemployed',  # Exempted from job seeking
    7: 'student',  # Attends school or studying
    8: 'homemaker',  # Takes care of housekeeping
    9: 'retired',  # Pensioner / old age
    10: 'disabled',  # (Partial) work disability
    11: 'volunteer',  # Unpaid work with unemployment benefit
    12: 'volunteer',  # Voluntary work
    13: 'other',  # Something else
    14: 'too young'  # Too young
}
merged_df['Occupation_Text'] = merged_df['Occupation'].map(occupation_mapping_numeric)

occupation_code_mapping = {
    'employed': 1,
    'student': 2,
    'retired': 3,
    'unemployed': 4,
    'disabled': 5,
    'homemaker': 6,
    'volunteer': 7,
    'too young': 8,
    'other': 9
}
merged_df['Occupation_Code'] = merged_df['Occupation_Text'].replace(occupation_code_mapping)

# ------------------------------------------------------------------------------------------------------
# MISSING DATA IMPUTATION

# Count observations per participant
observation_counts = merged_df.groupby('nomem_encr')['Loneliness_Score'].count()

# Keep only participants with 2+ observations
valid_subjects = observation_counts[observation_counts >= 2].index
merged_df = merged_df[merged_df['nomem_encr'].isin(valid_subjects)]

# Missing values imputation
numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
if 'Loneliness' in numeric_cols:
    numeric_cols.remove('Loneliness')

def impute(group):
    # last and next average
    group = group.interpolate(method="linear")

    # last observation carried forward
    group = group.ffill()

    # next observation carried backward
    group = group.bfill()

    return group

merged_df[numeric_cols] = (
    merged_df.groupby("nomem_encr")[numeric_cols]
    .apply(impute)
    .reset_index(level=0, drop=True)
)

cat_cols = ['Health_Perception', 'Family_Evening', 'Neighborhood_Evening', 'Others_Evening',
            'Education Level', 'Occupation_Code', 'IOS', 'Urbanity']
merged_df[cat_cols] = merged_df.groupby('nomem_encr')[cat_cols].ffill().bfill()

# Keep only participants with <=5% missing
missing_subjects = merged_df.groupby('nomem_encr')[numeric_cols].apply(
    lambda x: x.isna().mean().mean()
)

valid_subjects = missing_subjects[missing_subjects <= 0.05].index
merged_df = merged_df[merged_df['nomem_encr'].isin(valid_subjects)]

# multiple imputation in the rest
imputer = IterativeImputer(max_iter=5, random_state=42)
# Fit and transform the imputer on the remaining data
merged_df[numeric_cols] = imputer.fit_transform(merged_df[numeric_cols])

# ----------------------------------------------------------------------------------------------------
# MHI-5 scores calculation
mhi_cols = ['MHI_1', 'MHI_2', 'MHI_3', 'MHI_4', 'MHI_5']
def calculate_mhi_score(data, mhi_cols):
    reverse_cols = ['MHI_3', 'MHI_5']
    # reverse coding
    for col in reverse_cols:
        data[col] = data[col].apply(lambda x: 7 - x if pd.notna(x) else np.nan)
    data['MHI_Score'] = data[mhi_cols].mean(axis=1, skipna=True)
    return data
merged_df = calculate_mhi_score(merged_df, mhi_cols)

# 2. Life satisfaction
life_satisfaction_cols = ['LS_1', 'LS_2', 'LS_3', 'LS_4', 'LS_5']
for col in life_satisfaction_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

def calculate_ls_score(data, life_satisfaction_cols):
    data['Life_Satisfaction_Score'] = data[life_satisfaction_cols].mean(axis=1, skipna=True)
    return data
merged_df = calculate_ls_score(merged_df, life_satisfaction_cols)

# Big 5
# Define reverse scoring function
def reverse_score_5_likert(x):
    return 6 - x if pd.notna(x) else x

# Define items per trait
traits = {
    'Extraversion': {
        'pos': ['BIG-V_1','BIG-V_11','BIG-V_21','BIG-V_31','BIG-V_41'],
        'neg': ['BIG-V_6','BIG-V_16','BIG-V_26','BIG-V_36','BIG-V_46']
    },
    'Agreeableness': {
        'pos': ['BIG-V_7','BIG-V_17','BIG-V_27','BIG-V_37','BIG-V_42','BIG-V_47'],
        'neg': ['BIG-V_2','BIG-V_12','BIG-V_22','BIG-V_32']
    },
    'Conscientiousness': {
        'pos': ['BIG-V_3','BIG-V_13','BIG-V_23','BIG-V_33','BIG-V_43','BIG-V_48'],
        'neg': ['BIG-V_8','BIG-V_18','BIG-V_28','BIG-V_38']
    },
    'Emotional_Stability': {
        'pos': ['BIG-V_9','BIG-V_19'],
        'neg': ['BIG-V_4','BIG-V_14','BIG-V_24','BIG-V_29','BIG-V_34','BIG-V_39','BIG-V_44','BIG-V_49']
    },
    'Intellect_Imagination': {
        'pos': ['BIG-V_5','BIG-V_15','BIG-V_25','BIG-V_35','BIG-V_40','BIG-V_45','BIG-V_50'],
        'neg': ['BIG-V_10','BIG-V_20','BIG-V_30']
    }
}

# Reverse negative items
for trait, items in traits.items():
    for col in items['neg']:
        merged_df[col] = merged_df[col].apply(reverse_score_5_likert)

# Average scores per trait
for trait, items in traits.items():
    merged_df[trait] = merged_df[items['pos'] + items['neg']].mean(axis=1, skipna=True)

# Calculate self-esteem score
self_esteem_cols = ['Self-esteem_1', 'Self-esteem_2', 'Self-esteem_3', 'Self-esteem_4', 'Self-esteem_5',
                    'Self-esteem_6',
                    'Self-esteem_7', 'Self-esteem_8', 'Self-esteem_9', 'Self-esteem_10']
def calculate_selfesteem_score(data, self_esteem_cols):
    reverse_cols = ['Self-esteem_3', 'Self-esteem_5', 'Self-esteem_8', 'Self-esteem_9', 'Self-esteem_10']
    # reverse coding
    for col in reverse_cols:
        data[col] = data[col].apply(lambda x: 8 - x if pd.notna(x) else np.nan)
    data['Self_esteem_Score'] = data[self_esteem_cols].mean(axis=1, skipna=True)
    return data
merged_df = calculate_selfesteem_score(merged_df, self_esteem_cols)

# Calculate optimism score
optimism_cols = ['Optimism_1', 'Optimism_2', 'Optimism_3', 'Optimism_4', 'Optimism_5', 'Optimism_6',
                 'Optimism_7', 'Optimism_8', 'Optimism_9', 'Optimism_10']

def calculate_optimism_score(data, optimism_cols):
    reverse_cols = ['Optimism_3', 'Optimism_7', 'Optimism_9']
    # reverse coding
    for col in reverse_cols:
        data[col] = data[col].apply(lambda x: 6 - x if pd.notna(x) else np.nan)
    data['Optimism_Score'] = data[optimism_cols].mean(axis=1, skipna=True)
    return data
merged_df = calculate_optimism_score(merged_df, optimism_cols)

# ---------------------------------------------------------------------------------------------------
# SAVE CLEANED DATASET

columns_to_keep = [
    'nomem_encr', 'Year', 'Age', 'Nr household members', 'Gender', 'Partner',
    'Household Income', 'Urbanity', 'Education Level', 'Children living-at-home',
    'Occupation_Code', 'Health_Perception', 'Leisure_Satisfaction',
    'Social_Contact_Score',
    'MHI_Score', 'Life_Satisfaction_Score', 'Extraversion', 'Agreeableness',
    'Conscientiousness', 'Emotional_Stability', 'Intellect_Imagination',
    'Self_esteem_Score', 'IOS', 'Optimism_Score', 'Loneliness_Score'
]

cleaned_df = merged_df[columns_to_keep].copy()
cleaned_df.to_csv(absolute_path + 'Cleaned_dataset.csv', index=False)

# ----------------------------------------------------------------------------------------------------
# CALCULATE DERIVATIVE
# Computes the rate of change in loneliness per year between observations
cleaned_df['Loneliness_Slope'] = (
    cleaned_df.groupby('nomem_encr')['Loneliness_Score'].diff()
    / cleaned_df.groupby('nomem_encr')['Year'].diff()
)

# lagged slope
cleaned_df["Loneliness_Change"] = (
    cleaned_df.groupby("nomem_encr")["Loneliness_Score"].diff()
)

# Drop first observation per participant
cleaned_df = cleaned_df.dropna(subset=['Loneliness_Slope'])

# Move Loneliness_Score to last column
cols = [col for col in cleaned_df.columns if col != 'Loneliness_Score'] + ['Loneliness_Score']
cleaned_df = cleaned_df[cols]

# ---------------------------------------------------------------------------------------------------
# SAVE SLOPE DATASET

cleaned_df.to_csv(absolute_path + 'slope_dataset.csv', index=False)

