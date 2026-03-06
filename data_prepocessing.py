import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set display options
pd.set_option("display.max.columns", None)
pd.set_option('display.width', 180)
pd.set_option("display.precision", 4)
pd.set_option('future.no_silent_downcasting', True)

# Load merged dataset
dir_path = 'C:/thesis_dss'
absolute_path = dir_path + '/Output files/'

merged_df = pd.read_csv(absolute_path + 'Merged_dataset.csv')

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


# Calculate life loneliness score
merged_df = calculate_loneliness_score(merged_df, loneliness_cols)

# 1. Health Perception
health_mapping = {
    'poor': 1,
    'moderate': 2,
    'good': 3,
    'very good': 4,
    'excellent': 5
}

merged_df['Health_Perception'] = merged_df['Health_Perception'].map(health_mapping).fillna(merged_df['Health_Perception'])

# Social Satisfaction
satisfaction_mapping = {
    '0 not at all satisfied': 0,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 
    '6': 6, '7': 7, '8': 8, '9': 9, 
    '10 very satisfied': 10,
    999: np.nan,
    -9: np.nan
}

merged_df['Social_Satisfaction'] = merged_df['Social_Satisfaction'].replace(satisfaction_mapping)

# Leisure Satisfaction
merged_df['Leisure_Satisfaction'] = merged_df['Leisure_Satisfaction'].replace(satisfaction_mapping)


# MHI-5 scores calculation
MHI_cols = ['MHI_1', 'MHI_2', 'MHI_3', 'MHI_4', 'MHI_5']
for col in MHI_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

def calculate_mhi_score(data, MHI_cols):
    reverse_cols = ['MHI_3', 'MHI_5']
    # reverse coding
    for col in reverse_cols:
        data[col] = data[col].apply(lambda x: 7 - x if pd.notna(x) else np.nan)
    data['MHI_Score'] = data[MHI_cols].mean(axis=1, skipna=True)
    return data

merged_df = calculate_mhi_score(merged_df, MHI_cols)

# 2. Life satisfaction
Life_Satisfaction_cols = ['LS_1', 'LS_2', 'LS_3', 'LS_4', 'LS_5']
for col in Life_Satisfaction_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

def calculate_ls_score(data, Life_Satisfaction_cols):
    data['Life_Satisfaction_Score'] = data[Life_Satisfaction_cols].mean(axis=1, skipna=True)
    return data

merged_df = calculate_ls_score(merged_df, Life_Satisfaction_cols)


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


# Self-esteem
self_esteem_cols = ['Self-esteem_1', 'Self-esteem_2', 'Self-esteem_3', 'Self-esteem_4', 'Self-esteem_5', 'Self-esteem_6', 
                 'Self-esteem_7', 'Self-esteem_8', 'Self-esteem_9', 'Self-esteem_10']
for col in self_esteem_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

def calculate_selfesteem_score(data, self_esteem_cols):
    reverse_cols = ['Self-esteem_3', 'Self-esteem_5', 'Self-esteem_8', 'Self-esteem_9', 'Self-esteem_10']
    # reverse coding
    for col in reverse_cols:
        data[col] = data[col].apply(lambda x: 8 - x if pd.notna(x) else np.nan)
    data['Self_esteem_Score'] = data[self_esteem_cols].mean(axis=1, skipna=True)
    return data

merged_df = calculate_selfesteem_score(merged_df, self_esteem_cols)

# Optimism
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
merged_df['Family_Evening'] = merged_df['Family_Evening'].replace(contact_mapping)
merged_df['Neighborhood_Evening'] = merged_df['Neighborhood_Evening'].replace(contact_mapping)
merged_df['Others_Evening'] = merged_df['Others_Evening'].replace(contact_mapping)


# Subjective Happiness
subjective_happiness_mapping = {
    '0 totally unhappy': 0,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 
    '6': 6, '7': 7, '8': 8, '9': 9, 
    '10 totally happy': 10,
    999: np.nan,
    -9: np.nan
}

merged_df['Subjective_Happiness'] = merged_df['Subjective_Happiness'].replace(subjective_happiness_mapping)

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
def simplify_occupation(occupation):        # missing values
    if pd.isna(occupation):
        return np.nan
        
occupation_mapping_numeric = {
    1: 'employed',   # Paid employment
    2: 'employed',   # Works or assists in family business
    3: 'employed',   # Autonomous professional, freelancer, self-employed
    4: 'unemployed', # Job seeker following job loss
    5: 'unemployed', # First-time job seeker
    6: 'unemployed', # Exempted from job seeking
    7: 'student',    # Attends school or studying
    8: 'homemaker',  # Takes care of housekeeping
    9: 'retired',    # Pensioner / old age
    10: 'disabled',  # (Partial) work disability
    11: 'volunteer', # Unpaid work with unemployment benefit
    12: 'volunteer', # Voluntary work
    13: 'other',     # Something else
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

output_file = absolute_path + 'Cleaned_dataset.csv'
merged_df.to_csv(output_file)
