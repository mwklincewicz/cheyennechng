import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    for col in loneliness_cols:
        data[col + '_dich'] = data[col].apply(lambda x: 1 if x in [1,2] else (0 if x == 3 else np.nan))

    # sum of scores
    dich_cols = [col + '_dich' for col in loneliness_cols]
    data['Loneliness_Score'] = data[dich_cols].sum(axis=1, min_count=6)
    return data


# Calculate life loneliness score
merged_df = calculate_loneliness_score(merged_df, loneliness_cols)



# 1. MHI-5 scores calculation
MHI_cols = ['MHI_1', 'MHI_2', 'MHI_3', 'MHI_4', 'MHI_5']
for col in MHI_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

def calculate_mhi_score(data, MHI_cols):
    reverse_cols = ['MHI_3', 'MHI_5']
    # reverse coding
    for col in reverse_cols:
        data[col] = data[col].apply(lambda x: 7 - x if pd.notna(x) else np.nan)
    data['MHI_score'] = data[MHI_cols].mean(axis=1, skipna=True)
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


# 3. Big 5
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


# 4. Self-esteem
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

# 5. Optimism
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