import os
import pandas as pd
import pyreadstat

# Paths
dir_path = 'C:/thesis_dss'
absolute_path = os.path.join(dir_path, 'LISS Surveys')

# Data files for each LISS Survey wave stored as STATA files (Social Integration & Leisure, Health, Personality, and Background vars)
files = ['ch08b_EN_1.3p.dta', 'ch09c_EN_1.1p.dta', 'ch10d_EN_1.0p.dta', 'ch11e_EN_1.0p.dta', 'ch12f_EN_1.0p.dta', 'ch13g_EN_1.0p.dta', 'ch15h_EN_1.2p.dta',
         'ch16i_EN_1.0p.dta', 'ch17j_EN_1.0p.dta', 'ch18k_EN_1.0p.dta', 'ch19l_EN_1.0p.dta', 'ch20m_EN_1.0p.dta', 'ch21n_EN_1.0p.dta', 'ch23p_EN_1.0p.dta', 'ch24q_EN_1.1p.dta',
         'cp08a_1p_EN.dta', 'cp09b_1.0p_EN.dta', 'cp10c_1.0p_EN.dta', 'cp11d_1.0p_EN.dta', 'cp12e_1.0p_EN.dta', 'cp13f_EN_1.0p.dta', 'cp14g_EN_1.0p.dta', 'cp15h_EN_1.0p.dta',
         'cp17i_EN_1.0p.dta', 'cp18j_EN_1.0p.dta', 'cp19k_EN_1.0p.dta', 'cp20l_EN_1.0p.dta', 'cp21m_EN_1.0p.dta', 'cp22n_EN_1.0p.dta', 'cp23o_EN_1.0p.dta', 'cp24p_EN_1.0p.dta',
         'cs08a_2p_EN.dta', 'cs09b_1p_EN.dta', 'cs10c_1p_EN.dta', 'cs11d_EN_3.0p.dta', 'cs12e_1.0p_EN.dta', 'cs15h_EN_1.0p.dta', 'cs16i_EN_1.0p.dta',
         'cs17j_EN_1.0p.dta', 'cs18k_EN_1.0p.dta', 'cs19l_EN_1.0p.dta', 'cs20m_EN_1.1p.dta', 'cs21n_EN_1.1p.dta', 'cs22o_EN_1.1p.dta', 'cs23p_EN_1.0p.dta', 'cs24q_EN_1.0p.dta',
         'avars_200812_EN_2.0p.dta', 'avars_200912_EN_2.0p.dta', 'avars_201012_EN_2.0p.dta', 'avars_201112_EN_2.0p.dta', 'avars_201212_EN_1.0p.dta', 'avars_201312_EN_1.0p.dta',
         'avars_201412_EN_1.0p.dta', 'avars_201512_EN_1.0p.dta', 'avars_201612_EN_1.0p.dta', 'avars_201712_EN_1.0p.dta', 'avars_201812_EN_1.0p.dta',
         'avars_201912_EN_1.0p.dta', 'avars_202012_EN_1.0p.dta', 'avars_202112_EN_1.1p.dta', 'avars_202212_EN_1.0p.dta', 'avars_202312_EN_1.0p.dta', 'avars_202412_EN_1.0p.dta',
         ]

# List to store dataframes that will be merged
dataframes = []


# 3 files that are not compatible with read_stata -> pyreadstat
sav_files = ['ch22o_EN_1.0p.sav', 'cs13f_2.0p_EN.sav', 'cs14g_EN_2.0.sav']

for file in sav_files:
    filepath = os.path.join(absolute_path, file)
    try:
        df, meta = pyreadstat.read_sav(filepath)
    except Exception as e:
        print(f"Warning: Could not read {file}: {e}")
        continue

    if file[:2] =='ch':
        vars = {
        'nomem_encr': 'nomem_encr',
        'Health_Perception': file[:5] + '004',  # Measures person's own perception of their health, generally.
        # Mental Health Inventory
        'MHI_1': file[:5] + '011',   # I felt very anxious.
        'MHI_2': file[:5] + '012',   # I felt so down that nothing could cheer me up.
        'MHI_3': file[:5] + '013',   # I felt calm and peaceful.
        'MHI_4': file[:5] + '014',   # I felt depressed and gloomy
        'MHI_5': file[:5] + '015',   # I felt happy
        }
        df = df[[v for v in vars.values() if v in df.columns]]
        df.rename(columns={v:k for (k,v) in vars.items() if k != 'nomem_encr' and v in df.columns}, inplace=True)
        df['Year'] = '20' + file[2:4]

    if file[:2] =='cs':
        vars = {
        'nomem_encr': 'nomem_encr',
        'Leisure_Satisfaction': file[:5] + '001', # How satisfied are you with the amount of leisure time that you have?
        # Loneliness Scale
        'Loneliness_1': file[:5] + '284',  # I have a sense of emptiness around me.
        'Loneliness_2': file[:5] + '285',  # There are enough people I can count on in case of a misfortune.
        'Loneliness_3': file[:5] + '286',  # I know a lot of people that I can fully rely on.
        'Loneliness_4': file[:5] + '287',  # There are enough people to whom I feel closely connected.
        'Loneliness_5': file[:5] + '288',  # I miss having people around me.
        'Loneliness_6': file[:5] + '289',  # I often feel deserted.
        # Social Contacts
        'Social_Satisfaction' : file[:5] + '283',   # How satisfied are you with your social contacts?
        'Family_Evening': file[:5] + '290',         # Spend an evening with family
        'Neighborhood_Evening': file[:5] + '291',   # Spend an evening with someone from the neighborhood
        'Others_Evening': file[:5] + '292',         # Spend an evening with someone outside your neighborhood
        }
        df = df[[v for v in vars.values() if v in df.columns]]
        df.rename(columns={v:k for (k,v) in vars.items() if k != 'nomem_encr' and v in df.columns}, inplace=True)
        df['Year'] = '20' + file[2:4]

    dataframes.append(df)

# Existing .dta processing loop
def read_dta_file(filepath):
    try:
        df, meta = pyreadstat.read_dta(filepath)
        return df
    except UnicodeDecodeError:
        try:
            df, meta = pyreadstat.read_dta(filepath, encoding='latin1')
            print(f"Warning: Read {os.path.basename(filepath)} using latin1 encoding.")
            return df
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
            return None
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None

# Select relevant variables from LISS surveys
for file in files:
    filepath = os.path.join(absolute_path, file)
    df = read_dta_file(filepath)
    if df is None:
        continue

    if file[:2] == 'av':
        vars = {
        'nomem_encr': 'nomem_encr',
        'Age': 'leeftijd',                      # Measures age in numbers.
        'Age category': 'lftdcat',              # Age in CBS categories.
        'Nr household members': 'aantalhh',     # Number of household members.
        'Gender': 'geslacht',                   # Measures biological sex.
        'Partner': 'partner',                   # Living with partner (wedded or unwedded).
        'Household Income': 'brutohh_f',        # Measures gross household income.
        'Urbanity': 'sted',                     # Measures the urban character of place of residence. A 'semi'-geolocation indicator.
        'Education Level': 'oplcat',            # Measures level of education in CBS (Statistics Netherlands) categories.
        'Children living-at-home': 'aantalki',  # Number of children living-at-home in the household.
        'Occupation': 'belbezig',               # Primary occupation
        }

    if file[:2] =='ch':
        vars = {
        'nomem_encr': 'nomem_encr',
        'Health_Perception': file[:5] + '004',  # Measures person's own perception of their health, generally.
        # Mental Health Inventory
        'MHI_1': file[:5] + '011',   # I felt very anxious.
        'MHI_2': file[:5] + '012',   # I felt so down that nothing could cheer me up.
        'MHI_3': file[:5] + '013',   # I felt calm and peaceful.
        'MHI_4': file[:5] + '014',   # I felt depressed and gloomy
        'MHI_5': file[:5] + '015',   # I felt happy
        }

    if file[:2] =='cs':
        vars = {
        'nomem_encr': 'nomem_encr',
        'Leisure_Satisfaction': file[:5] + '001', # How satisfied are you with the amount of leisure time that you have?
        # Loneliness Scale
        'Loneliness_1': file[:5] + '284',  # I have a sense of emptiness around me.
        'Loneliness_2': file[:5] + '285',  # There are enough people I can count on in case of a misfortune.
        'Loneliness_3': file[:5] + '286',  # I know a lot of people that I can fully rely on.
        'Loneliness_4': file[:5] + '287',  # There are enough people to whom I feel closely connected.
        'Loneliness_5': file[:5] + '288',  # I miss having people around me.
        'Loneliness_6': file[:5] + '289',  # I often feel deserted.
        # Social Contacts
        'Social_Satisfaction' : file[:5] + '283',   # How satisfied are you with your social contacts?
        'Family_Evening': file[:5] + '290',         # Spend an evening with family
        'Neighborhood_Evening': file[:5] + '291',   # Spend an evening with someone from the neighborhood
        'Others_Evening': file[:5] + '292',         # Spend an evening with someone outside your neighborhood
        }

    if file[:2] =='cp':
        vars = {
        'nomem_encr': 'nomem_encr',
        # Subjective Happiness
        'Subjective_Happiness': file[:5] + '010', # On the whole, how happy would you say you are?

        # Satisfaction with life (Diener)
        'LS_1': file[:5] + '014',   # In most ways my life is close to my ideal
        'LS_2': file[:5] + '015',   # The conditions of my life are excellent
        'LS_3': file[:5] + '016',   # I am satisfied with my life
        'LS_4': file[:5] + '017',   # So far I have gotten the important things I want in life
        'LS_5': file[:5] + '018',   # If I could live my life over, I would change almost nothing  

        # BIG-V (Goldberg et al., 1992) #cp___020 - cp___069 See: https://ipip.ori.org/New_IPIP-50-item-scale.htm
        'BIG-V_1': file[:5] + '020', 'BIG-V_2': file[:5] + '021', 'BIG-V_3': file[:5] + '022',
        'BIG-V_4': file[:5] + '023', 'BIG-V_5': file[:5] + '024',
        'BIG-V_6': file[:5] + '025', 'BIG-V_7': file[:5] + '026', 'BIG-V_8': file[:5] + '027',
        'BIG-V_9': file[:5] + '028', 'BIG-V_10': file[:5] + '029',
        'BIG-V_11': file[:5] + '030', 'BIG-V_12': file[:5] + '031', 'BIG-V_13': file[:5] + '032',
        'BIG-V_14': file[:5] + '033', 'BIG-V_15': file[:5] + '034',
        'BIG-V_16': file[:5] + '035', 'BIG-V_17': file[:5] + '036', 'BIG-V_18': file[:5] + '037',
        'BIG-V_19': file[:5] + '038', 'BIG-V_20': file[:5] + '039',
        'BIG-V_21': file[:5] + '040', 'BIG-V_22': file[:5] + '041', 'BIG-V_23': file[:5] + '042',
        'BIG-V_24': file[:5] + '043', 'BIG-V_25': file[:5] + '044',
        'BIG-V_26': file[:5] + '045', 'BIG-V_27': file[:5] + '046', 'BIG-V_28': file[:5] + '047',
        'BIG-V_29': file[:5] + '048', 'BIG-V_30': file[:5] + '049',
        'BIG-V_31': file[:5] + '050', 'BIG-V_32': file[:5] + '051', 'BIG-V_33': file[:5] + '052',
        'BIG-V_34': file[:5] + '053', 'BIG-V_35': file[:5] + '054',
        'BIG-V_36': file[:5] + '055', 'BIG-V_37': file[:5] + '056', 'BIG-V_38': file[:5] + '057',
        'BIG-V_39': file[:5] + '058', 'BIG-V_40': file[:5] + '059',
        'BIG-V_41': file[:5] + '060', 'BIG-V_42': file[:5] + '061', 'BIG-V_43': file[:5] + '062',
        'BIG-V_44': file[:5] + '063', 'BIG-V_45': file[:5] + '064',
        'BIG-V_46': file[:5] + '065', 'BIG-V_47': file[:5] + '066', 'BIG-V_48': file[:5] + '067',
        'BIG-V_49': file[:5] + '068', 'BIG-V_50': file[:5] + '069',

        # Self-esteem, Rosenberg (1965): cp___070 - cp___079 See: https://fetzer.org/sites/default/files/images/stories/pdf/selfmeasures/Self_Measures_for_Self-Esteem_ROSENBERG_SELF-ESTEEM.pdf
        'Self-esteem_1': file[:5] + '070', 'Self-esteem_2': file[:5] + '071', 'Self-esteem_3': file[:5] + '072',
        'Self-esteem_4': file[:5] + '073', 'Self-esteem_5': file[:5] + '074',
        'Self-esteem_6': file[:5] + '075', 'Self-esteem_7': file[:5] + '076', 'Self-esteem_8': file[:5] + '077',
        'Self-esteem_9': file[:5] + '078', 'Self-esteem_10': file[:5] + '079',

        # Inclusion of Others in the Self scale (IOS)
        'IOS': file[:5] + '135', # Please indicate to what extent you generally feel connected to other people.

        # Optimism (Scheier, Carver, and Bridges, 1994): cp___198 - cp___207 See: https://www.cmu.edu/dietrich/psychology/pdf/scales/LOTR_Scale.pdf
        'Optimism_1': file[:5] + '198', 'Optimism_2': file[:5] + '199', 'Optimism_3': file[:5] + '200',
        'Optimism_4': file[:5] + '201', 'Optimism_5': file[:5] + '202',
        'Optimism_6': file[:5] + '203', 'Optimism_7': file[:5] + '204', 'Optimism_8': file[:5] + '205',
        'Optimism_9': file[:5] + '206', 'Optimism_10': file[:5] + '207',
        }

    existing_cols = [v for v in vars.values() if v in df.columns]
    df = df.loc[:, existing_cols]

    rename_dict = {v:k for k,v in vars.items() if k != 'nomem_encr' and v in df.columns}
    df.rename(columns=rename_dict, inplace=True)

    if file[:2] == 'av':
        df['Year'] = file[6:10]
    else:
        df['Year'] = '20' + file[2:4]

    dataframes.append(df)


# Merging dataframes
clean_dataframes = []

for df in dataframes:
    if df is None or df.empty:
        continue
    df = df.loc[:, ~df.columns.duplicated()]
    clean_dataframes.append(df)

if not clean_dataframes:
    raise ValueError("No valid DataFrames to merge! Check file paths and formats.")

stacked_df = pd.concat(clean_dataframes, axis=0, ignore_index=True)

#  Convert 'nomem_encr' to integer type.
stacked_df['nomem_encr'] = stacked_df['nomem_encr'].astype(int)

# Set 'nomem_encr' and 'Year' as indices.
stacked_df.set_index(['nomem_encr', 'Year'], inplace=True)

# Sort the dataframe based on indices.
stacked_df.sort_index(level=['nomem_encr', 'Year'], inplace=True)

# Select numeric and non-numeric columns.
numeric_cols = stacked_df.select_dtypes(include=['number']).groupby(level=['nomem_encr','Year']).first()
non_numeric_cols = stacked_df.select_dtypes(exclude=['number']).groupby(level=['nomem_encr','Year']).first()


# Merge numeric and non-numeric DataFrames.
merged_stacked_df = pd.merge(numeric_cols, non_numeric_cols, left_index=True, right_index=True)
output_path = os.path.join(dir_path, 'Output files', 'Merged_dataset.csv')

# Save the merged dataframe to CSV.
merged_stacked_df.to_csv(output_path)

