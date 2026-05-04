import pandas as pd
import shap
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Load cleaned dataset
dir_path = 'C:/thesis_dss'
absolute_path = dir_path + '/Output files/'
model_df = pd.read_csv(absolute_path + 'slope_dataset.csv')

# outcome and features
groups = model_df['nomem_encr']
outcome = 'Loneliness_Score'

numerical_cols = [
    'Age', 'Nr household members', 'Household Income',
    'Urbanity', 'Education Level', 'Health_Perception', 'Leisure_Satisfaction', 'Social_Contact_Score',
    'MHI_Score', 'Life_Satisfaction_Score', 'Extraversion',
    'Agreeableness', 'Conscientiousness', 'Emotional_Stability', 'Intellect_Imagination',
    'Self_esteem_Score', 'IOS', 'Optimism_Score', 'Loneliness_Slope', 'Loneliness_Change', 'Loneliness_std'
]

features = [
    'Age', 'Nr household members', 'Gender', 'Partner', 'Household Income',
    'Urbanity', 'Education Level', 'Occupation_Code',
    'Health_Perception', 'Leisure_Satisfaction', 'Social_Contact_Score', 'MHI_Score', 'Life_Satisfaction_Score', 'Extraversion',
    'Agreeableness', 'Conscientiousness', 'Emotional_Stability', 'Intellect_Imagination',
    'Self_esteem_Score', 'IOS', 'Optimism_Score', 'Loneliness_Slope', 'Loneliness_Change', 'Loneliness_std'
]

X = model_df[features]
y = model_df[outcome]

#%%
# -------------------------------------
# Create ID-level dataset ONLY for stratification
id_level = model_df[['nomem_encr', 'Loneliness_Score']].copy()

# Take ONE value per ID for stratification (e.g., last measurement)
id_level = id_level.sort_values('nomem_encr').groupby('nomem_encr').last().reset_index()

# Create bins for stratification
id_level['Loneliness_bin'] = pd.qcut(
    id_level['Loneliness_Score'],
    q=5,
    duplicates='drop'
)

# -------------------------------------
# create stratified train and test sets based on unique IDs
train_group, test_group = train_test_split(
    id_level,
    test_size=0.2,
    random_state=42,
    stratify=id_level['Loneliness_bin']
)

# indices to split the complete dataset
train_ids = train_group['nomem_encr']
test_ids = test_group['nomem_encr']

train_df = model_df[model_df['nomem_encr'].isin(train_ids)].reset_index(drop=True)
test_df = model_df[model_df['nomem_encr'].isin(test_ids)].reset_index(drop=True)

# save the train and test sets as text documents for HistoricalRF
train_file_path = dir_path + '/HistoricalRF/Hist_RF_train'
test_file_path = dir_path + '/HistoricalRF/Hist_RF_test'

# save the train DataFrame
train_df.to_csv(train_file_path + '.txt', sep='\t', index=False)

# save the test DataFrame
test_df.to_csv(test_file_path + '.txt', sep='\t', index=False)

# create stratified train and validation sets for hyper-parameter tuning
val_train_group, val_validation_group = train_test_split(
    train_group,
    test_size=0.25,
    random_state=42,
    stratify=train_group['Loneliness_bin']
)

val_train_ids = val_train_group['nomem_encr']
val_validation_ids = val_validation_group['nomem_encr']

v_train_df = model_df[model_df['nomem_encr'].isin(val_train_ids)].reset_index(drop=True)
v_validation_df = model_df[model_df['nomem_encr'].isin(val_validation_ids)].reset_index(drop=True)

# save the cv_train and cv_validation set as text documents for HistoricalRF
v_train_file_path = dir_path + '/HistoricalRF/Hist_RF_CV_Train'
v_validation_file_path = dir_path + '/HistoricalRF/Hist_RF_CV_Validation'

# Save the cv_train DataFrame
v_train_df.to_csv(v_train_file_path + '.txt', sep='\t', index=False)

# Save the cv_validation DataFrame
v_validation_df.to_csv(v_validation_file_path + '.txt', sep='\t', index=False)

# %%
# BASELINE MODEL
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor_non_tree = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_non_tree),
    ("model", LinearRegression())
])

param_grid = {
    "model__fit_intercept": [True, False]
}

outer_cv = GroupKFold(n_splits=5)

r2_scores = []
mse_scores = []
mae_scores = []

for train_idx, test_idx in outer_cv.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=GroupKFold(n_splits=5),
        scoring="r2"
    )

    # groups passed to inner CV
    grid_search.fit(X_train, y_train, groups=groups_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    r2_scores.append(r2_score(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))
    mae_scores.append(mean_absolute_error(y_test, y_pred))

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor_non_tree),
    ("model", LinearRegression())
])

final_model.fit(X, y)

print("R2:", np.mean(r2_scores))
print("MSE:", np.mean(mse_scores))
print("MAE:", np.mean(mae_scores))


#%%
# SHAP
X_sample = X.sample(500, random_state=42)

explainer = shap.Explainer(final_model.predict, X_sample)
shap_values = explainer(X_sample)

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=X.columns
)

#%%
# random forest nested cv
outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=5)

r2_scores, mse_scores, mae_scores = [], [], []

# Parameter grid for grid search
param_grid_rf = {
    'n_estimators': [64, 80, 96, 112, 128],
    'max_features': ['sqrt', 'log2', None]
}

for train_idx, test_idx in outer_cv.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train    = groups.iloc[train_idx]

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid_rf,
        cv=inner_cv,
        scoring='r2',
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train, groups=groups_train)

    y_pred = grid_search.best_estimator_.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))
    mse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae_scores.append(mean_absolute_error(y_test, y_pred))

print(f"R²:   {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
print(f"MSE: {np.mean(mse_scores):.3f} ± {np.std(mse_scores):.3f}")
print(f"MAE:  {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")

# Final model trained on ALL data with best params from last fold
final_model = grid_search.best_estimator_


#%%
# Refit best params on all data
final_model = RandomForestRegressor(
    **grid_search.best_params_,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X, y)

# SHAP on a sample of the full dataset
X_sample = X.sample(500, random_state=42)

explainer   = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_sample)

shap_results["RandomForest"] = shap_values

shap.summary_plot(shap_values, X_sample, feature_names=features, show=True)
shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=features, show=True)


#%%
#xgboost
performance_cv = {}
performance_test = {}

param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 6],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# group
groups_train = X_train.index

# Model
xgb = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1
)

# Grid search
grid_search_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid_xgb,
    cv=GroupKFold(n_splits=5),
    scoring='r2',
    n_jobs=-1
)

# Fit
grid_search_xgb.fit(X_train, y_train, groups=groups_train)

best_model = grid_search_xgb.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

label = "XGBoost"
performance_cv[label]   = [grid_search_xgb.best_score_]
performance_test[label] = {
    "R2":  r2_score(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred)
}

print(f"Best params:   {grid_search_xgb.best_params_}")
print(f"Validation R2: {grid_search_xgb.best_score_:.3f}")
print(f"Test R2:       {performance_test[label]['R2']:.3f}")
print(f"Test MSE:      {performance_test[label]['MSE']:.3f}")
print(f"Test MAE:      {performance_test[label]['MAE']:.3f}")
