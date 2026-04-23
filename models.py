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
    'Self_esteem_Score', 'IOS', 'Optimism_Score', 'Loneliness_Change'
]

features = [
    'Age', 'Nr household members', 'Gender', 'Partner', 'Household Income',
    'Urbanity', 'Education Level', 'Occupation_Code',
    'Health_Perception', 'Leisure_Satisfaction', 'Social_Contact_Score', 'MHI_Score', 'Life_Satisfaction_Score', 'Extraversion',
    'Agreeableness', 'Conscientiousness', 'Emotional_Stability', 'Intellect_Imagination',
    'Self_esteem_Score', 'IOS', 'Optimism_Score', 'Loneliness_Change'
]

X = model_df[features]
y = model_df[outcome]

# -------------------------------------
# Aggregate the target column to 1 row per ID/nomem_encr
aggregated_data = model_df.groupby(['nomem_encr']).agg({
    'Loneliness_Score': 'last'
}).reset_index()

# Create bins for stratification
aggregated_data['Loneliness_bin'] = pd.qcut(
    aggregated_data['Loneliness_Score'],
    q=5,
    duplicates='drop'
)

# -------------------------------------
# create stratified train and test sets based on unique IDs
train_group, test_group = train_test_split(
    aggregated_data,
    test_size=0.2,
    random_state=42,
    stratify=aggregated_data['Loneliness_bin']
)

# indices to split the complete dataset
train_ids = list(train_group['nomem_encr'])
test_ids = list(test_group['nomem_encr'])

train_df = model_df[model_df['nomem_encr'].isin(train_ids)].reset_index()
test_df = model_df[model_df['nomem_encr'].isin(test_ids)].reset_index()

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

val_train_ids = list(val_train_group['nomem_encr'])
val_validation_ids = list(val_validation_group['nomem_encr'])

v_train_df = model_df[model_df['nomem_encr'].isin(val_train_ids)].reset_index()
v_validation_df = model_df[model_df['nomem_encr'].isin(val_validation_ids)].reset_index()

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

outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=5)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_non_tree),
    ("model", LinearRegression())
])

param_grid = {
    "model__fit_intercept": [True, False]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=inner_cv,
    scoring="r2"
)

cv_results = cross_validate(
    estimator=grid_search,
    X=X,
    y=y,
    groups=groups,
    cv=outer_cv,
    scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
    return_estimator=True
)

print("R2:", np.mean(cv_results["test_r2"]))
print("MSE:", -np.mean(cv_results["test_neg_mean_squared_error"]))
print("MAE:", -np.mean(cv_results["test_neg_mean_absolute_error"]))


#%%
# SHAP
grid = cv_results["estimator"][0]
best_pipeline = grid.best_estimator_
preprocessor = best_pipeline.named_steps["preprocessor"]
model = best_pipeline.named_steps["model"]
X_transformed = preprocessor.transform(X)
explainer = shap.Explainer(model, X_transformed)
shap_values = explainer(X_transformed)

# Global feature importance
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=features
)

#%%
# random forest

# Initialize dictionaries to store performances
performance_cv = {}
performance_test = {}

# Parameter grid for grid search
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2', None]
}

# Iterate over different measurements
for measurement in ['mean']:
    # Aggregate data
    agg_cols = {col: measurement for col in model_df.columns}
    agg_cols[outcome] = 'last'
    agg_data = model_df.groupby('nomem_encr').agg(agg_cols)

    # create train and validation set
    X_train = agg_data.loc[train_ids].drop(columns=outcome)
    y_train = agg_data.loc[train_ids][outcome]
    X_test = agg_data.loc[test_ids].drop(columns=outcome)
    y_test = agg_data.loc[test_ids][outcome]

    groups = X_train.index
    # perform grid search
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid_rf,
        scoring='r2',
        cv=GroupKFold(n_splits=5),
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train, groups=groups)

    best_model = grid_search.best_estimator_
    label = f"RandomForest using {measurement}"
    performance_cv[label] = [grid_search.best_score_]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    performance_test[label] = {
        "R2": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred)
    }

    print(f"\n{label}")
    print(f"Best params:   {grid_search.best_params_}")
    print(f"Validation R2: {grid_search.best_score_:.3f}")
    print(f"Test R2:       {performance_test[label]['R2']:.3f}")
    print(f"Test MSE:      {performance_test[label]['MSE']:.3f}")
    print(f"Test MAE:      {performance_test[label]['MAE']:.3f}")

#%%
#shap
shap_results = {}

print(f"\nGenerating SHAP for: {measurement}")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap_results[measurement] = shap_values
shap.summary_plot(shap_values, X_test, show=True)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)


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
