# Install and load htree
install.packages(
  "https://cran.r-project.org/src/contrib/Archive/htree/htree_2.0.0.tar.gz",
  repos = NULL,
  type = "source"
)
library(htree)

set.seed(42)

# Paths
dir_path <- 'C:/thesis_dss'
absolute_path <- file.path(dir_path, "HistoricalRF")

# Load train/validation sets
RF_validation_train_df <- read.delim(file.path(absolute_path, "Hist_RF_CV_Train.txt"))
RF_validation_df       <- read.delim(file.path(absolute_path, "Hist_RF_CV_Validation.txt"))

# Remove extra index columns
RF_validation_train_df <- RF_validation_train_df[, !grepl("^index|^Unnamed", names(RF_validation_train_df))]
RF_validation_df       <- RF_validation_df[, !grepl("^index|^Unnamed", names(RF_validation_df))]

# Convert 'Year' to numeric
RF_validation_train_df$Year <- as.numeric(RF_validation_train_df$Year)
RF_validation_df$Year       <- as.numeric(RF_validation_df$Year)

# Predictors (all except ID and Year)
predictors_train <- RF_validation_train_df[, !(names(RF_validation_train_df) %in% c("nomem_encr", "Year", "Loneliness_Score"))]
predictors_val   <- RF_validation_df[, names(predictors_train)] 

# -----------------------------
# Hyperparameter tuning
ntrees_values <- c(128)
mtry_values   <- c(21)

# Metrics storage
rmse_scores <- matrix(NA, nrow=length(ntrees_values), ncol=length(mtry_values))
mae_scores  <- matrix(NA, nrow=length(ntrees_values), ncol=length(mtry_values))
mse_scores  <- matrix(NA, nrow=length(ntrees_values), ncol=length(mtry_values))
r2_scores   <- matrix(NA, nrow=length(ntrees_values), ncol=length(mtry_values))

best_rmse <- Inf
best_ntrees <- 0
best_mtry <- 0


for (i in seq_along(ntrees_values)) {
  for (j in seq_along(mtry_values)) {
    
    cat("ntrees:", ntrees_values[i], "mtry:", mtry_values[j], "\n")
    
    # Train HistoricalRF
    model <- hrf(
      x = predictors_train,
      time = RF_validation_train_df$Year,
      id = RF_validation_train_df$nomem_encr,
      y = "Loneliness_Score",
      ntrees = ntrees_values[i],
      mtry = mtry_values[j],
      classify = FALSE,
      se = FALSE
    )
    
    # Predict
    p <- predict_hrf(
      model,
      x = predictors_val,
      time = RF_validation_df$Year,
      id = RF_validation_df$nomem_encr
    )
    
    RF_validation_df$pred <- p[,1]
    
    eval_df <- RF_validation_df
    
    # Metrics
    mse_val  <- mean((eval_df$Loneliness_Score - eval_df$pred)^2)
    rmse_val <- sqrt(mse_val)
    mae_val  <- mean(abs(eval_df$Loneliness_Score - eval_df$pred))
    r2_val <- 1 - sum((eval_df$Loneliness_Score - eval_df$pred)^2) /
      sum((eval_df$Loneliness_Score - mean(eval_df$Loneliness_Score))^2)
    
    # Store metrics
    mse_scores[i,j]  <- mse_val
    rmse_scores[i,j] <- rmse_val
    mae_scores[i,j]  <- mae_val
    r2_scores[i,j]   <- r2_val
    
    cat("RMSE:", round(rmse_val,4),
        "MAE:", round(mae_val,4),
        "MSE:", round(mse_val,4),
        "R2:", round(r2_val,4), "\n\n")
    
    # Update best
    if (!is.na(rmse_val) && rmse_val < best_rmse) {
      best_rmse <- rmse_val
      best_ntrees <- ntrees_values[i]
      best_mtry <- mtry_values[j]
    }
  }
}

cat("Best hyperparameters -> ntrees:", best_ntrees, "mtry:", best_mtry, "with RMSE:", round(best_rmse,3), "\n")

# -----------------------------
# Evaluate on full train + validation sets
RF_train_df <- RF_validation_train_df
RF_test_df  <- RF_validation_df

# Remove ID and time columns for predictors
predictors_train_full <- RF_train_df[, !(names(RF_train_df) %in% c("nomem_encr","Year", "Loneliness_Score"))]
predictors_test_full  <- RF_test_df[, names(predictors_train_full)]



# Train the model once
final_model <- hrf(
  x = predictors_train_full,
  time = RF_train_df$Year,
  id = RF_train_df$nomem_encr,
  y = "Loneliness_Score",
  ntrees = best_ntrees,
  mtry = best_mtry,
  classify = FALSE,
  se = FALSE
)

# Predict once
p <- predict_hrf(
  final_model,
  x = predictors_test_full,
  time = RF_test_df$Year,
  id = RF_test_df$nomem_encr
)

RF_test_df <- RF_test_df[order(RF_test_df$nomem_encr, RF_test_df$Year), ]


# final predictions
final_eval <- RF_test_df


# Compute metrics
mse_val  <- mean((final_eval$Loneliness_Score - final_eval$pred)^2)
rmse_val <- sqrt(mse_val)
mae_val  <- mean(abs(final_eval$Loneliness_Score - final_eval$pred))
r2_val <- 1 - sum((final_eval$Loneliness_Score - final_eval$pred)^2) /
  sum((final_eval$Loneliness_Score - mean(final_eval$Loneliness_Score))^2)

cat("Metrics on full train+validation:\n")
cat("RMSE:", round(rmse_val,3),
    "MAE:", round(mae_val,3),
    "MSE:", round(mse_val,3),
    "R²:", round(r2_val,3), "\n")

# Variable importance
vi <- varimp_hrf(final_model)
vi

