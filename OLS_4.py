import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
data = pd.read_csv('data/coffee_volatility.csv')
print(f"Initial data shape: {data.shape}")
print(f"Columns in data: {list(data.columns)}")

# Load variable descriptions to categorize variables
var_desc = pd.read_csv('data/Variable_Desc.csv')
print(f"Loaded variable descriptions: {len(var_desc)} variables")

# Ensure data is sorted by date
date_column = None
if 'date' in data.columns:
    date_column = 'date'
elif 'Date' in data.columns:
    date_column = 'Date'
else:
    print("Warning: No date/Date column found in data")
    print(f"Available columns: {list(data.columns)}")

if date_column:
    print(f"Found date column: {date_column}")
    print(f"Sample date values: {data[date_column].head()}")

    # Try different date parsing approaches
    try:
        data[date_column] = pd.to_datetime(data[date_column], dayfirst=True)
        print("Successfully parsed dates with dayfirst=True")
    except:
        try:
            data[date_column] = pd.to_datetime(data[date_column], format='%d/%m/%Y')
            print("Successfully parsed dates with format='%d/%m/%Y'")
        except:
            try:
                data[date_column] = pd.to_datetime(data[date_column], infer_datetime_format=True)
                print("Successfully parsed dates with infer_datetime_format=True")
            except:
                print("Could not parse dates, proceeding without date sorting")
                date_column = None

    if date_column:
        # Check for NaT values
        nat_count = data[date_column].isna().sum()
        if nat_count > 0:
            print(f"Warning: {nat_count} NaT values found in date column")
            data = data.dropna(subset=[date_column])
            print(f"Data shape after removing NaT dates: {data.shape}")

        data = data.sort_values(date_column)
        print(f"Data sorted by {date_column}, from {data[date_column].min()} to {data[date_column].max()}")

# Check for required volatility columns
required_cols = ['parkinson_vol', 'garman_klass_vol']
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    print(f"Error: Missing required columns: {missing_cols}")
    print(f"Available columns: {list(data.columns)}")
    exit()

# Check for rolling volatility columns
rolling_cols = ['parkinson_vol_rolling_5', 'parkinson_vol_rolling_22',
                'garman_klass_vol_rolling_5', 'garman_klass_vol_rolling_22']
missing_rolling = [col for col in rolling_cols if col not in data.columns]
if missing_rolling:
    print(f"Warning: Missing rolling volatility columns: {missing_rolling}")

# Create lagged features for parkinson_vol
print("Creating lagged features...")
print(f"Data shape before creating lags: {data.shape}")

data['parkinson_vol_lag1'] = data['parkinson_vol'].shift(1)

if 'parkinson_vol_rolling_5' in data.columns:
    data['parkinson_vol_lag5'] = data['parkinson_vol_rolling_5'].shift(1)
else:
    print("Warning: parkinson_vol_rolling_5 not found, using parkinson_vol for lag5")
    data['parkinson_vol_lag5'] = data['parkinson_vol'].shift(5)

if 'parkinson_vol_rolling_22' in data.columns:
    data['parkinson_vol_lag22'] = data['parkinson_vol_rolling_22'].shift(1)
else:
    print("Warning: parkinson_vol_rolling_22 not found, using parkinson_vol for lag22")
    data['parkinson_vol_lag22'] = data['parkinson_vol'].shift(22)

# Create lagged features for garman_klass_vol
data['garman_klass_vol_lag1'] = data['garman_klass_vol'].shift(1)

if 'garman_klass_vol_rolling_5' in data.columns:
    data['garman_klass_vol_lag5'] = data['garman_klass_vol_rolling_5'].shift(1)
else:
    print("Warning: garman_klass_vol_rolling_5 not found, using garman_klass_vol for lag5")
    data['garman_klass_vol_lag5'] = data['garman_klass_vol'].shift(5)

if 'garman_klass_vol_rolling_22' in data.columns:
    data['garman_klass_vol_lag22'] = data['garman_klass_vol_rolling_22'].shift(1)
else:
    print("Warning: garman_klass_vol_rolling_22 not found, using garman_klass_vol for lag22")
    data['garman_klass_vol_lag22'] = data['garman_klass_vol'].shift(22)

print(f"Data shape after creating lag features: {data.shape}")

# Check how many NaN values we have
nan_counts = data.isnull().sum()
print(f"NaN counts per column (showing only columns with NaNs):")
for col, count in nan_counts[nan_counts > 0].items():
    print(f"  {col}: {count}")

# Drop NaN values resulting from lag creation - be more selective
initial_shape = data.shape[0]
lag_columns = ['parkinson_vol_lag1', 'parkinson_vol_lag5', 'parkinson_vol_lag22',
               'garman_klass_vol_lag1', 'garman_klass_vol_lag5', 'garman_klass_vol_lag22']

# Only drop rows where lag columns are NaN
data = data.dropna(subset=lag_columns)
final_shape = data.shape[0]

print(f"Data shape after creating lags: {data.shape}")
print(f"Rows dropped due to lag creation: {initial_shape - final_shape}")

if data.shape[0] == 0:
    print("ERROR: All data was dropped! Check your data and column names.")
    print("This usually means the required volatility columns don't exist or contain no valid data.")
    exit()

# Define base feature sets (HAR components)
parkinson_features_base = ['parkinson_vol_lag1', 'parkinson_vol_lag5', 'parkinson_vol_lag22']
garman_klass_features_base = ['garman_klass_vol_lag1', 'garman_klass_vol_lag5', 'garman_klass_vol_lag22']

# Create a dictionary to map variables to categories
var_categories = {}
for _, row in var_desc.iterrows():
    var_categories[row['Variable Name']] = row['Category']

# Add categories for the lag variables that may not be in the original descriptions
for var in parkinson_features_base:
    var_categories[var] = 'Volatility'
for var in garman_klass_features_base:
    var_categories[var] = 'Volatility'

# Group variables by category
weather_vars = [col for col in data.columns if col in var_categories and var_categories[col] == 'Weather']
uncertainty_vars = [col for col in data.columns if col in var_categories and var_categories[col] == 'Uncertainty']
equity_vars = [col for col in data.columns if col in var_categories and var_categories[col] == 'Equity']
fx_vars = [col for col in data.columns if col in var_categories and var_categories[col] == 'FX']
commodity_vars = [col for col in data.columns if col in var_categories and var_categories[col] == 'Commodities']
moments_vars = [col for col in data.columns if col in var_categories and var_categories[col] == 'Moments']

# Print category statistics
print(f"Weather variables: {len(weather_vars)}")
print(f"Uncertainty variables: {len(uncertainty_vars)}")
print(f"Equity variables: {len(equity_vars)}")
print(f"FX variables: {len(fx_vars)}")
print(f"Commodity variables: {len(commodity_vars)}")
print(f"Moments variables: {len(moments_vars)}")

# Create feature sets for each target and category
feature_sets = {}

# Base features
feature_sets['base'] = {
    'parkinson_vol': parkinson_features_base.copy(),
    'garman_klass_vol': garman_klass_features_base.copy()
}

# All available features (excluding the other volatility measure)
all_features = [col for col in data.columns
                if col not in ['date', 'Date', 'parkinson_vol', 'garman_klass_vol']]

feature_sets['all'] = {
    'parkinson_vol': parkinson_features_base.copy(),
    'garman_klass_vol': garman_klass_features_base.copy()
}

# Add all valid features to the respective 'all' sets
for col in all_features:
    if not col.startswith('garman_klass'):
        feature_sets['all']['parkinson_vol'].append(col)
    if not col.startswith('parkinson'):
        feature_sets['all']['garman_klass_vol'].append(col)

# Weather features
feature_sets['weather'] = {
    'parkinson_vol': parkinson_features_base.copy(),
    'garman_klass_vol': garman_klass_features_base.copy()
}
for col in weather_vars:
    if not col.startswith('garman_klass'):
        feature_sets['weather']['parkinson_vol'].append(col)
    if not col.startswith('parkinson'):
        feature_sets['weather']['garman_klass_vol'].append(col)

# Uncertainty features
feature_sets['uncertainty'] = {
    'parkinson_vol': parkinson_features_base.copy(),
    'garman_klass_vol': garman_klass_features_base.copy()
}
for col in uncertainty_vars:
    if not col.startswith('garman_klass'):
        feature_sets['uncertainty']['parkinson_vol'].append(col)
    if not col.startswith('parkinson'):
        feature_sets['uncertainty']['garman_klass_vol'].append(col)

# Equity features
feature_sets['equity'] = {
    'parkinson_vol': parkinson_features_base.copy(),
    'garman_klass_vol': garman_klass_features_base.copy()
}
for col in equity_vars:
    if not col.startswith('garman_klass'):
        feature_sets['equity']['parkinson_vol'].append(col)
    if not col.startswith('parkinson'):
        feature_sets['equity']['garman_klass_vol'].append(col)

# FX features
feature_sets['fx'] = {
    'parkinson_vol': parkinson_features_base.copy(),
    'garman_klass_vol': garman_klass_features_base.copy()
}
for col in fx_vars:
    if not col.startswith('garman_klass'):
        feature_sets['fx']['parkinson_vol'].append(col)
    if not col.startswith('parkinson'):
        feature_sets['fx']['garman_klass_vol'].append(col)

# Commodity features
feature_sets['commodity'] = {
    'parkinson_vol': parkinson_features_base.copy(),
    'garman_klass_vol': garman_klass_features_base.copy()
}
for col in commodity_vars:
    if not col.startswith('garman_klass'):
        feature_sets['commodity']['parkinson_vol'].append(col)
    if not col.startswith('parkinson'):
        feature_sets['commodity']['garman_klass_vol'].append(col)

# Moments features
feature_sets['moments'] = {
    'parkinson_vol': parkinson_features_base.copy(),
    'garman_klass_vol': garman_klass_features_base.copy()
}
for col in moments_vars:
    if not col.startswith('garman_klass'):
        feature_sets['moments']['parkinson_vol'].append(col)
    if not col.startswith('parkinson'):
        feature_sets['moments']['garman_klass_vol'].append(col)

# Add PCA feature set
feature_sets['pca'] = {
    'parkinson_vol': parkinson_features_base.copy(),
    'garman_klass_vol': garman_klass_features_base.copy()
}

# Remove duplicates from feature sets
for feature_set_name in feature_sets:
    for target_var in feature_sets[feature_set_name]:
        feature_sets[feature_set_name][target_var] = list(set(feature_sets[feature_set_name][target_var]))

# Verify all features exist in the dataset
for feature_set_name in feature_sets:
    for target_var in feature_sets[feature_set_name]:
        features = feature_sets[feature_set_name][target_var]
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            print(
                f"Warning: The following features in {feature_set_name} for {target_var} are missing in the dataset: {missing_features}")
            # Filter out missing features
            feature_sets[feature_set_name][target_var] = [f for f in features if f in data.columns]
            print(f"Proceeding with {len(feature_sets[feature_set_name][target_var])} available features")

# Print feature set sizes
for feature_set_name, feature_sets_dict in feature_sets.items():
    for target_var, features in feature_sets_dict.items():
        print(f"{feature_set_name} features for {target_var}: {len(features)}")

# Define horizons
horizons = [1, 5, 10, 22, 44, 66]
print(f"Forecasting horizons: {horizons}")

# Initialize results dictionary
results = {
    'parkinson_vol': {
        feature_set: {horizon: {} for horizon in horizons}
        for feature_set in feature_sets
    },
    'garman_klass_vol': {
        feature_set: {horizon: {} for horizon in horizons}
        for feature_set in feature_sets
    }
}

# Store coefficients for variable importance calculation
coefficients = {
    'parkinson_vol': {
        feature_set: {horizon: [] for horizon in horizons}
        for feature_set in feature_sets
    },
    'garman_klass_vol': {
        feature_set: {horizon: [] for horizon in horizons}
        for feature_set in feature_sets
    }
}


# Function to apply PCA to variable groups
def apply_pca_to_groups(train_data, test_data, var_categories, exclude_vars=None):
    """
    Apply PCA to each variable group and return transformed data with the first component.
    """
    if exclude_vars is None:
        exclude_vars = []

    # Get unique categories
    categories = set(var_categories.values())

    # Initialize DataFrames for transformed data
    train_transformed = pd.DataFrame(index=train_data.index)
    test_transformed = pd.DataFrame(index=test_data.index)

    # Process each category
    for category in categories:
        # Get variables in this category
        category_vars = [var for var in var_categories.keys()
                         if var in var_categories and var_categories[var] == category
                         and var in train_data.columns and var not in exclude_vars]

        # Skip if no variables in this category
        if not category_vars or len(category_vars) < 2:
            if len(category_vars) == 1:
                var = category_vars[0]
                train_transformed[f'PC1_{category}'] = train_data[var].values
                test_transformed[f'PC1_{category}'] = test_data[var].values
            continue

        try:
            # Extract data for this category
            X_train_category = train_data[category_vars].copy()
            X_test_category = test_data[category_vars].copy()

            # Standardize data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_category)
            X_test_scaled = scaler.transform(X_test_category)

            # Apply PCA and get first component
            pca = PCA(n_components=1, random_state=1)
            train_pc1 = pca.fit_transform(X_train_scaled)
            test_pc1 = pca.transform(X_test_scaled)

            # Add to transformed DataFrames
            train_transformed[f'PC1_{category}'] = train_pc1.flatten()
            test_transformed[f'PC1_{category}'] = test_pc1.flatten()

            var_ratio = pca.explained_variance_ratio_[0] if pca.explained_variance_ratio_.size > 0 else 0
            print(f"  PCA for {category}: Used {len(category_vars)} variables, explained variance: {var_ratio:.4f}")

        except Exception as e:
            print(f"  Error applying PCA to {category}: {str(e)}")
            continue

    return train_transformed, test_transformed


def calculate_variable_importance(coefs_list):
    """
    Calculate variable importance according to the formula:
    VI_i = T^(-1) * sum(VI_t,i), where
    VI_t,i = |β_i^(t)| / sum(|β_i^(t)|)
    """
    if not coefs_list:
        return None

    # Filter out const term if it exists
    variable_coefs = []
    for coefs in coefs_list:
        if 'const' in coefs:
            variable_coefs.append(coefs.drop('const'))
        else:
            variable_coefs.append(coefs)

    # Get all variable names
    all_vars = set()
    for coefs in variable_coefs:
        all_vars.update(coefs.index)

    # Initialize importance dictionary
    importance = {var: 0.0 for var in all_vars}

    # Calculate importance for each fold
    for coefs in variable_coefs:
        abs_coefs = coefs.abs()
        norm_factor = abs_coefs.sum()

        if norm_factor > 0:  # Avoid division by zero
            for var in coefs.index:
                importance[var] += abs_coefs[var] / norm_factor

    # Average across folds
    T = len(variable_coefs)
    if T > 0:
        for var in importance:
            importance[var] /= T

    # Convert to DataFrame for easier display
    importance_df = pd.DataFrame({
        'Variable': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)

    return importance_df


def calculate_category_importance(coefs_list, var_categories):
    """
    Calculate aggregated variable importance by category.
    """
    if not coefs_list:
        return None

    # Filter out const term if it exists
    variable_coefs = []
    for coefs in coefs_list:
        if 'const' in coefs:
            variable_coefs.append(coefs.drop('const'))
        else:
            variable_coefs.append(coefs)

    # Get all variable names
    all_vars = set()
    for coefs in variable_coefs:
        all_vars.update(coefs.index)

    # Initialize category importance dictionary
    category_importance = {}

    # Calculate importance for each fold and aggregate by category
    for coefs in variable_coefs:
        abs_coefs = coefs.abs()
        norm_factor = abs_coefs.sum()

        if norm_factor > 0:  # Avoid division by zero
            # Initialize fold category importance
            fold_category_importance = {}

            for var in coefs.index:
                # Get category for this variable
                if var.startswith('PC1_'):
                    # Extract category from PC1 variable name
                    category = var.split('_', 1)[1]
                else:
                    # Use regular category mapping
                    category = var_categories.get(var, "Unknown")

                # Add to category importance
                var_importance = abs_coefs[var] / norm_factor
                if category not in fold_category_importance:
                    fold_category_importance[category] = 0.0
                fold_category_importance[category] += var_importance

            # Add fold results to overall category importance
            for category, importance in fold_category_importance.items():
                if category not in category_importance:
                    category_importance[category] = []
                category_importance[category].append(importance)

    # Average across folds for each category
    category_avg_importance = {}
    for category, importance_list in category_importance.items():
        category_avg_importance[category] = np.mean(importance_list)

    # Convert to DataFrame for easier display
    importance_df = pd.DataFrame({
        'Category': list(category_avg_importance.keys()),
        'Importance': list(category_avg_importance.values())
    }).sort_values('Importance', ascending=False)

    return importance_df


def rolling_window_forecast(data, features, target_var, horizon, n_folds=5, forecast_window=66, feature_set_name=None):
    """
    Implement rolling window forecasting with proper window shifting as specified.
    Modified to handle PCA transformations for the 'pca' feature set.
    Returns fold-by-fold losses for MCS testing.
    """
    # Create the target variable
    data[f"{target_var}_target"] = data[target_var].shift(-horizon)

    # Drop rows with NaN in the target
    valid_data = data.dropna(subset=[f"{target_var}_target"])

    # Calculate total observations
    N = len(valid_data)

    # Calculate training window size (80% of total dataset)
    T = int(0.8 * N)

    # Calculate step size to ensure 5 folds with 66 observations at the end
    S = int((N - T - forecast_window) / (n_folds - 1))

    if S <= 0:
        raise ValueError(
            f"Dataset too small for {n_folds} folds with {forecast_window} forecast window. Need more data.")

    print(f"  Total observations: {N}")
    print(f"  Training window size (80%): {T}")
    print(f"  Forecast window size: {forecast_window}")
    print(f"  Step size: {S}")

    if feature_set_name == 'pca':
        print(f"  Using PCA on variable groups with base features")
    else:
        print(f"  Using {len(features)} features")

    # Check for perfect multicollinearity if not using PCA
    if feature_set_name != 'pca':
        X = valid_data[features]
        try:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            perfect_collinear = [column for column in upper.columns if any(upper[column] > 0.999)]

            if perfect_collinear:
                print(f"  Warning: Found {len(perfect_collinear)} features with near-perfect correlation")
        except Exception as e:
            print(f"  Warning: Could not check for multicollinearity: {e}")

    # Store predictions
    all_y_true = []
    all_y_pred = []

    # Store model coefficients for each fold
    all_coefs = []

    # Store fold-by-fold losses for MCS testing
    fold_losses = []

    # Perform cross-validation
    for fold in range(n_folds):
        # Calculate window positions
        train_start = fold * S
        train_end = train_start + T
        test_start = train_end
        test_end = min(test_start + forecast_window, N)

        # Ensure we don't exceed available data
        if test_end > N:
            print(f"  Warning: Fold {fold + 1} would exceed available data. Adjusting...")
            test_end = N
        if train_end > N:
            print(f"  Warning: Fold {fold + 1} training window exceeds available data. Skipping...")
            continue

        print(f"  Fold {fold + 1}: Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]")

        # Split into train and test
        train_data = valid_data.iloc[train_start:train_end]
        test_data = valid_data.iloc[test_start:test_end]

        # Skip fold if there's not enough test data
        if len(test_data) < horizon:
            print(f"  Warning: Not enough test data for horizon {horizon} in fold {fold + 1}. Skipping...")
            continue

        try:
            # Handle PCA feature set differently
            if feature_set_name == 'pca':
                # Define variables to exclude from PCA (target variables and date columns)
                exclude_vars = [col for col in valid_data.columns
                                if col == target_var or col == f"{target_var}_target"
                                or col in ['date', 'Date']]

                # Apply PCA to variable groups
                X_train_pca, X_test_pca = apply_pca_to_groups(
                    train_data, test_data, var_categories, exclude_vars
                )

                # Add base features (lag variables) to PCA-transformed data
                for feature in features:
                    if feature in train_data.columns:
                        X_train_pca[feature] = train_data[feature].values
                        X_test_pca[feature] = test_data[feature].values

                X_train = X_train_pca
                X_test = X_test_pca
            else:
                # Regular feature set
                X_train = train_data[features].copy()
                X_test = test_data[features].copy()

            # Get y for train and test
            y_train = train_data[f"{target_var}_target"].copy()
            y_test = test_data[f"{target_var}_target"].copy()

            # Add constant to X_train
            X_train_with_const = sm.add_constant(X_train)

            # Train the model
            model = sm.OLS(y_train, X_train_with_const).fit()

            # Add constant to X_test
            X_test_with_const = sm.add_constant(X_test)

            # Ensure X_test_with_const has the same columns as X_train_with_const
            for col in X_train_with_const.columns:
                if col not in X_test_with_const.columns:
                    X_test_with_const[col] = 0

            # Ensure columns are in same order as during training
            X_test_with_const = X_test_with_const[X_train_with_const.columns]

            # Make predictions
            preds = model.predict(X_test_with_const)

            # Store model coefficients
            all_coefs.append(model.params)

            # Calculate fold-specific losses
            fold_mse = mean_squared_error(y_test.values, preds)
            fold_mae = mean_absolute_error(y_test.values, preds)

            # Calculate fold-specific QLike
            fold_qlikes = []
            for true, pred in zip(y_test.values, preds):
                if true <= 0 or pred <= 0:
                    continue  # Skip invalid values
                qlike = np.log(pred / true) + true / pred - 1
                fold_qlikes.append(qlike)

            fold_qlike = np.mean(fold_qlikes) if fold_qlikes else np.nan

            # Store fold loss information
            fold_losses.append({
                'fold': fold + 1,
                'mse': fold_mse,
                'mae': fold_mae,
                'qlike': fold_qlike,
                'n_observations': len(y_test)
            })

            # Store results for overall metrics
            all_y_true.extend(y_test.values)
            all_y_pred.extend(preds)

        except Exception as e:
            print(f"  Error in fold {fold + 1}: {str(e)}")
            continue

    # Skip evaluation if no predictions were made
    if not all_y_true:
        print(f"  Warning: No predictions were made for {target_var} with horizon {horizon}")
        return np.nan, np.nan, np.nan, [], []

    # Calculate overall metrics
    mse = mean_squared_error(all_y_true, all_y_pred)
    mae = mean_absolute_error(all_y_true, all_y_pred)

    # Calculate overall QLike (Quasi-likelihood)
    qlikes = []
    for true, pred in zip(all_y_true, all_y_pred):
        if true <= 0 or pred <= 0:
            continue  # Skip invalid values
        qlike = np.log(pred / true) + true / pred - 1
        qlikes.append(qlike)

    qlike = np.mean(qlikes) if qlikes else np.nan

    return mse, mae, qlike, all_coefs, fold_losses


# Function to calculate Model Confidence Set
def calculate_mcs(loss_dict, alpha=0.1):
    """
    Simple implementation of Model Confidence Set.
    Returns the models that are not significantly worse than the best model.
    """
    # Sort models by loss
    sorted_models = sorted(loss_dict.items(), key=lambda x: x[1])

    if not sorted_models:
        return []

    # Get best model's loss
    best_loss = sorted_models[0][1]

    # Include models that are within (1+alpha) of the best loss
    # This is a simplified approach compared to the full MCS procedure
    mcs_models = [model for model, loss in sorted_models
                  if loss <= best_loss * (1 + alpha) and not np.isnan(loss)]

    return mcs_models


# Evaluate models
print("\nEvaluating models...")

# Store all fold losses for MCS testing
all_fold_losses = []

for target_var in ['parkinson_vol', 'garman_klass_vol']:
    print(f"\nEvaluating target: {target_var}")

    for feature_set_name in feature_sets:
        print(f"\n  Feature set: {feature_set_name}")
        features = feature_sets[feature_set_name][target_var]

        for horizon in horizons:
            print(f"    Evaluating horizon: {horizon} days")

            mse, mae, qlike, coefs, fold_losses = rolling_window_forecast(
                data, features, target_var, horizon, feature_set_name=feature_set_name
            )

            results[target_var][feature_set_name][horizon] = {
                'MSE': mse,
                'MAE': mae,
                'QLike': qlike
            }

            # Store coefficients for variable importance calculation
            coefficients[target_var][feature_set_name][horizon] = coefs

            # Store fold losses for MCS testing
            for fold_loss in fold_losses:
                fold_loss.update({
                    'target_variable': target_var,
                    'horizon': horizon,
                    'feature_set': feature_set_name,
                    'model_id': f'OLS_{feature_set_name}',
                    'model_type': 'OLS',
                    'volatility_type': target_var
                })
                all_fold_losses.append(fold_loss)

            print(f"    {target_var} - {feature_set_name} - Horizon {horizon}: "
                  f"MSE = {mse:.6f}, MAE = {mae:.6f}, QLike = {qlike:.6f}")

# Calculate Model Confidence Set for each target and horizon
print("\nCalculating Model Confidence Set...")
mcs_results = {
    'parkinson_vol': {horizon: [] for horizon in horizons},
    'garman_klass_vol': {horizon: [] for horizon in horizons}
}

# Also find the best model for each target and horizon
best_models = {
    'parkinson_vol': {horizon: {'feature_set': None, 'metrics': {}} for horizon in horizons},
    'garman_klass_vol': {horizon: {'feature_set': None, 'metrics': {}} for horizon in horizons}
}

# Store detailed MCS results for export
mcs_detailed_results = []

for target_var in ['parkinson_vol', 'garman_klass_vol']:
    for horizon in horizons:
        # Collect MSE values for all feature sets
        all_mses = {}

        for feature_set_name in feature_sets:
            mse = results[target_var][feature_set_name][horizon]['MSE']

            if not np.isnan(mse):
                all_mses[feature_set_name] = mse

        # Calculate MCS for this horizon
        mcs_models = calculate_mcs(all_mses)
        mcs_results[target_var][horizon] = mcs_models

        print(f"  {target_var} - Horizon {horizon} MCS: {mcs_models}")

        # Create detailed MCS results entries for CSV export
        for model in mcs_models:
            mcs_detailed_results.append({
                'target_variable': target_var,
                'horizon': horizon,
                'feature_set': model,
                'MSE': all_mses[model],
                'MAE': results[target_var][model][horizon]['MAE'],
                'QLike': results[target_var][model][horizon]['QLike'],
                'rank_within_mcs': mcs_models.index(model) + 1,
                'total_models_in_mcs': len(mcs_models)
            })

        # Find best model for this horizon
        if all_mses:
            best_feature_set = min(all_mses, key=all_mses.get)

            best_models[target_var][horizon] = {
                'feature_set': best_feature_set,
                'metrics': results[target_var][best_feature_set][horizon]
            }

            print(f"  Best model for {target_var}, horizon {horizon}: "
                  f"OLS with {best_feature_set} features, "
                  f"MSE = {best_models[target_var][horizon]['metrics']['MSE']:.6f}")

# Save MCS results to CSV
if mcs_detailed_results:
    mcs_df = pd.DataFrame(mcs_detailed_results)
    mcs_df.to_csv("mcs_results_ols.csv", index=False)
    print(f"MCS results saved to mcs_results_ols.csv")

# Calculate variable importance for the best models
print("\nCalculating Variable Importance for Best Models...")
var_importance_results = {
    'parkinson_vol': {horizon: None for horizon in horizons},
    'garman_klass_vol': {horizon: None for horizon in horizons}
}

for target_var in ['parkinson_vol', 'garman_klass_vol']:
    for horizon in horizons:
        best_info = best_models[target_var][horizon]

        if best_info['feature_set'] is not None:
            feature_set = best_info['feature_set']

            # Get coefficients for the best model
            coefs = coefficients[target_var][feature_set][horizon]

            if coefs:
                # Calculate variable importance
                importance = calculate_variable_importance(coefs)
                var_importance_results[target_var][horizon] = importance

                print(f"\n  Variable Importance for {target_var}, horizon {horizon}, "
                      f"OLS with {feature_set} features:")
                print(importance.head(10))

# NEW SECTION: Calculate category-level variable importance for uncertainty feature set
print("\nCalculating Category-Level Variable Importance for Uncertainty Feature Set...")

# Store category-level importance results specifically for uncertainty feature set
uncertainty_category_importance = []

for target_var in ['parkinson_vol', 'garman_klass_vol']:
    for horizon in horizons:
        # Get coefficients for uncertainty feature set
        uncertainty_coefs = coefficients[target_var]['uncertainty'][horizon]

        if uncertainty_coefs:
            # Calculate category-level importance
            category_importance = calculate_category_importance(uncertainty_coefs, var_categories)

            if category_importance is not None:
                print(f"\n  Category-Level Variable Importance for {target_var}, horizon {horizon}, "
                      f"OLS with uncertainty features:")
                print(category_importance)

                # Store results for CSV export
                for _, row in category_importance.iterrows():
                    uncertainty_category_importance.append({
                        'target_variable': target_var,
                        'horizon': horizon,
                        'feature_set': 'uncertainty',
                        'category': row['Category'],
                        'importance': row['Importance'],
                        'method': 'OLS'
                    })

# Save category-level variable importance for uncertainty feature set to CSV
if uncertainty_category_importance:
    uncertainty_category_df = pd.DataFrame(uncertainty_category_importance)
    uncertainty_category_df.to_csv("uncertainty_category_importance_ols.csv", index=False)
    print(
        f"\nCategory-level variable importance for uncertainty feature set saved to uncertainty_category_importance_ols.csv")

# Create final output DataFrame
final_results = []

for target_var in ['parkinson_vol', 'garman_klass_vol']:
    for horizon in horizons:
        best_info = best_models[target_var][horizon]

        if best_info['feature_set'] is not None:
            final_results.append({
                'target_variable': target_var,
                'horizon': horizon,
                'best_feature_set': best_info['feature_set'],
                'MSE': best_info['metrics']['MSE'],
                'MAE': best_info['metrics']['MAE'],
                'QLike': best_info['metrics']['QLike']
            })

# Create DataFrame of best models and metrics
best_models_df = pd.DataFrame(final_results)
best_models_df.to_csv("best_models_ols.csv", index=False)

# Create DataFrame of variable importance for each best model
importance_results = []

for target_var in ['parkinson_vol', 'garman_klass_vol']:
    for horizon in horizons:
        importance = var_importance_results[target_var][horizon]

        if importance is not None:
            best_feature_set = best_models[target_var][horizon]['feature_set']

            for _, row in importance.iterrows():
                variable = row['Variable']

                # Get the category for this variable
                if variable.startswith('PC1_'):
                    # Extract category directly from PC1 variable name
                    category = variable.split('_', 1)[1]  # Get everything after 'PC1_'
                else:
                    # Use the regular category mapping for non-PCA variables
                    category = var_categories.get(variable, "Unknown")

                importance_results.append({
                    'target_variable': target_var,
                    'horizon': horizon,
                    'feature_set': best_feature_set,
                    'variable': variable,
                    'importance': row['Importance'],
                    'category': category,  # Now properly assigns category for PC1 variables
                    'method': 'OLS'  # Add the methodological approach
                })

# Create DataFrame of variable importance
if importance_results:
    importance_df = pd.DataFrame(importance_results)
    importance_df.to_csv("var_imp_ols.csv", index=False)  # Changed filename

# Print final results in a tabular format for each target variable and horizon
print("\nFinal Results with Model Confidence Set:")

for target_var in ['parkinson_vol', 'garman_klass_vol']:
    print(f"\n{target_var}:")
    print("  Horizon\tBest Feature Set\tMSE\t\tMAE\t\tQLike")

    for horizon in horizons:
        if best_models[target_var][horizon]['feature_set'] is not None:
            feature_set = best_models[target_var][horizon]['feature_set']
            mse = best_models[target_var][horizon]['metrics']['MSE']
            mae = best_models[target_var][horizon]['metrics']['MAE']
            qlike = best_models[target_var][horizon]['metrics']['QLike']

            print(f"  {horizon}\t{feature_set}\t{mse:.6f}\t{mae:.6f}\t{qlike:.6f}")

print("\nResults saved to best_models_ols.csv, var_imp_ols.csv, and mcs_results_ols.csv")

# Create comprehensive MCS testing dataset
print("\nCreating comprehensive MCS testing dataset...")

if all_fold_losses:
    mcs_fold_df = pd.DataFrame(all_fold_losses)

    # Reorder columns for better readability
    mcs_columns = ['model_id', 'model_type', 'target_variable', 'volatility_type', 'feature_set',
                   'horizon', 'fold', 'mse', 'mae', 'qlike', 'n_observations']

    mcs_fold_df = mcs_fold_df[mcs_columns]

    # Sort by target variable, horizon, and fold for easier analysis
    mcs_fold_df = mcs_fold_df.sort_values(['target_variable', 'horizon', 'model_id', 'fold'])

    # Save to CSV
    mcs_fold_df.to_csv("ols_fold_losses_mcs.csv", index=False)

    print(f"Saved {len(mcs_fold_df)} fold-level loss records to ols_fold_losses_mcs.csv")
    print(f"Dataset contains {mcs_fold_df['model_id'].nunique()} unique models across {len(horizons)} horizons")
    print(f"Models included: {sorted(mcs_fold_df['model_id'].unique())}")

    # Print summary statistics
    print("\nSummary of fold-level losses:")
    summary_stats = mcs_fold_df.groupby(['target_variable', 'horizon', 'model_id']).agg({
        'mse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'qlike': ['mean', 'std'],
        'fold': 'count'
    }).round(6)

    print(summary_stats.head(10))
else:
    print("Warning: No fold losses were collected. MCS dataset not created.")

print("Data processing and results export complete!")