import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import random

# Model configuration
MODEL_NAME = 'ElasticNet'

# Set random seeds for reproducibility
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)

# Filter out the specific convergence warnings to keep output clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Objective did not converge")

# Load the data with comprehensive diagnostics
print("Loading data...")
try:
    data = pd.read_csv('data/coffee_volatility.csv')
    print(f"Successfully loaded data with shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Load variable descriptions to categorize variables
try:
    var_desc = pd.read_csv('data/Variable_Desc.csv')
    print(f"Loaded variable descriptions: {len(var_desc)} variables")
except Exception as e:
    print(f"Warning: Could not load variable descriptions: {e}")
    var_desc = pd.DataFrame()

# Robust date parsing
date_column = None
if 'date' in data.columns:
    date_column = 'date'
elif 'Date' in data.columns:
    date_column = 'Date'

if date_column:
    print(f"Processing date column: {date_column}")
    original_count = len(data)
    try:
        data[date_column] = pd.to_datetime(data[date_column], dayfirst=True)
        print("Successfully parsed dates using dayfirst=True")
    except:
        try:
            data[date_column] = pd.to_datetime(data[date_column], format='%d/%m/%Y')
            print("Successfully parsed dates using DD/MM/YYYY format")
        except:
            try:
                data[date_column] = pd.to_datetime(data[date_column], infer_datetime_format=True)
                print("Successfully parsed dates using inferred format")
            except:
                print("Could not parse dates, proceeding without date sorting")
                date_column = None

    if date_column:
        data = data.sort_values(date_column)
        print(f"Data sorted by {date_column}, from {data[date_column].min()} to {data[date_column].max()}")
        print(f"Date parsing complete. Retained {len(data)} of {original_count} rows")

# Check for required volatility columns before processing
required_volatility_cols = ['parkinson_vol', 'garman_klass_vol']
missing_vol_cols = [col for col in required_volatility_cols if col not in data.columns]

if missing_vol_cols:
    print(f"ERROR: Missing required volatility columns: {missing_vol_cols}")
    print(f"Available columns: {list(data.columns)}")
    raise ValueError("Cannot proceed without required volatility columns")

print(f"Found required volatility columns: {[col for col in required_volatility_cols if col in data.columns]}")

# Create lagged features with diagnostics
print("Creating lagged features...")
print(f"Data shape before lag creation: {data.shape}")

# Check for rolling volatility columns
rolling_cols_parkinson = ['parkinson_vol_rolling_5', 'parkinson_vol_rolling_22']
rolling_cols_garman = ['garman_klass_vol_rolling_5', 'garman_klass_vol_rolling_22']

missing_rolling_parkinson = [col for col in rolling_cols_parkinson if col not in data.columns]
missing_rolling_garman = [col for col in rolling_cols_garman if col not in data.columns]

if missing_rolling_parkinson:
    print(f"Warning: Missing Parkinson rolling columns: {missing_rolling_parkinson}")
    print("Creating fallback rolling averages...")
    if 'parkinson_vol_rolling_5' not in data.columns:
        data['parkinson_vol_rolling_5'] = data['parkinson_vol'].rolling(window=5, min_periods=1).mean()
    if 'parkinson_vol_rolling_22' not in data.columns:
        data['parkinson_vol_rolling_22'] = data['parkinson_vol'].rolling(window=22, min_periods=1).mean()

if missing_rolling_garman:
    print(f"Warning: Missing Garman-Klass rolling columns: {missing_rolling_garman}")
    print("Creating fallback rolling averages...")
    if 'garman_klass_vol_rolling_5' not in data.columns:
        data['garman_klass_vol_rolling_5'] = data['garman_klass_vol'].rolling(window=5, min_periods=1).mean()
    if 'garman_klass_vol_rolling_22' not in data.columns:
        data['garman_klass_vol_rolling_22'] = data['garman_klass_vol'].rolling(window=22, min_periods=1).mean()

# Create lagged features for parkinson_vol
data['parkinson_vol_lag1'] = data['parkinson_vol'].shift(1)
data['parkinson_vol_lag5'] = data['parkinson_vol_rolling_5'].shift(1)
data['parkinson_vol_lag22'] = data['parkinson_vol_rolling_22'].shift(1)

# Create lagged features for garman_klass_vol
data['garman_klass_vol_lag1'] = data['garman_klass_vol'].shift(1)
data['garman_klass_vol_lag5'] = data['garman_klass_vol_rolling_5'].shift(1)
data['garman_klass_vol_lag22'] = data['garman_klass_vol_rolling_22'].shift(1)

print(f"Data shape after lag creation: {data.shape}")

# Selective data cleaning with diagnostics
essential_columns = ['parkinson_vol', 'garman_klass_vol',
                     'parkinson_vol_lag1', 'parkinson_vol_lag5', 'parkinson_vol_lag22',
                     'garman_klass_vol_lag1', 'garman_klass_vol_lag5', 'garman_klass_vol_lag22']

print("Performing selective data cleaning...")
print(f"Data shape before cleaning: {data.shape}")

# Check for essential columns
missing_essential = [col for col in essential_columns if col not in data.columns]
if missing_essential:
    print(f"Warning: Missing essential columns: {missing_essential}")

available_essential = [col for col in essential_columns if col in data.columns]
print(f"Available essential columns: {available_essential}")

# Drop rows only where essential columns have NaN
data_cleaned = data.dropna(subset=available_essential)
print(f"Data shape after selective cleaning: {data_cleaned.shape}")

if len(data_cleaned) == 0:
    print("ERROR: No data remaining after cleaning!")
    print("Checking for data issues...")
    for col in available_essential:
        nan_count = data[col].isna().sum()
        print(f"  {col}: {nan_count} NaN values out of {len(data)}")
    raise ValueError("All data removed during cleaning - check data quality")

# Use cleaned data
data = data_cleaned
print(f"Final data shape after lag creation and cleaning: {data.shape}")

# Define base feature sets (HAR components)
parkinson_features_base = ['parkinson_vol_lag1', 'parkinson_vol_lag5', 'parkinson_vol_lag22']
garman_klass_features_base = ['garman_klass_vol_lag1', 'garman_klass_vol_lag5', 'garman_klass_vol_lag22']

# Create a dictionary to map variables to categories
var_categories = {}
if not var_desc.empty:
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

# Initialize fold losses collection for MCS
all_fold_losses = []


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

            # Apply PCA and get first component, with fixed random state
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


def calculate_variable_importance(coefs_list, feature_names=None):
    """
    Calculate variable importance according to the formula:
    VI_i = T^(-1) * sum(VI_t,i), where
    VI_t,i = |β_i^(t)| / sum(|β_i^(t)|)

    Modified to handle both ElasticNet coefficient arrays and named Series
    """
    if not coefs_list:
        return None

    # Handle different formats of coefficient lists
    if isinstance(coefs_list[0], tuple):
        # ElasticNet format: coefs are tuples of (coef_array, intercept)
        # Initialize importance dictionary using feature_names
        all_vars = feature_names
        importance = {var: 0.0 for var in all_vars}

        # Calculate importance for each fold
        for coefs, intercept in coefs_list:
            # Skip intercept for variable importance calculation
            abs_coefs = np.abs(coefs)
            norm_factor = np.sum(abs_coefs)

            if norm_factor > 0:  # Avoid division by zero
                for i, var in enumerate(feature_names):
                    importance[var] += abs_coefs[i] / norm_factor
    else:
        # OLS format: coefs are Series with named indices
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
    T = len(coefs_list)
    if T > 0:
        for var in importance:
            importance[var] /= T

    # Convert to DataFrame for easier display
    importance_df = pd.DataFrame({
        'Variable': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)

    return importance_df


def rolling_window_forecast(data, features, target_var, horizon, n_folds=5, forecast_window=66, feature_set_name=None):
    """
    Implement rolling window forecasting with Elastic Net regression and proper window shifting as specified.
    Modified to handle PCA transformations for the 'pca' feature set and return fold losses.
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
                # We'll keep these features as Elastic Net can handle multicollinearity
        except Exception as e:
            print(f"  Warning: Could not check for multicollinearity: {e}")

    # Store predictions
    all_y_true = []
    all_y_pred = []

    # Store feature names for PCA-transformed data
    pca_feature_names = None

    # Store model coefficients for each fold
    all_coefs = []

    # Store fold losses for MCS
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

                # Store feature names for variable importance calculation
                if pca_feature_names is None:
                    pca_feature_names = X_train.columns.tolist()
            else:
                # Regular feature set
                X_train = train_data[features].copy()
                X_test = test_data[features].copy()

            # Get y for train and test
            y_train = train_data[f"{target_var}_target"].copy()
            y_test = test_data[f"{target_var}_target"].copy()

            # Scale features for better Elastic Net performance
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Use ElasticNetCV to find optimal alpha and l1_ratio through cross-validation
            elastic_net_cv = ElasticNetCV(
                l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                alphas=np.logspace(-6, 2, 20),
                cv=5,
                random_state=random_seed,
                max_iter=100000,
                tol=1e-4,
                fit_intercept=True
            )
            elastic_net_cv.fit(X_train_scaled, y_train)

            # Get the optimal parameters
            optimal_alpha = elastic_net_cv.alpha_
            optimal_l1_ratio = elastic_net_cv.l1_ratio_
            print(
                f"  Optimal parameters for fold {fold + 1}: alpha={optimal_alpha:.6f}, l1_ratio={optimal_l1_ratio:.6f}")

            # Train Elastic Net with the optimal parameters
            model = ElasticNet(
                alpha=optimal_alpha,
                l1_ratio=optimal_l1_ratio,
                max_iter=100000,
                tol=1e-4,
                random_state=random_seed,
                fit_intercept=True
            )
            model.fit(X_train_scaled, y_train)

            # Make predictions
            preds = model.predict(X_test_scaled)

            # Store model coefficients
            all_coefs.append((model.coef_, model.intercept_))

            # Calculate fold-level metrics
            fold_mse = mean_squared_error(y_test.values, preds)
            fold_mae = mean_absolute_error(y_test.values, preds)

            # Calculate fold-level QLike
            fold_qlikes = []
            for true, pred in zip(y_test.values, preds):
                if true <= 0 or pred <= 0:
                    continue  # Skip invalid values
                qlike = np.log(pred / true) + true / pred - 1
                fold_qlikes.append(qlike)

            fold_qlike = np.mean(fold_qlikes) if fold_qlikes else np.nan

            # Store fold loss information
            fold_loss = {
                'model_id': f'{MODEL_NAME}_{feature_set_name}',
                'model_type': MODEL_NAME,
                'target_variable': target_var,
                'volatility_type': target_var,
                'feature_set': feature_set_name,
                'horizon': horizon,
                'fold': fold + 1,
                'mse': fold_mse,
                'mae': fold_mae,
                'qlike': fold_qlike,
                'n_observations': len(y_test)
            }
            fold_losses.append(fold_loss)

            # Store results
            all_y_true.extend(y_test.values)
            all_y_pred.extend(preds)

        except Exception as e:
            print(f"  Error in fold {fold + 1}: {str(e)}")
            # Print additional debugging information
            if 'model' in locals():
                print(
                    f"  Model has intercept: {model.intercept_:.6f} and {sum(model.coef_ != 0)} non-zero coefficients")
            if 'X_train_scaled' in locals() and 'X_test_scaled' in locals():
                print(f"  X_train shape: {X_train_scaled.shape}")
                print(f"  X_test shape: {X_test_scaled.shape}")
            continue

    # Skip evaluation if no predictions were made
    if not all_y_true:
        print(f"  Warning: No predictions were made for {target_var} with horizon {horizon}")
        return np.nan, np.nan, np.nan, [], fold_losses

    # Calculate metrics
    mse = mean_squared_error(all_y_true, all_y_pred)
    mae = mean_absolute_error(all_y_true, all_y_pred)

    # Calculate QLike (Quasi-likelihood)
    qlikes = []
    for true, pred in zip(all_y_true, all_y_pred):
        if true <= 0 or pred <= 0:
            continue  # Skip invalid values
        qlike = np.log(pred / true) + true / pred - 1
        qlikes.append(qlike)

    qlike = np.mean(qlikes) if qlikes else np.nan

    # Return feature names along with coefficients for PCA models
    if feature_set_name == 'pca' and pca_feature_names is not None:
        return mse, mae, qlike, (all_coefs, pca_feature_names), fold_losses
    else:
        return mse, mae, qlike, (all_coefs, features), fold_losses


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
for target_var in ['parkinson_vol', 'garman_klass_vol']:
    print(f"\nEvaluating target: {target_var}")

    for feature_set_name in feature_sets:
        print(f"\n  Feature set: {feature_set_name}")
        features = feature_sets[feature_set_name][target_var]

        for horizon in horizons:
            print(f"    Evaluating horizon: {horizon} days")

            mse, mae, qlike, coefs_data, fold_losses = rolling_window_forecast(
                data, features, target_var, horizon, feature_set_name=feature_set_name
            )

            results[target_var][feature_set_name][horizon] = {
                'MSE': mse,
                'MAE': mae,
                'QLike': qlike
            }

            # Store coefficients for variable importance calculation
            coefficients[target_var][feature_set_name][horizon] = coefs_data

            # Collect fold losses for MCS
            all_fold_losses.extend(fold_losses)

            print(f"    {target_var} - {feature_set_name} - Horizon {horizon}: "
                  f"MSE = {mse:.6f}, MAE = {mae:.6f}, QLike = {qlike:.6f}")

# Export fold losses for MCS testing
if all_fold_losses:
    fold_losses_df = pd.DataFrame(all_fold_losses)
    # Sort by target_variable, horizon, model_id, fold as specified
    fold_losses_df = fold_losses_df.sort_values(['target_variable', 'horizon', 'model_id', 'fold'])
    fold_losses_df.to_csv("elasticnet_fold_losses_mcs.csv", index=False)
    print(f"\nFold losses for MCS testing saved to elasticnet_fold_losses_mcs.csv")
    print(f"Total fold records: {len(fold_losses_df)}")

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
                  f"ElasticNet with {best_feature_set} features, "
                  f"MSE = {best_models[target_var][horizon]['metrics']['MSE']:.6f}")

# Save MCS results to CSV
if mcs_detailed_results:
    mcs_df = pd.DataFrame(mcs_detailed_results)
    mcs_df.to_csv("mcs_results_en.csv", index=False)
    print(f"MCS results saved to mcs_results_en.csv")

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
            coefs_data = coefficients[target_var][feature_set][horizon]

            if coefs_data:
                # Unpack coefficients and feature names
                if isinstance(coefs_data, tuple) and len(coefs_data) == 2:
                    coefs, feature_names = coefs_data
                else:
                    coefs = coefs_data
                    feature_names = feature_sets[feature_set][target_var]

                # Calculate variable importance
                importance = calculate_variable_importance(coefs, feature_names)
                var_importance_results[target_var][horizon] = importance

                print(f"\n  Variable Importance for {target_var}, horizon {horizon}, "
                      f"ElasticNet with {feature_set} features:")
                print(importance.head(10))

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
best_models_df.to_csv("best_models_en.csv", index=False)  # Changed to en

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
                    'method': 'ElasticNet'  # Add the methodological approach
                })

# Create DataFrame of variable importance
if importance_results:
    importance_df = pd.DataFrame(importance_results)
    importance_df.to_csv("var_imp_en.csv", index=False)  # Changed to en

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

# Original CSV export for all results
print("\nGenerating CSV export for all models...")
results_data = []

for target_var in ['parkinson_vol', 'garman_klass_vol']:
    for feature_set_name in feature_sets:
        for horizon in horizons:
            # Get all metrics for this combination
            mse = results[target_var][feature_set_name][horizon]['MSE']
            mae = results[target_var][feature_set_name][horizon]['MAE']
            qlike = results[target_var][feature_set_name][horizon]['QLike']

            # Add a single row with all metrics
            results_data.append({
                'target_variable': target_var,
                'feature_set': feature_set_name,
                'horizon': horizon,
                'MSE': mse,
                'MAE': mae,
                'QLike': qlike
            })

# Create DataFrame and save to CSV
results_df = pd.DataFrame(results_data)
results_df.to_csv("har_en_all.csv", index=False)
print(f"Results saved to har_en_all.csv")

print("\nResults saved to:")
print("- best_models_en.csv (best models summary)")
print("- var_imp_en.csv (variable importance)")
print("- mcs_results_en.csv (Model Confidence Set results)")
print("- elasticnet_fold_losses_mcs.csv (fold-level losses for MCS testing)")
print("- har_en_all.csv (all model results)")
print("Data processing and results export complete!")