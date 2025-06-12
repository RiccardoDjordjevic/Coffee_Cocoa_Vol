import polars as pl
import numpy as np
from numba import njit
import math

# Define Numba-accelerated volatility functions
@njit
def parkinson_volatility_numba(high, low):
    """Calculate Parkinson volatility with Numba acceleration"""
    if np.isnan(high) or np.isnan(low) or high <= 0 or low <= 0 or high < low:
        return np.nan
    return math.sqrt((1 / (4 * math.log(2))) * (math.log(high / low))**2)

@njit
def garman_klass_volatility_numba(open_price, high, low, close):
    """Calculate Garman-Klass volatility with Numba acceleration"""
    if (np.isnan(open_price) or np.isnan(high) or np.isnan(low) or np.isnan(close) or 
        open_price <= 0 or high <= 0 or low <= 0 or close <= 0 or high < low):
        return np.nan
    
    # Calculate components of the formula
    hl_component = 0.5 * (math.log(high / low))**2
    oc_component = (2 * math.log(2) - 1) * (math.log(close / open_price))**2
    
    # Check if the expression under the square root is non-negative
    if hl_component < oc_component:
        return np.nan
    
    return math.sqrt(hl_component - oc_component)

# Vectorize the Numba functions using NumPy
def apply_parkinson(df):
    high_vals = df["High_Co"].to_numpy()
    low_vals = df["Low_Co"].to_numpy()
    result = np.empty(len(df), dtype=np.float64)
    
    for i in range(len(df)):
        result[i] = parkinson_volatility_numba(high_vals[i], low_vals[i])
    
    return result

def apply_garman_klass(df):
    open_vals = df["Open_Co"].to_numpy()
    high_vals = df["High_Co"].to_numpy()
    low_vals = df["Low_Co"].to_numpy()
    close_vals = df["Last_Co"].to_numpy()
    result = np.empty(len(df), dtype=np.float64)
    
    for i in range(len(df)):
        result[i] = garman_klass_volatility_numba(open_vals[i], high_vals[i], low_vals[i], close_vals[i])
    
    return result

# Calculate realized volatility moments
def calculate_realized_moments(df):
    """Calculate realized moments using Polars expressions"""
    # Add realized variance, RVB, and RVG columns
    df = df.with_columns([
        pl.col("returns").pow(2).alias("RV"),
        pl.when(pl.col("returns") < 0).then(pl.col("returns").pow(2)).otherwise(0).alias("RVB"),
        pl.when(pl.col("returns") > 0).then(pl.col("returns").pow(2)).otherwise(0).alias("RVG")
    ])
    
    # Calculate rolling window metrics for skewness and kurtosis
    window_sizes = [5, 22]
    expressions = []
    
    for window in window_sizes:
        # Skewness formula: (√M * sum(r^3)) / (sum(r^2))^(3/2)
        expressions.append(
            (pl.lit(np.sqrt(window)) * pl.col("returns").pow(3).rolling_sum(window_size=window) / 
            pl.col("returns").pow(2).rolling_sum(window_size=window).pow(1.5)).alias(f"RSK_{window}d")
        )
        
        # Kurtosis formula: (M * sum(r^4)) / (sum(r^2))^2
        expressions.append(
            (pl.lit(window) * pl.col("returns").pow(4).rolling_sum(window_size=window) / 
            pl.col("returns").pow(2).rolling_sum(window_size=window).pow(2)).alias(f"RKU_{window}d")
        )
        
        # Add rolling means for RVB and RVG
        expressions.append(pl.col("RVB").rolling_mean(window_size=window).alias(f"RVB_{window}d"))
        expressions.append(pl.col("RVG").rolling_mean(window_size=window).alias(f"RVG_{window}d"))
    
    return df.with_columns(expressions)

# Import the CSV file using Polars
clean_df = pl.read_csv('data/clean_merged_data.csv')

# Calculate volatilities
parkinson_values = apply_parkinson(clean_df)
garman_klass_values = apply_garman_klass(clean_df)

# Calculate returns for realized moments
clean_df = clean_df.with_columns([
    pl.col("Last_Co").log().diff().alias("returns"),
    pl.Series("parkinson_vol", parkinson_values),
    pl.Series("garman_klass_vol", garman_klass_values)
])

# Calculate realized moments
clean_df = calculate_realized_moments(clean_df)

# Calculate rolling averages with Polars
clean_df = clean_df.with_columns([
    pl.col("parkinson_vol").rolling_mean(window_size=5).alias("parkinson_vol_rolling_5"),
    pl.col("parkinson_vol").rolling_mean(window_size=22).alias("parkinson_vol_rolling_22"),
    pl.col("garman_klass_vol").rolling_mean(window_size=5).alias("garman_klass_vol_rolling_5"),
    pl.col("garman_klass_vol").rolling_mean(window_size=22).alias("garman_klass_vol_rolling_22")
])

# Create ONI_dummy variable based on ONI values
clean_df = clean_df.with_columns([
    pl.when(pl.col("ONI") < -0.5).then(1)
    .when((pl.col("ONI") >= -0.5) & (pl.col("ONI") <= 0.5)).then(2)
    .otherwise(3)
    .alias("ONI_dummy")
])

# List of columns for which to calculate first differences
diff_columns = [
    "Temp_Cote", "Prec_Cote", "Temp_Ghana", "Prec_Ghana", 
    "Temp_Indo_Lampung", "Prec_Indo_Lampung", "Temp_Indo_Palu", "Prec_Indo_Palu",
    "ONI", "ADS_Index", "VIX", "GEPU_ppp"
]

# Calculate first differences for all specified columns
diff_expressions = []
for col in diff_columns:
    diff_expressions.append(pl.col(col).diff().alias(f"{col}_first"))

# Add all the first difference columns to the DataFrame
clean_df = clean_df.with_columns(diff_expressions)

# Calculate squared returns for the specified columns
squared_return_columns = [
    "GSCI_Commodity", "Soybean", "Corn", "Wheat", "Diesel", 
    "XOFUSD", "IDRUSD", "GHSUSD", "EURUSD", "GBPUSD", 
    "Netherlands", "Germany", "Malaysia"
]

# Create a list to hold all the squared return expressions
squared_return_expressions = []

for col in squared_return_columns:
    # Calculate (100 * (log(P(t)) - log(P(t-1))))^2
    squared_return_expressions.append(
        (100 * pl.col(col).log().diff()).pow(2).alias(f"{col}_squared_return")
    )

# Add all the squared return columns to the DataFrame
clean_df = clean_df.with_columns(squared_return_expressions)

# 1. Create an equal-weighted index of the 5 chocolate company stock prices
chocolate_columns = ["Hershey", "Lindt", "Fuji", "Guan_Chong", "Barry_Calle"]

# Calculate the equal-weighted index (simple average of the 5 stocks)
clean_df = clean_df.with_columns([
    pl.mean_horizontal(chocolate_columns).alias("chocolate_index")
])

# 2. Calculate squared returns for the chocolate index using the same formula
# Formula: (100 * (log(P(t)) - log(P(t-1))))^2
clean_df = clean_df.with_columns([
    (100 * pl.col("chocolate_index").log().diff()).pow(2).alias("chocolate_index_squared_return")
])

# Display the first few rows with the new columns
print(clean_df.head())

# Count NaN values and calculate percentages
print("\n===== NaN COUNT AND PERCENTAGE =====")
total_rows = len(clean_df)
nan_stats = []

for col in clean_df.columns:
    # Count null values 
    null_count = clean_df.select(pl.col(col).is_null().sum()).item()
    null_percentage = (null_count / total_rows) * 100
    
    nan_stats.append({
        "Column": col,
        "NaN Count": null_count,
        "NaN Percentage": f"{null_percentage:.2f}%"
    })

# Convert to DataFrame for better display
nan_df = pl.DataFrame(nan_stats)
print(nan_df)

# List of columns for which to calculate rolling means
rolling_mean_columns = [
    "ONI_dummy", "Temp_Cote_first", "Prec_Cote_first", "Temp_Ghana_first", 
    "Prec_Ghana_first", "Temp_Indo_Lampung_first", "Prec_Indo_Lampung_first", 
    "Temp_Indo_Palu_first", "Prec_Indo_Palu_first", "ONI_first", 
    "ADS_Index_first", "VIX_first", "GEPU_ppp_first", 
    "GSCI_Commodity_squared_return", "Soybean_squared_return", "Corn_squared_return", 
    "Wheat_squared_return", "Diesel_squared_return", "XOFUSD_squared_return", 
    "IDRUSD_squared_return", "GHSUSD_squared_return", "EURUSD_squared_return", 
    "GBPUSD_squared_return", "Netherlands_squared_return", "Germany_squared_return",
    "Malaysia_squared_return", "chocolate_index", "chocolate_index_squared_return",
    "RV", "RVB", "RVG"  # Add realized moments to rolling means
]

# Generate rolling mean expressions for each column
rolling_mean_expressions = []

for col in rolling_mean_columns:
    # For ONI_dummy, round the rolling means to the nearest integer
    if col == "ONI_dummy":
        rolling_mean_expressions.append(
            pl.col(col).rolling_mean(window_size=5).round().alias(f"{col}_rolling_5")
        )
        rolling_mean_expressions.append(
            pl.col(col).rolling_mean(window_size=22).round().alias(f"{col}_rolling_22")
        )
    # For all other columns, don't round
    else:
        rolling_mean_expressions.append(
            pl.col(col).rolling_mean(window_size=5).alias(f"{col}_rolling_5")
        )
        rolling_mean_expressions.append(
            pl.col(col).rolling_mean(window_size=22).alias(f"{col}_rolling_22")
        )

# Add all the rolling mean columns to the DataFrame
clean_df = clean_df.with_columns(rolling_mean_expressions)

columns_to_drop = [
    "GSCI_Commodity", "Soybean", "Corn", "Wheat", "Diesel", 
    "Open_Co", "High_Co", "Low_Co", "Last_Co", 
    "XOFUSD", "IDRUSD", "GHSUSD", "EURUSD", "GBPUSD", 
    "Netherlands", "Germany", "Malaysia", 
    "Hershey", "Lindt", "Fuji", "Guan_Chong", "Barry_Calle", 
    "Temp_Cote", "Prec_Cote", "Temp_Ghana", "Prec_Ghana", 
    "Temp_Indo_Lampung", "Prec_Indo_Lampung", "Temp_Indo_Palu", "Prec_Indo_Palu", 
    "ONI", "ADS_Index", "VIX", "GEPU_ppp", "returns", "RV"  # Including returns in columns to drop
]

# Drop the specified columns
clean_df = clean_df.drop(columns_to_drop)

# Display the first few rows with the new columns
print(clean_df.head())

# Save to CSV
clean_df.write_csv('data/clean_volatility.csv')

print("===== COLUMN DATA TYPES =====")
for col_name, dtype in zip(clean_df.columns, clean_df.dtypes):
    print(f"{col_name}: {dtype}")