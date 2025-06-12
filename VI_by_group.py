import pandas as pd
import os

# Read the model summary files (for horizon information)
garman_models = pd.read_csv('best_models_summary_garman_klass_vol.csv')
parkinson_models = pd.read_csv('best_models_summary_parkinson_vol.csv')

# Read the variable descriptions to get categories
var_desc = pd.read_csv('data/Variable_Desc.csv')
var_categories = dict(zip(var_desc['Variable Name'], var_desc['Category']))

# Create a dictionary mapping volatility measures to their summary dataframes
volatility_models = {
    'parkinson_vol': parkinson_models,
    'garman_klass_vol': garman_models
}

# Define specific variable importance files for each volatility and horizon
specific_files = {
    'parkinson_vol': {
        1: 'var_imp_ols.csv',
        5: 'var_imp_bag.csv',
        10: 'var_imp_bag.csv',
        22: 'var_imp_en.csv',
        44: 'var_imp_bag.csv',
        66: 'var_imp_bag.csv'
    },
    'garman_klass_vol': {
        1: 'var_imp_lasso.csv',
        5: 'var_imp_ols.csv',
        10: 'var_imp_ols.csv',
        22: 'var_imp_en.csv',
        44: 'var_imp_bag.csv',
        66: 'var_imp_lasso.csv'
    }
}

# Initialize an empty DataFrame to store all results
all_results = pd.DataFrame()

# Process each volatility measure
for vol_measure, models_df in volatility_models.items():
    # Get the unique horizons for this volatility measure
    horizons = models_df['horizon'].unique()

    for horizon in horizons:
        print(f"Processing {vol_measure} for horizon {horizon}...")

        try:
            # Get the specific variable importance file to use
            if horizon in specific_files[vol_measure]:
                var_imp_filename = specific_files[vol_measure][horizon]
                print(f"Using {var_imp_filename} for {vol_measure}, horizon {horizon} as specified")
            else:
                print(f"No specific file assigned for {vol_measure}, horizon {horizon}")
                continue

            # Check if the file exists in the current directory
            if not os.path.exists(var_imp_filename):
                print(f"Variable importance file {var_imp_filename} not found for {vol_measure}, horizon {horizon}")
                continue

            # Read the CSV file with variable importance data
            df = pd.read_csv(var_imp_filename)

            # Filter for specific target variable and horizon
            filtered_df = df[(df['target_variable'] == vol_measure) & (df['horizon'] == horizon)]

            # Check if we have any data after filtering
            if filtered_df.empty:
                print(f"No data found for {vol_measure} with horizon {horizon} in {var_imp_filename}")
                continue

            # If 'category' column is not in the dataset, map features to categories
            if 'category' not in filtered_df.columns:
                # Map each feature to its category
                filtered_df['category'] = filtered_df['variable'].map(var_categories)
                # Handle features not found in var_categories
                filtered_df.loc[filtered_df['category'].isna(), 'category'] = 'Unknown'

            # Group by category and sum importance
            result_df = filtered_df.groupby('category')['importance'].sum().reset_index()

            # Multiply the summed importance values by 100 to convert to percentage
            result_df['importance'] = result_df['importance'] * 100

            # Rename the column to indicate percentage
            result_df.rename(columns={'importance': 'total importance (%)'}, inplace=True)

            # Add target variable, horizon, and file source columns
            result_df['target variable'] = vol_measure
            result_df['horizon'] = horizon
            result_df['source file'] = var_imp_filename

            # Extract model type from filename
            model_type = var_imp_filename.replace('var_imp_', '').replace('.csv', '')
            result_df['model type'] = model_type

            # Reorder columns
            result_df = result_df[
                ['target variable', 'horizon', 'model type', 'source file', 'total importance (%)', 'category']]

            # Append to the combined results
            all_results = pd.concat([all_results, result_df], ignore_index=True)

        except Exception as e:
            print(f"Error processing {vol_measure} for horizon {horizon}: {e}")

# Sort the combined results by target variable, horizon, and total importance
all_results = all_results.sort_values(
    by=['target variable', 'horizon', 'total importance (%)'],
    ascending=[True, True, False]
)

# Save all results to a CSV file
output_file = 'coffee_volatility_importance_results.csv'
all_results.to_csv(output_file, index=False)

print(f"\nAll results saved to {output_file}")

# Print a preview of the results
print("\nResults preview:")
print(all_results.head(10).to_string(index=False))