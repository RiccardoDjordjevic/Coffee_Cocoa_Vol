import pandas as pd
import numpy as np


def calculate_descriptive_stats(file_path):
    """
    Calculate comprehensive descriptive statistics for all variables in a dataset.
    Variables will be displayed as rows and statistics as columns.
    """

    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("-" * 50)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Separate numeric and categorical variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Calculate statistics for numeric variables
    numeric_stats = pd.DataFrame()

    if numeric_cols:
        # Basic descriptive statistics
        desc_stats = df[numeric_cols].describe().T

        # Additional statistics
        additional_stats = pd.DataFrame(index=numeric_cols)
        additional_stats['missing'] = df[numeric_cols].isnull().sum()
        additional_stats['missing_pct'] = (df[numeric_cols].isnull().sum() / len(df)) * 100
        additional_stats['variance'] = df[numeric_cols].var()
        additional_stats['skewness'] = df[numeric_cols].skew()
        additional_stats['kurtosis'] = df[numeric_cols].kurtosis()
        additional_stats['range'] = df[numeric_cols].max() - df[numeric_cols].min()
        additional_stats['iqr'] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
        additional_stats['cv'] = (df[numeric_cols].std() / df[numeric_cols].mean()) * 100  # Coefficient of variation

        # Combine all numeric statistics
        numeric_stats = pd.concat([desc_stats, additional_stats], axis=1)
        numeric_stats['data_type'] = 'numeric'

    # Calculate statistics for categorical variables
    categorical_stats = pd.DataFrame()

    if categorical_cols:
        cat_stats_list = []

        for col in categorical_cols:
            col_stats = {
                'count': df[col].count(),
                'missing': df[col].isnull().sum(),
                'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
                'unique': df[col].nunique(),
                'top': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else np.nan,
                'freq': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else np.nan,
                'data_type': 'categorical'
            }
            cat_stats_list.append(col_stats)

        categorical_stats = pd.DataFrame(cat_stats_list, index=categorical_cols)

    # Combine all statistics
    if not numeric_stats.empty and not categorical_stats.empty:
        # Align columns for concatenation
        all_cols = set(numeric_stats.columns) | set(categorical_stats.columns)

        for col in all_cols:
            if col not in numeric_stats.columns:
                numeric_stats[col] = np.nan
            if col not in categorical_stats.columns:
                categorical_stats[col] = np.nan

        # Reorder columns to match
        numeric_stats = numeric_stats.reindex(columns=sorted(all_cols))
        categorical_stats = categorical_stats.reindex(columns=sorted(all_cols))

        final_stats = pd.concat([numeric_stats, categorical_stats])
    elif not numeric_stats.empty:
        final_stats = numeric_stats
    elif not categorical_stats.empty:
        final_stats = categorical_stats
    else:
        print("No valid columns found for analysis.")
        return None

    # Reorder columns for better readability
    col_order = ['data_type', 'count', 'missing', 'missing_pct']

    # Add numeric-specific columns if they exist
    numeric_order = ['mean', 'std', 'variance', 'min', '25%', '50%', '75%', 'max',
                     'range', 'iqr', 'skewness', 'kurtosis', 'cv']

    # Add categorical-specific columns if they exist
    cat_order = ['unique', 'top', 'freq']

    # Build final column order
    final_col_order = col_order.copy()
    for col in numeric_order + cat_order:
        if col in final_stats.columns:
            final_col_order.append(col)

    # Add any remaining columns
    remaining_cols = [col for col in final_stats.columns if col not in final_col_order]
    final_col_order.extend(remaining_cols)

    final_stats = final_stats.reindex(columns=final_col_order)

    return final_stats


def display_stats_summary(stats_df):
    """Display a summary of the descriptive statistics."""
    if stats_df is None:
        return

    print("\nDESCRIPTIVE STATISTICS SUMMARY")
    print("=" * 60)

    numeric_vars = stats_df[stats_df['data_type'] == 'numeric']
    categorical_vars = stats_df[stats_df['data_type'] == 'categorical']

    print(f"Total variables: {len(stats_df)}")
    print(f"Numeric variables: {len(numeric_vars)}")
    print(f"Categorical variables: {len(categorical_vars)}")
    print(f"Variables with missing data: {len(stats_df[stats_df['missing'] > 0])}")

    if len(numeric_vars) > 0:
        print(f"\nHighest missing data (numeric): {numeric_vars['missing_pct'].max():.1f}%")
        print(f"Variable with highest CV: {numeric_vars['cv'].idxmax()} ({numeric_vars['cv'].max():.1f}%)")

    if len(categorical_vars) > 0:
        print(f"Most unique categories: {categorical_vars['unique'].max()}")


# Main execution
if __name__ == "__main__":
    file_path = "data/all_merged_data_clean.csv"

    # Calculate descriptive statistics
    results = calculate_descriptive_stats(file_path)

    if results is not None:
        # Display summary
        display_stats_summary(results)

        print("\n" + "=" * 80)
        print("DETAILED DESCRIPTIVE STATISTICS")
        print("=" * 80)
        print("(Variables as rows, Statistics as columns)")
        print()

        # Display with better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)

        print(results.round(3))

        # Save to file (optional)
        output_file = "descriptive_statistics_coffee.csv"
        results.to_csv(output_file)
        print(f"\nResults saved to: {output_file}")

        # Reset display options
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')