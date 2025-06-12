import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


class ModelConfidenceSet:
    """
    Implementation of the Model Confidence Set (MCS) procedure
    by Hansen, Lunde, and Nason (2011)
    """

    def __init__(self, alpha=0.1, B=1000, w=0.5):
        """
        Parameters:
        alpha: significance level (default 0.1 for 10% MCS)
        B: number of bootstrap samples
        w: weight for combining max and range statistics
        """
        self.alpha = alpha
        self.B = B
        self.w = w

    def _relative_performance(self, losses):
        """Calculate relative performance matrix"""
        n_models = losses.shape[1]
        n_obs = losses.shape[0]

        # Calculate pairwise differences
        rel_perf = np.zeros((n_obs, n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                rel_perf[:, i, j] = losses[:, i] - losses[:, j]

        return rel_perf

    def _test_statistics(self, rel_perf):
        """Calculate test statistics for MCS procedure"""
        n_obs, n_models, _ = rel_perf.shape

        # Average relative performance
        avg_rel_perf = np.mean(rel_perf, axis=0)

        # Variance estimation (using HAC estimator)
        var_matrix = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                series = rel_perf[:, i, j]
                var_matrix[i, j] = np.var(series, ddof=1)

        # Test statistics
        t_max = np.zeros(n_models)
        t_range = np.zeros(n_models)

        for i in range(n_models):
            # max statistic
            max_vals = []
            range_vals = []

            for j in range(n_models):
                if i != j:
                    if var_matrix[i, j] > 0:
                        t_ij = avg_rel_perf[i, j] / np.sqrt(var_matrix[i, j] / n_obs)
                        max_vals.append(t_ij)
                        range_vals.append(abs(t_ij))

            t_max[i] = max(max_vals) if max_vals else 0
            t_range[i] = max(range_vals) if range_vals else 0

        return t_max, t_range, avg_rel_perf, var_matrix

    def _bootstrap_critical_values(self, rel_perf, var_matrix):
        """Generate bootstrap critical values"""
        n_obs, n_models, _ = rel_perf.shape

        bootstrap_max = []
        bootstrap_range = []

        # Set random seed for reproducibility
        np.random.seed(42)

        for b in range(self.B):
            # Resample with replacement
            boot_indices = np.random.choice(n_obs, n_obs, replace=True)
            boot_rel_perf = rel_perf[boot_indices]

            # Center the bootstrap sample
            boot_avg = np.mean(boot_rel_perf, axis=0)
            centered_boot = boot_rel_perf - boot_avg[np.newaxis, :, :]
            recentered_avg = np.mean(centered_boot, axis=0)

            # Calculate bootstrap test statistics
            boot_t_max = []
            boot_t_range = []

            for i in range(n_models):
                max_vals = []
                range_vals = []

                for j in range(n_models):
                    if i != j and var_matrix[i, j] > 0:
                        t_ij = recentered_avg[i, j] / np.sqrt(var_matrix[i, j] / n_obs)
                        max_vals.append(t_ij)
                        range_vals.append(abs(t_ij))

                boot_t_max.append(max(max_vals) if max_vals else 0)
                boot_t_range.append(max(range_vals) if range_vals else 0)

            bootstrap_max.append(max(boot_t_max) if boot_t_max else 0)
            bootstrap_range.append(max(boot_t_range) if boot_t_range else 0)

        return np.array(bootstrap_max), np.array(bootstrap_range)

    def compute_mcs(self, losses, model_names=None):
        """
        Compute Model Confidence Set

        Parameters:
        losses: array-like (n_observations x n_models)
        model_names: list of model names

        Returns:
        dict with MCS results
        """
        losses = np.array(losses)
        n_obs, n_models = losses.shape

        if model_names is None:
            model_names = [f'Model_{i}' for i in range(n_models)]

        # Initialize
        included_models = list(range(n_models))
        eliminated_models = []
        mcs_p_values = {}

        iteration = 0

        while len(included_models) > 1:
            iteration += 1
            current_losses = losses[:, included_models]
            current_names = [model_names[i] for i in included_models]

            # Calculate relative performance
            rel_perf = self._relative_performance(current_losses)

            # Calculate test statistics
            t_max, t_range, avg_rel_perf, var_matrix = self._test_statistics(rel_perf)

            # Bootstrap critical values
            boot_max, boot_range = self._bootstrap_critical_values(rel_perf, var_matrix)

            # Combined test statistic
            t_combined = self.w * t_max + (1 - self.w) * t_range
            boot_combined = self.w * boot_max + (1 - self.w) * boot_range

            # P-values
            p_values = []
            for i in range(len(included_models)):
                p_val = np.mean(boot_combined >= t_combined[i])
                p_values.append(p_val)
                mcs_p_values[model_names[included_models[i]]] = p_val

            # Find model to eliminate (worst performing with p-value < alpha)
            worst_model_idx = None
            min_p_value = float('inf')

            for i, p_val in enumerate(p_values):
                if p_val < self.alpha and p_val < min_p_value:
                    min_p_value = p_val
                    worst_model_idx = i

            if worst_model_idx is not None:
                eliminated_model = included_models.pop(worst_model_idx)
                eliminated_models.append(eliminated_model)
            else:
                break

        # Final MCS
        mcs_models = [model_names[i] for i in included_models]

        return {
            'mcs_models': mcs_models,
            'mcs_indices': included_models,
            'eliminated_models': [model_names[i] for i in eliminated_models],
            'p_values': mcs_p_values,
            'n_iterations': iteration
        }


def load_and_combine_data():
    """Load data from all four CSV files"""

    try:
        # Load the four datasets
        elasticnet_df = pd.read_csv('elasticnet_fold_losses_mcs.csv')
        lasso_df = pd.read_csv('lasso_fold_losses_mcs.csv')
        ols_df = pd.read_csv('ols_fold_losses_mcs.csv')
        bagging_df = pd.read_csv('bagging_fold_losses_mcs.csv')

        # Combine all datasets
        combined_df = pd.concat([elasticnet_df, lasso_df, ols_df, bagging_df], ignore_index=True)

        print(f"Total datasets loaded: {len(combined_df)}")
        print(f"Model types: {sorted(combined_df['model_type'].unique())}")
        print(f"Volatility types: {sorted(combined_df['volatility_type'].unique())}")
        print(f"Horizons: {sorted(combined_df['horizon'].unique())}")
        print(f"Target variables: {sorted(combined_df['target_variable'].unique())}")

        # Check for required columns
        required_cols = ['model_type', 'volatility_type', 'horizon', 'fold', 'model_id', 'mse', 'mae', 'qlike']
        missing_cols = [col for col in required_cols if col not in combined_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")

        # Print data overview
        print(f"\nDataset overview:")
        print(f"- Total rows: {len(combined_df)}")
        print(f"- Unique model_ids: {combined_df['model_id'].nunique()}")
        print(
            f"- Unique combinations of (volatility_type, horizon): {len(combined_df.groupby(['volatility_type', 'horizon']))}")

        return combined_df

    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        print("Please ensure all required CSV files are in the current directory:")
        print("- elasticnet_clean_fold_losses_mcs.csv")
        print("- lasso_clean_fold_losses_mcs.csv")
        print("- ols_fold_losses_mcs.csv")
        print("- bagging_clean_fold_losses_mcs.csv")
        raise
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def run_mcs_analysis(df, loss_metric='mse'):
    """
    Run MCS analysis for each volatility type and horizon combination

    Parameters:
    df: combined dataframe with all model results
    loss_metric: which loss metric to use ('mse', 'mae', 'qlike')
    """

    mcs = ModelConfidenceSet(alpha=0.1, B=1000)  # 10% MCS with 1000 bootstrap samples
    results = {}
    detailed_results = []  # For CSV output

    # Get unique combinations of volatility_type and horizon
    volatility_types = sorted(df['volatility_type'].unique())
    horizons = sorted(df['horizon'].unique())

    print(f"\nRunning MCS analysis using {loss_metric.upper()} loss metric...")
    print("=" * 60)

    for vol_type in volatility_types:
        for horizon in horizons:
            print(f"\nVolatility Type: {vol_type}, Horizon: {horizon}")
            print("-" * 40)

            # Filter data for current volatility type and horizon
            subset = df[(df['volatility_type'] == vol_type) &
                        (df['horizon'] == horizon)]

            if len(subset) == 0:
                print(f"No data available for {vol_type}, horizon {horizon}")
                continue

            # Check if loss metric exists
            if loss_metric not in subset.columns:
                print(f"Loss metric '{loss_metric}' not found in data")
                continue

            # Pivot to get losses matrix (folds x models)
            try:
                pivot_data = subset.pivot_table(
                    index='fold',
                    columns='model_id',
                    values=loss_metric,
                    aggfunc='mean'
                )
            except Exception as e:
                print(f"Error creating pivot table: {str(e)}")
                continue

            if pivot_data.empty or pivot_data.shape[1] < 2:
                print(f"Insufficient models for MCS analysis (need at least 2 models)")
                continue

            # Remove any models with missing data
            pivot_data = pivot_data.dropna(axis=1)

            if pivot_data.shape[1] < 2:
                print(f"Insufficient models after removing missing data")
                continue

            # Check if we have sufficient observations
            if pivot_data.shape[0] < 3:
                print(f"Insufficient observations for MCS analysis (need at least 3 folds)")
                continue

            # Prepare data for MCS
            losses_matrix = pivot_data.values
            model_names = list(pivot_data.columns)

            print(f"Models included: {model_names}")
            print(f"Number of folds: {losses_matrix.shape[0]}")

            # Run MCS
            try:
                mcs_result = mcs.compute_mcs(losses_matrix, model_names)

                # Store results
                key = (vol_type, horizon)
                results[key] = mcs_result

                # Calculate average losses for all models
                avg_losses = np.mean(losses_matrix, axis=0)
                model_performance = dict(zip(model_names, avg_losses))

                # Store detailed results for CSV
                for model in mcs_result['mcs_models']:
                    detailed_results.append({
                        'volatility_type': vol_type,
                        'horizon': horizon,
                        'loss_metric': loss_metric,
                        'model_id': model,
                        'in_mcs': True,
                        'mcs_p_value': mcs_result['p_values'][model],
                        'avg_loss': model_performance[model],
                        'n_models_in_mcs': len(mcs_result['mcs_models']),
                        'total_models_tested': len(model_names),
                        'mcs_models_list': ';'.join(mcs_result['mcs_models'])  # All models in MCS
                    })

                # Also store eliminated models for completeness
                for model in mcs_result['eliminated_models']:
                    detailed_results.append({
                        'volatility_type': vol_type,
                        'horizon': horizon,
                        'loss_metric': loss_metric,
                        'model_id': model,
                        'in_mcs': False,
                        'mcs_p_value': mcs_result['p_values'][model],
                        'avg_loss': model_performance[model],
                        'n_models_in_mcs': len(mcs_result['mcs_models']),
                        'total_models_tested': len(model_names),
                        'mcs_models_list': ';'.join(mcs_result['mcs_models'])
                    })

                # Display results
                print(f"MCS Models (10% level): {mcs_result['mcs_models']}")
                print(f"Number of models in MCS: {len(mcs_result['mcs_models'])}")
                print(f"Eliminated models: {mcs_result['eliminated_models']}")
                print(f"Iterations: {mcs_result['n_iterations']}")

                # Show p-values for MCS models
                print("\nP-values for MCS models:")
                for model in mcs_result['mcs_models']:
                    p_val = mcs_result['p_values'][model]
                    print(f"  {model}: {p_val:.4f}")

                best_model = min(model_performance, key=model_performance.get)
                print(f"\nBest performing model: {best_model} (avg {loss_metric}: {model_performance[best_model]:.6f})")

            except Exception as e:
                print(f"Error in MCS computation: {str(e)}")
                continue

    return results, detailed_results


def summarize_mcs_results(results):
    """Summarize MCS results across all volatility types and horizons"""

    print("\n" + "=" * 80)
    print("SUMMARY OF MCS RESULTS")
    print("=" * 80)

    if not results:
        print("No results to summarize.")
        return

    # Count how often each model appears in MCS
    model_mcs_count = {}
    total_cases = len(results)

    for (vol_type, horizon), result in results.items():
        for model in result['mcs_models']:
            if model not in model_mcs_count:
                model_mcs_count[model] = 0
            model_mcs_count[model] += 1

    # Sort by frequency
    sorted_models = sorted(model_mcs_count.items(), key=lambda x: x[1], reverse=True)

    print(f"\nModel inclusion frequency across {total_cases} cases:")
    print("-" * 50)
    for model, count in sorted_models:
        percentage = (count / total_cases) * 100
        print(f"{model}: {count}/{total_cases} ({percentage:.1f}%)")

    # Summary by volatility type and horizon
    print(f"\nDetailed results by volatility type and horizon:")
    print("-" * 60)

    for (vol_type, horizon), result in results.items():
        mcs_models_str = ", ".join(result['mcs_models'])
        print(f"{vol_type}, H={horizon}: {mcs_models_str}")


def create_mcs_results_summary(results_df):
    """Create additional summary statistics from the results"""

    if results_df.empty:
        print("No results available for summary statistics.")
        return

    print(f"\n{'=' * 80}")
    print("ADDITIONAL SUMMARY STATISTICS")
    print(f"{'=' * 80}")

    # MCS models only
    mcs_df = results_df[results_df['in_mcs'] == True]

    if mcs_df.empty:
        print("No models found in any MCS.")
        return

    # Count by loss metric
    print("\nMCS model counts by loss metric:")
    print("-" * 40)
    for metric in ['mse', 'mae', 'qlike']:
        metric_df = mcs_df[mcs_df['loss_metric'] == metric]
        if not metric_df.empty:
            unique_cases = len(metric_df.groupby(['volatility_type', 'horizon']))
            print(f"{metric.upper()}: {len(metric_df)} model entries across {unique_cases} cases")

    # Average MCS size
    print(f"\nAverage MCS size by loss metric:")
    print("-" * 40)
    try:
        avg_mcs_size = mcs_df.groupby(['volatility_type', 'horizon', 'loss_metric'])['n_models_in_mcs'].first().groupby(
            'loss_metric').mean()
        for metric, avg_size in avg_mcs_size.items():
            print(f"{metric.upper()}: {avg_size:.2f} models on average")
    except Exception as e:
        print(f"Could not calculate average MCS size: {str(e)}")

    # Model frequency across all cases
    print(f"\nModel inclusion frequency across all loss metrics:")
    print("-" * 50)
    model_freq = mcs_df['model_id'].value_counts()
    total_cases = len(mcs_df.groupby(['volatility_type', 'horizon', 'loss_metric']))
    for model, count in model_freq.head(15).items():  # Show top 15
        percentage = (count / total_cases) * 100
        print(f"{model}: {count}/{total_cases} ({percentage:.1f}%)")


# Main execution
if __name__ == "__main__":
    try:
        # Load data
        df = load_and_combine_data()

        all_detailed_results = []

        # Run MCS analysis for each loss metric
        for metric in ['mse', 'mae', 'qlike']:
            print(f"\n{'=' * 80}")
            print(f"MCS ANALYSIS FOR {metric.upper()}")
            print(f"{'=' * 80}")

            results, detailed_results = run_mcs_analysis(df, loss_metric=metric)
            all_detailed_results.extend(detailed_results)
            summarize_mcs_results(results)

        # Convert to DataFrame and save to CSV
        results_df = pd.DataFrame(all_detailed_results)

        if not results_df.empty:
            # Save complete results
            results_df.to_csv('mcs_complete_results.csv', index=False)
            print(f"\n{'=' * 80}")
            print("RESULTS SAVED TO CSV")
            print(f"{'=' * 80}")
            print(f"Complete results saved to: mcs_complete_results.csv")
            print(f"Total rows: {len(results_df)}")

            # Create a summary CSV with only MCS models
            mcs_only_df = results_df[results_df['in_mcs'] == True].copy()
            if not mcs_only_df.empty:
                mcs_only_df = mcs_only_df[['volatility_type', 'horizon', 'loss_metric', 'model_id',
                                           'mcs_p_value', 'avg_loss', 'n_models_in_mcs', 'mcs_models_list']]
                mcs_only_df.to_csv('mcs_models_only.csv', index=False)
                print(f"MCS models only saved to: mcs_models_only.csv")
                print(f"Total MCS entries: {len(mcs_only_df)}")

                # Create a pivot table showing MCS models for each volatility type and horizon
                mcs_summary_list = []
                for (vol_type, horizon, metric), group in mcs_only_df.groupby(
                        ['volatility_type', 'horizon', 'loss_metric']):
                    mcs_models = list(group['model_id'])
                    mcs_summary_list.append({
                        'volatility_type': vol_type,
                        'horizon': horizon,
                        'loss_metric': metric,
                        'mcs_models': ';'.join(mcs_models),
                        'n_models_in_mcs': len(mcs_models),
                        'best_model': group.loc[group['avg_loss'].idxmin(), 'model_id'],
                        'best_model_loss': group['avg_loss'].min()
                    })

                if mcs_summary_list:
                    mcs_summary_df = pd.DataFrame(mcs_summary_list)
                    mcs_summary_df.to_csv('mcs_summary_by_case.csv', index=False)
                    print(f"Summary by case saved to: mcs_summary_by_case.csv")
                    print(f"Total cases: {len(mcs_summary_df)}")

                    # Display sample of results
                    print(f"\nSample of MCS results:")
                    print("-" * 50)
                    sample_cases = mcs_summary_df.head(10)[
                        ['volatility_type', 'horizon', 'loss_metric', 'mcs_models', 'n_models_in_mcs']]
                    for _, row in sample_cases.iterrows():
                        print(
                            f"{row['volatility_type']}, H={row['horizon']}, {row['loss_metric']}: {row['mcs_models']} ({row['n_models_in_mcs']} models)")

                # Create additional summary statistics
                create_mcs_results_summary(results_df)
            else:
                print("No models found in any MCS - check data and parameters")

        else:
            print("No results to save - check data and analysis parameters")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback

        traceback.print_exc()