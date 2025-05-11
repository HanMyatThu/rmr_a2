# This file will contain the script for plotting ablation study results.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from adjustText import adjust_text # Import for non-overlapping text labels

def load_main_ablation_data(file_path="ablation/ablation_results_full_training.csv"):
    """Loads the main ablation study results from a CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: Main ablation results file not found at {file_path}")
        return pd.DataFrame()
    return pd.read_csv(file_path)

def load_specific_ablation_data(base_dir_template, param_name, param_values):
    """
    Loads ablation data for a specific parameter (e.g., 'num_layers', 'seq_length')
    from subdirectories containing 'training_history.csv' files.

    Args:
        base_dir_template (str): Template for the base directory, e.g., 'ablation/{param_name}'.
        param_name (str): The name of the hyperparameter (e.g., 'num_layers', 'seq_length').
        param_values (list): A list of values for the hyperparameter that correspond to directory names.

    Returns:
        pd.DataFrame: DataFrame with columns ['hyperparameter', 'value', 'recall', 'ndcg', 'run_time'].
    """
    results = []
    for val in param_values:
        param_name_plural = param_name 
        if param_name == "num_layers":
            param_name_plural = "layers"
        elif param_name == "seq_length":
            param_name_plural = "seq"
        else:
            print(f"Warning: Unhandled param_name '{param_name}' for directory construction. Assuming direct use.")

        history_file = os.path.join(base_dir_template.format(param_name=param_name_plural), str(val), "training_history.csv")

        if not os.path.exists(history_file):
            print(f"Warning: training_history.csv not found for {param_name}={val} at {history_file}")
            continue

        try:
            df_history = pd.read_csv(history_file)
            if df_history.empty:
                print(f"Warning: training_history.csv is empty for {param_name}={val}")
                continue

            max_ndcg = df_history['val_ndcg'].max()
            tied_rows_ndcg = df_history[df_history['val_ndcg'] == max_ndcg]
            
            if len(tied_rows_ndcg) == 1:
                best_row = tied_rows_ndcg.iloc[0]
            else:
                # Tie in NDCG, resolve by max recall
                max_recall = tied_rows_ndcg['val_recall'].max()
                tied_rows_recall = tied_rows_ndcg[tied_rows_ndcg['val_recall'] == max_recall]
                if len(tied_rows_recall) == 1:
                    best_row = tied_rows_recall.iloc[0]
                else:
                    # Still tied, resolve by latest epoch
                    best_row = tied_rows_recall.sort_values('epoch', ascending=False).iloc[0]

            recall = best_row['val_recall']
            ndcg = best_row['val_ndcg']
            run_time = df_history.loc[df_history['epoch'] <= best_row['epoch'], 'epoch_time_s'].sum()

            results.append({
                'hyperparameter': param_name,
                'value': str(val), # Ensure value is string for consistency with main ablation CSV if needed
                'recall': recall,
                'ndcg': ndcg,
                'run_time': run_time
            })
        except Exception as e:
            print(f"Error processing file {history_file}: {e}")
            
    return pd.DataFrame(results)

def plot_default_vs_best(df, defaults, output_dir="ablation_plots"):
    """Generates a grouped bar plot comparing default vs best configuration for each hyperparameter."""
    os.makedirs(output_dir, exist_ok=True)
    
    unique_hyperparameters = df['hyperparameter'].unique()
    if not unique_hyperparameters.any():
        print("No hyperparameters found to plot.")
        return
    
    plot_params_map = {
        "batch_size": "batch_size",
        "dropout": "dropout",
        "hidden_size": "embedding_size", 
        "learning_rate": "learning_rate",
        "neg_samples": "neg_samples", 
        "num_layers": "num_layers", 
        "seq_length": "seq_length"
    }

    # Filter and get display names for parameters we will plot
    plotted_params_orig_names = [hp for hp in unique_hyperparameters if hp in plot_params_map]
    if not plotted_params_orig_names:
        print("No plottable hyperparameters found after mapping.")
        return
    
    # Prepare data for the grouped bar plot
    plot_data = []
    for hp_name_orig in plotted_params_orig_names:
        sub_df = df[df['hyperparameter'] == hp_name_orig].copy()
        if sub_df.empty:
            continue

        try:
            sub_df.loc[:, 'value_numeric'] = pd.to_numeric(sub_df['value'], errors='coerce')
        except Exception:
            sub_df.loc[:, 'value_numeric'] = sub_df['value'] 
            
        default_val_from_dict = defaults.get(hp_name_orig)
        default_ndcg = 0
        default_display_val = "N/A"

        if default_val_from_dict is not None:
            if isinstance(default_val_from_dict, (int, float)):
                default_row = sub_df[sub_df['value_numeric'] == default_val_from_dict]
            else:
                default_row = sub_df[sub_df['value'] == str(default_val_from_dict)]
            
            if not default_row.empty:
                default_ndcg = default_row.iloc[0]['ndcg']
                default_display_val = default_row.iloc[0]['value'] 
            else:
                print(f"Default value {default_val_from_dict} for {hp_name_orig} not in its ablation runs. Using 0 NDCG for plot.")
                default_display_val = str(default_val_from_dict)
        else:
            print(f"Default value for {hp_name_orig} not found in DEFAULTS_PLOT. Using 0 NDCG for plot.")

        best_row = sub_df.loc[sub_df['ndcg'].idxmax()]
        best_ndcg = best_row['ndcg']
        best_display_val = best_row['value']
        
        display_hp_name = plot_params_map.get(hp_name_orig, hp_name_orig)
        plot_data.append({'Hyperparameter': display_hp_name, 'Type': 'Default', 'NDCG@10': default_ndcg, 'Value': default_display_val})
        plot_data.append({'Hyperparameter': display_hp_name, 'Type': 'Best', 'NDCG@10': best_ndcg, 'Value': best_display_val})

    if not plot_data:
        print("No data prepared for plotting Default vs Best.")
        return
    
    plot_df = pd.DataFrame(plot_data)
    # Ensure order of parameters on x-axis
    hp_display_names_ordered = [plot_params_map.get(hp, hp) for hp in plotted_params_orig_names]

    plt.figure(figsize=(14, 7)) # Adjusted figure size for horizontal layout
    ax = sns.barplot(x='Hyperparameter', y='NDCG@10', hue='Type', data=plot_df, 
                     palette={'Default': 'blue', 'Best': 'red'}, 
                     order=hp_display_names_ordered, hue_order=['Default', 'Best'])

    # Add text labels on bars using ax.bar_label for robustness
    if len(ax.containers) == 2:
        container_default = ax.containers[0] # Bars for 'Default'
        container_best = ax.containers[1]   # Bars for 'Best'

        # Prepare labels in the correct order for each container
        # The bars within each container are ordered by hp_display_names_ordered
        labels_default = plot_df[plot_df['Type'] == 'Default'].set_index('Hyperparameter').loc[hp_display_names_ordered]['Value'].tolist()
        labels_best = plot_df[plot_df['Type'] == 'Best'].set_index('Hyperparameter').loc[hp_display_names_ordered]['Value'].tolist()

        # Add labels to 'Default' bars
        default_text_objects = ax.bar_label(container_default, labels=labels_default, padding=2, 
                                            fontsize=9, color='black')
        for text_obj in default_text_objects:
            text_obj.set_bbox(dict(facecolor='white', alpha=0.7, pad=0.1, boxstyle='round,pad=0.2'))

        # Add labels to 'Best' bars
        best_text_objects = ax.bar_label(container_best, labels=labels_best, padding=2, 
                                         fontsize=9, color='black')
        for text_obj in best_text_objects:
            text_obj.set_bbox(dict(facecolor='white', alpha=0.7, pad=0.1, boxstyle='round,pad=0.2'))
    else:
        print("Warning: Expected 2 bar containers for labeling. Labels might be incorrect or skipped.")

    ax.set_title('Default vs Best Configuration by Parameter (NDCG@10)', fontsize=14)
    ax.set_xlabel('Hyperparameter', fontsize=12)
    ax.set_ylabel('NDCG@10', fontsize=12)
    plt.xticks(rotation=0) # Keep hyperparameter names horizontal if they fit
    ax.legend(title='Configuration Type')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "default_vs_best_params_grouped.png") # New filename
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

def plot_hyperparameter_impact(df, defaults, output_dir="ablation_plots"):
    """Generates a bar plot showing the % impact of each hyperparameter on Recall@10 and NDCG@10."""
    os.makedirs(output_dir, exist_ok=True)
    impact_data = []

    plot_params_map = {
        "batch_size": "batch_size",
        "dropout": "dropout",
        "hidden_size": "embedding_size", 
        "learning_rate": "learning_rate",
        "neg_samples": "neg_samples", 
        "num_layers": "num_layers", 
        "seq_length": "seq_length",
        "mlp_layers": "mlp_layers"
    }
    hyperparameters_to_plot = [hp for hp in df['hyperparameter'].unique() if hp in plot_params_map]

    for hp_name in hyperparameters_to_plot:
        sub_df = df[df['hyperparameter'] == hp_name].copy()
        if sub_df.empty:
            continue

        default_val = defaults.get(hp_name)
        if default_val is None:
            print(f"Default value for {hp_name} not in DEFAULTS_PLOT, cannot calculate impact accurately.")
            continue
        
        sub_df.loc[:, 'value_numeric'] = pd.to_numeric(sub_df['value'], errors='coerce')

        if isinstance(default_val, (int, float)):
            default_run_series = sub_df[sub_df['value_numeric'] == default_val]
        else:
            default_run_series = sub_df[sub_df['value'] == str(default_val)]

        if default_run_series.empty:
            print(f"Could not find default run for {hp_name}={default_val} to calculate impact. Skipping {hp_name}.")
            continue
        
        default_recall = default_run_series.iloc[0]['recall']
        default_ndcg = default_run_series.iloc[0]['ndcg']

        best_recall_val = sub_df.loc[sub_df['recall'].idxmax()]['recall']
        best_ndcg_val = sub_df.loc[sub_df['ndcg'].idxmax()]['ndcg']

        recall_impact = ((best_recall_val - default_recall) / default_recall * 100) if default_recall != 0 else 0
        ndcg_impact = ((best_ndcg_val - default_ndcg) / default_ndcg * 100) if default_ndcg != 0 else 0
        
        display_hp_name = plot_params_map.get(hp_name, hp_name)

        impact_data.append({'Hyperparameter': display_hp_name, 'Metric': 'Recall@10', 'Performance Impact (%)': recall_impact})
        impact_data.append({'Hyperparameter': display_hp_name, 'Metric': 'NDCG@10', 'Performance Impact (%)': ndcg_impact})

    if not impact_data:
        print("No impact data to plot.")
        return

    impact_df = pd.DataFrame(impact_data)
    sorted_params = impact_df[impact_df['Metric'] == 'NDCG@10'].sort_values('Performance Impact (%)', ascending=False)['Hyperparameter'].unique()
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Hyperparameter', y='Performance Impact (%)', hue='Metric', data=impact_df, order=sorted_params, palette=["steelblue", "sandybrown"])
    
    plt.axhline(1, color='lightcoral', linestyle='--', label='1% threshold')
    
    plt.title('Impact of Each Hyperparameter on Model Performance', fontsize=14)
    plt.ylabel('Performance Impact (%)', fontsize=12)
    plt.xlabel('Hyperparameter', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title=None)
    plt.tight_layout()
    
    for p in plt.gca().patches:
        if p.get_height() == 0: 
            continue
        plt.gca().annotate(f"{p.get_height():.2f}%", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontsize=9)

    plot_path = os.path.join(output_dir, "hyperparameter_impact.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

def plot_combined_performance_summary(df, output_dir="ablation_plots"):
    """
    Generates a combined plot with three subplots:
    1. Recall@10 vs NDCG@10
    2. Training Time vs Recall@10
    3. Training Time vs NDCG@10
    """
    os.makedirs(output_dir, exist_ok=True)

    if df.empty or not all(col in df.columns for col in ['recall', 'ndcg', 'run_time', 'hyperparameter', 'value']):
        print("Dataframe is missing required columns for combined performance summary plot. Skipping.")
        print(f"Columns available: {df.columns.tolist()}")
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    plot_params_map = {
        "batch_size": "batch_size", "dropout": "dropout", "hidden_size": "embedding_size",
        "learning_rate": "learning_rate", "neg_samples": "neg_samples",
        "num_layers": "num_layers", "seq_length": "seq_length"
    }
    df_copy = df.copy()
    df_copy['Parameter Type'] = df_copy['hyperparameter'].map(plot_params_map).fillna(df_copy['hyperparameter'])
    
    # Label text for Recall vs NDCG plot
    df_copy['label_text_recall_ndcg'] = df_copy.apply(
        lambda row: f"{plot_params_map.get(row['hyperparameter'], row['hyperparameter'])[:6].replace('_', '')}_{row['value']}",
        axis=1
    )

    fig, axs = plt.subplots(1, 3, figsize=(24, 7)) # 1 row, 3 columns

    # --- Subplot 1: Recall@10 vs NDCG@10 ---
    sns.scatterplot(data=df_copy, x='recall', y='ndcg', hue='Parameter Type', ax=axs[0], s=100, alpha=0.7, legend="full")
    axs[0].set_title('Performance: Recall vs NDCG', fontsize=14)
    axs[0].set_xlabel('Recall@10', fontsize=12)
    axs[0].set_ylabel('NDCG@10', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend(title='Parameter Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    texts_recall_ndcg = []
    labeled_indices_recall_ndcg = set()
    for param_type in df_copy['Parameter Type'].unique():
        group = df_copy[df_copy['Parameter Type'] == param_type]
        if not group.empty:
            best_ndcg_in_group = group.sort_values('ndcg', ascending=False).iloc[0]
            if best_ndcg_in_group.name not in labeled_indices_recall_ndcg:
                texts_recall_ndcg.append(axs[0].text(best_ndcg_in_group['recall'], best_ndcg_in_group['ndcg'],
                                                     best_ndcg_in_group['label_text_recall_ndcg'], fontsize=8, alpha=0.9))
                labeled_indices_recall_ndcg.add(best_ndcg_in_group.name)

    lr_param_name = 'learning_rate'
    lr_display_value = '0.0001'
    lr_point = df_copy[(df_copy['hyperparameter'] == lr_param_name) & (df_copy['value'] == lr_display_value)]
    if not lr_point.empty and lr_point.index[0] not in labeled_indices_recall_ndcg:
        texts_recall_ndcg.append(axs[0].text(lr_point.iloc[0]['recall'], lr_point.iloc[0]['ndcg'], lr_point.iloc[0]['label_text_recall_ndcg'], fontsize=8))
        labeled_indices_recall_ndcg.add(lr_point.index[0])

    neg_param_name = 'neg_samples'
    neg_example_values = ['1', '8', '32', '50'] # Example value to ensure one label for neg_samples if not best
    for neg_val in neg_example_values:
        neg_point = df_copy[(df_copy['hyperparameter'] == neg_param_name) & (df_copy['value'] == neg_val)]
        if not neg_point.empty and neg_point.index[0] not in labeled_indices_recall_ndcg:
            texts_recall_ndcg.append(axs[0].text(neg_point.iloc[0]['recall'], neg_point.iloc[0]['ndcg'], neg_point.iloc[0]['label_text_recall_ndcg'], fontsize=8))
            labeled_indices_recall_ndcg.add(neg_point.index[0])
            break
    if texts_recall_ndcg:
        adjust_text(texts_recall_ndcg, ax=axs[0], arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # --- Subplot 2: Training Time vs Recall@10 ---
    sns.scatterplot(data=df_copy, x='run_time', y='recall', hue='Parameter Type', ax=axs[1], s=100, alpha=0.7, legend="full")
    axs[1].set_title('Training Time vs Recall@10', fontsize=14)
    axs[1].set_xlabel('Training Time (seconds)', fontsize=12)
    axs[1].set_ylabel('Recall@10', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend(title='Parameter Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # --- Subplot 3: Training Time vs NDCG@10 ---
    sns.scatterplot(data=df_copy, x='run_time', y='ndcg', hue='Parameter Type', ax=axs[2], s=100, alpha=0.7, legend="full")
    axs[2].set_title('Training Time vs NDCG@10', fontsize=14)
    axs[2].set_xlabel('Training Time (seconds)', fontsize=12)
    axs[2].set_ylabel('NDCG@10', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend(title='Parameter Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # Add annotations for the best configuration of each parameter type to time plots
    texts_time_recall = []
    texts_time_ndcg = []

    for param_type_display_name in df_copy['Parameter Type'].unique():
        group_df = df_copy[df_copy['Parameter Type'] == param_type_display_name]
        if group_df.empty:
            continue

        best_in_group = group_df.sort_values(by=['ndcg', 'recall', 'run_time'], ascending=[False, False, True]).iloc[0]
        label_text_time = f"{param_type_display_name}_{best_in_group['value']}"

        texts_time_recall.append(axs[1].text(best_in_group['run_time'], best_in_group['recall'], label_text_time, fontsize=8, alpha=0.9))
        texts_time_ndcg.append(axs[2].text(best_in_group['run_time'], best_in_group['ndcg'], label_text_time, fontsize=8, alpha=0.9))

    if texts_time_recall:
        adjust_text(texts_time_recall, ax=axs[1], arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    if texts_time_ndcg:
        adjust_text(texts_time_ndcg, ax=axs[2], arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legends outside
    plot_path = os.path.join(output_dir, "combined_performance_summary.png")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()

def print_best_overall_configuration(df, defaults):
    """Prints the best hyperparameter configuration found in the ablation data."""
    if df.empty:
        print("No ablation data available to determine the best configuration.")
        return

    # Sort by NDCG (descending) then Recall (descending) to find the best run
    # The .copy() is to avoid SettingWithCopyWarning if df is a slice
    sorted_df = df.copy().sort_values(by=['ndcg', 'recall'], ascending=[False, False])
    
    if sorted_df.empty:
        print("Sorted ablation data is empty. Cannot determine best configuration.")
        return
        
    best_run = sorted_df.iloc[0]
    
    varied_hyperparameter = best_run['hyperparameter']
    varied_value = best_run['value']
    best_ndcg = best_run['ndcg']
    best_recall = best_run['recall']
    best_run_time = best_run['run_time']

    print("\n--- Best Overall Hyperparameter Configuration Found ---")
    print("(Based on highest NDCG@10, then Recall@10, from varying one parameter at a time or specific tests)\n")
    
    print(f"Achieved by varying '{varied_hyperparameter}' to value: {varied_value}")
    print("Resulting Metrics:")
    print(f"  NDCG@10:    {best_ndcg:.4f}")
    print(f"  Recall@10:  {best_recall:.4f}")
    print(f"  Run Time:   {best_run_time:.2f}s")
    print("\nWith other parameters set to their default values during this run:")
    
    # Create a temporary dict of the full configuration for this best run
    best_config_details = defaults.copy()
    best_config_details[varied_hyperparameter] = varied_value # Update with the varied parameter's value

    for param, value in best_config_details.items():
        if param == varied_hyperparameter:
            print(f"  - {param}: {value} (Varied)")
        else:
            print(f"  - {param}: {value} (Default)")
    print("-----------------------------------------------------")

if __name__ == "__main__":
    DEFAULTS_PLOT = {
        'batch_size': 64,
        'dropout': 0.2,
        'hidden_size': 128, 
        'learning_rate': 5e-4,
        'num_layers': 2,
        'neg_samples': 99, 
        'epochs': 5, 
        'seq_length': 20 
    }
    
    main_ablation_df = load_main_ablation_data()
    
    num_layers_values = [2, 4, 6] 
    seq_length_values = [10, 20, 50]

    layers_ablation_df = load_specific_ablation_data(
        base_dir_template="ablation/{param_name}", 
        param_name="num_layers", 
        param_values=num_layers_values
    )
    
    seq_ablation_df = load_specific_ablation_data(
        base_dir_template="ablation/{param_name}", 
        param_name="seq_length", 
        param_values=seq_length_values
    )

    if not main_ablation_df.empty:
        main_ablation_df['value'] = main_ablation_df['value'].astype(str)

    all_ablation_data = pd.concat([main_ablation_df, layers_ablation_df, seq_ablation_df], ignore_index=True)
    
    if not all_ablation_data.empty:
        print("Combined ablation data (first 5 rows):")
        print(all_ablation_data.head())
        
        # Call the new function to print the best configuration
        print_best_overall_configuration(all_ablation_data, DEFAULTS_PLOT)
        
        # Existing plot calls
        plot_default_vs_best(all_ablation_data, DEFAULTS_PLOT)
        plot_hyperparameter_impact(all_ablation_data, DEFAULTS_PLOT)
        plot_combined_performance_summary(all_ablation_data)
    else:
        print("No ablation data loaded. Exiting.") 