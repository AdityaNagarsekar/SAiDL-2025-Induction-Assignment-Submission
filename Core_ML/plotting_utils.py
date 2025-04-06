# plotting_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from pathlib import Path
from collections import defaultdict
import json # For loading/saving if used independently
import argparse # Moved import here for standalone use

# Consistent Color Palette and Styles (Example)
DEFAULT_PALETTE = plt.cm.tab10.colors
LOSS_TYPE_COLORS = {
    'standard': DEFAULT_PALETTE[0],   # Blue
    'normalized': DEFAULT_PALETTE[1], # Orange
    'robust': DEFAULT_PALETTE[2],     # Green
    'apl': DEFAULT_PALETTE[3],        # Red
    'other': DEFAULT_PALETTE[7],      # Gray
    'unknown': DEFAULT_PALETTE[7]
}
LOSS_TYPE_STYLES = {
    'standard': '-',
    'normalized': '--',
    'robust': '-.',
    'apl': ':',
    'other': '-',
    'unknown': '--'
}
LOSS_TYPE_MARKERS = {
    'standard': 'o',
    'normalized': 's',
    'robust': '^',
    'apl': 'D',
    'other': 'x',
    'unknown': '.'
}


def plot_accuracy_vs_noise(results, loss_functions_metadata, float_noise_rates, noise_type, output_dir_path):
    """Plot test accuracy vs noise rate for different loss types."""
    print("Plotting: Accuracy vs Noise Rate...")
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
    plt.figure(figsize=(12, 8))

    plotted_names = []
    all_accuracies_flat = [] # To calculate y-axis limits

    for loss_name, metadata in loss_functions_metadata.items():
        loss_type = metadata.get('type', 'unknown')
        accuracies = []
        valid_rates_for_loss = []

        # Check and retrieve data points that exist
        for rate in float_noise_rates:
            rate_str = str(rate)
            # final_test_acc now refers to the LAST epoch's test accuracy
            if loss_name in results.get(rate_str, {}):
                result_data = results[rate_str][loss_name]
                if result_data and 'final_test_acc' in result_data:
                    acc_value = result_data['final_test_acc']
                    accuracies.append(acc_value)
                    valid_rates_for_loss.append(rate)
                    all_accuracies_flat.append(acc_value)
                else:
                    accuracies.append(np.nan)
                    valid_rates_for_loss.append(rate)


        # Only plot if we have valid (non-NaN) data points
        if valid_rates_for_loss and not np.isnan(accuracies).all():
             plt.plot(valid_rates_for_loss, accuracies,
                     linestyle=LOSS_TYPE_STYLES.get(loss_type, '-'),
                     marker=LOSS_TYPE_MARKERS.get(loss_type, 'o'),
                     label=f"{loss_name}",
                     color=LOSS_TYPE_COLORS.get(loss_type, LOSS_TYPE_COLORS['other']),
                     markersize=6, alpha=0.9)
             plotted_names.append(loss_name)


    if not plotted_names:
        print("  Skipping: No data found for accuracy vs noise plot.")
        plt.close()
        return

    plt.xlabel('Noise Rate', fontsize=14)
    plt.ylabel('Final Test Accuracy (%)', fontsize=14) # Updated Label
    plt.title(f'Final Test Accuracy vs. Noise Rate ({noise_type} Noise)', fontsize=16) # Updated Title
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, bbox_to_anchor=(1.04, 1), loc='upper left', title="Loss Functions", frameon=True, shadow=True)

    # Adjust y-axis limits based on actual data
    if all_accuracies_flat:
        min_acc = np.nanmin(all_accuracies_flat)
        max_acc = np.nanmax(all_accuracies_flat)
        padding = (max_acc - min_acc) * 0.05
        plt.ylim(bottom=max(0, min_acc - padding), top=min(100, max_acc + padding))
    else:
        plt.ylim(bottom=0, top=100)

    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=len(float_noise_rates)*1.5 , integer=False))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    output_path = output_dir_path / f'accuracy_vs_noise_{noise_type.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_apl_analysis(results, loss_functions_metadata, float_noise_rates, noise_type, output_dir_path):
    """Plot analysis of APL combinations versus their component losses."""
    print("Plotting: APL Analysis...")
    apl_combinations = {name: metadata for name, metadata in loss_functions_metadata.items() if metadata.get('type') == 'apl'}

    if not apl_combinations:
        print("  Skipping: No APL losses found in metadata.")
        return

    plots_generated = 0
    for apl_name, apl_metadata in apl_combinations.items():
        active_name = apl_metadata.get('active')
        passive_name = apl_metadata.get('passive')

        if not active_name or not passive_name:
             if '+' in apl_name:
                  components = apl_name.split('+')
                  if len(components) == 2: active_name, passive_name = components[0], components[1]
                  else: continue
             else: continue

        active_meta = loss_functions_metadata.get(active_name, {})
        passive_meta = loss_functions_metadata.get(passive_name, {})

        required_losses = [apl_name, active_name, passive_name]
        data_to_plot = {'rates': [], apl_name: [], active_name: [], passive_name: []}

        for rate in float_noise_rates:
            rate_str = str(rate)
            rate_results = results.get(rate_str, {})
            # final_test_acc now refers to last epoch's accuracy
            if not all(loss in rate_results and rate_results[loss] and 'final_test_acc' in rate_results[loss] for loss in required_losses):
                print(f"  Warning: Missing complete data for APL plot '{apl_name}' at noise rate {rate}. Skipping this rate.")
                continue

            data_to_plot['rates'].append(rate)
            data_to_plot[apl_name].append(rate_results[apl_name]['final_test_acc'])
            data_to_plot[active_name].append(rate_results[active_name]['final_test_acc'])
            data_to_plot[passive_name].append(rate_results[passive_name]['final_test_acc'])

        if not data_to_plot['rates']:
            print(f"  Skipping APL plot for {apl_name}: No rates with complete data.")
            continue

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))

        plt.plot(data_to_plot['rates'], data_to_plot[apl_name], marker='D', linestyle=':', linewidth=2.5, markersize=7, label=f"{apl_name} (APL)", color=LOSS_TYPE_COLORS['apl'], zorder=10)
        plt.plot(data_to_plot['rates'], data_to_plot[active_name], marker=LOSS_TYPE_MARKERS.get(active_meta.get('type')), linestyle='--', linewidth=1.5, label=active_name, color=LOSS_TYPE_COLORS.get(active_meta.get('type'),'gray'))
        plt.plot(data_to_plot['rates'], data_to_plot[passive_name], marker=LOSS_TYPE_MARKERS.get(passive_meta.get('type')), linestyle='-.', linewidth=1.5, label=passive_name, color=LOSS_TYPE_COLORS.get(passive_meta.get('type'),'gray'))

        plt.xlabel('Noise Rate', fontsize=12)
        plt.ylabel('Final Test Accuracy (%)', fontsize=12) # Updated Label
        plt.title(f'APL Comparison: {apl_name} vs Components ({noise_type} Noise)', fontsize=14)
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.legend(fontsize=10)
        plt.ylim(bottom=0)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=len(data_to_plot['rates']), integer=False))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.tight_layout()
        output_path = output_dir_path / f'apl_comparison_{apl_name}_{noise_type.lower()}.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        plots_generated += 1
        print(f"  Saved: {output_path.name}")


    if plots_generated == 0:
        print("  No APL comparison plots were generated.")


def plot_learning_curves(results, loss_functions_metadata, float_noise_rates, noise_type, output_dir_path):
    """
    Plot learning curves (train and test accuracy vs. epoch), generating separate
    plots for different loss categories. Within each plot, assign unique color/style
    combinations to each loss function line.
    """
    print("Plotting: Learning Curves (Grouped by Category, Unique Styles)...") # Updated message

    total_plots_generated = 0

    for noise_rate in float_noise_rates:
        rate_str = str(noise_rate)
        rate_data = results.get(rate_str, {})
        if not rate_data:
            continue

        # --- Group losses by category for this noise rate ---
        losses_by_category = defaultdict(list)
        available_losses_at_rate = []
        for loss_name in sorted(rate_data.keys()):
            if rate_data[loss_name]:
                history = rate_data[loss_name].get('history')
                if history and history.get('train_accs') and history.get('test_accs'):
                     available_losses_at_rate.append(loss_name)
                     # Use 'category' from metadata for grouping plots
                     category = loss_functions_metadata.get(loss_name, {}).get('category', 'unknown')
                     losses_by_category[category].append(loss_name)

        if not available_losses_at_rate:
            continue

        print(f"  Generating learning curve plots for noise rate {noise_rate}...")
        plots_generated_this_rate = 0

        # --- Loop through each category and create a plot ---
        for category, loss_names_in_category in losses_by_category.items():
            if not loss_names_in_category:
                continue

            print(f"    Plotting category: '{category}' ({len(loss_names_in_category)} losses)")

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True) # Train | Test
            any_data_plotted, max_epochs = False, 0

            # <<< DEFINE Styles for cycling within THIS category plot >>>
            # Use a larger palette like tab20 for more distinct colors
            category_colors = plt.cm.tab20.colors
            category_linestyles = ['-', '--', '-.', ':']
            # Optional: Add markers if linestyles aren't enough
            # category_markers = ['o', 's', '^', 'D', 'v', 'p', '*', '+', '<', '>']

            # Plot losses within this category, assigning unique styles
            for idx, loss_name in enumerate(loss_names_in_category): # Use enumerate for index
                hist = rate_data[loss_name].get('history', {})
                tr_acc, te_acc = hist.get('train_accs', []), hist.get('test_accs', [])
                curr_ep = min(len(tr_acc), len(te_acc))
                if curr_ep == 0: continue

                ep_ax = range(1, curr_ep + 1)
                max_epochs = max(max_epochs, curr_ep)

                # <<< Assign unique style based on index within category >>>
                color = category_colors[idx % len(category_colors)] # Cycle through colors
                ls = category_linestyles[idx % len(category_linestyles)] # Cycle through linestyles
                # marker = category_markers[idx % len(category_markers)] # Optional

                # Plot using the assigned unique style
                axes[0].plot(ep_ax, tr_acc[:curr_ep], linestyle=ls, color=color, alpha=0.9, label=f"{loss_name}")#, marker=marker, markevery=max(1, curr_ep // 10)) # Example: marker every 10%
                axes[1].plot(ep_ax, te_acc[:curr_ep], linestyle=ls, color=color, alpha=0.9, label=f"{loss_name}")#, marker=marker, markevery=max(1, curr_ep // 10))
                any_data_plotted = True

            if not any_data_plotted:
                plt.close(fig)
                print(f"      Failed to plot any data for category '{category}'.")
                continue

            # --- Styling for the category plot (Titles, Labels, Legend, Saving as before) ---
            for ax in axes:
                ax.grid(True, alpha=0.4, linestyle='--'); ax.tick_params(axis='both', which='major', labelsize=10)
                ax.set_xlabel('Epoch', fontsize=12);
                if max_epochs > 0: ax.set_xlim(left=0.5, right=max_epochs + 0.5); ax.xaxis.set_major_locator(MaxNLocator(nbins=min(10, max_epochs), integer=True))
            axes[0].set_title(f'Train Accuracy ({category.capitalize()} Losses)', fontsize=14); axes[0].set_ylabel('Accuracy (%)', fontsize=12); axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            axes[1].set_title(f'Test Accuracy ({category.capitalize()} Losses)', fontsize=14)
            handles, labels = axes[1].get_legend_handles_labels(); leg_fs = 10; leg_loc, leg_anc = 'upper left', (1.02, 1.0)
            fig.legend(handles, labels, bbox_to_anchor=leg_anc, loc=leg_loc, title="Loss", fontsize=leg_fs, frameon=True)
            fig.suptitle(f'Learning Curves ({noise_type} Noise Rate: {noise_rate}, Category: {category.capitalize()})', fontsize=16, y=1.02)
            all_vals = [item for h in handles for item in h.get_ydata()];
            if all_vals: min_v,max_v=np.nanmin(all_vals),np.nanmax(all_vals); pad=(max_v-min_v)*0.05 if (max_v-min_v)>0 else 1.0; plt.ylim(bottom=max(0,min_v-pad-1), top=min(101,max_v+pad+1))
            else: plt.ylim(bottom=0, top=101)
            plt.tight_layout(rect=[0, 0, 0.88 if leg_anc[0]>1 else 1, 1]);
            category_str = category.replace(" ", "_"); output_path = output_dir_path / f'learning_curves_noise_{noise_rate}_cat_{category_str}_{noise_type.lower()}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close(fig); plots_generated_this_rate += 1
        # --- End category loop ---

        if plots_generated_this_rate > 0: print(f"    Saved {plots_generated_this_rate} learning curve plot(s) for noise rate {noise_rate}.")
        else: print(f"    No learning curve plots generated for noise rate {noise_rate}.")
        total_plots_generated += plots_generated_this_rate
    # --- End noise_rate loop ---

    if total_plots_generated > 0: print(f"\nTotal learning curve plots generated: {total_plots_generated}.")
    else: print("\nNo learning curve plots were generated overall.")

def plot_normalization_effect(results, loss_functions_metadata, noise_rate_to_plot, noise_type, output_dir_path):
    """
    Generate plots similar to Figures 1 and 2 in the paper, comparing
    losses over training epochs at a specific noise rate.
    """
    print(f"Plotting: Normalization Effect (Comparison for Fig 1/2) at Noise Rate {noise_rate_to_plot}...") # No changes needed here

    noise_rate_str = str(noise_rate_to_plot)
    if noise_rate_str not in results:
        print(f"  Skipping: Results for noise rate {noise_rate_to_plot} not found.")
        return

    epoch_data = results[noise_rate_str]

    pairs_nonrobust = [('CE', 'NCE'), ('FL', 'NFL')]
    pairs_robust = [('MAE', 'NMAE'), ('RCE', 'NRCE')]

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Non-Robust vs Normalized Non-Robust ---
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))
    has_data_1 = False
    colors1 = plt.cm.tab10.colors
    max_epochs1 = 0

    for i, pair in enumerate(pairs_nonrobust):
        unnorm, norm = pair
        color_idx = i * 2

        if unnorm in epoch_data and epoch_data[unnorm] and 'history' in epoch_data[unnorm] and epoch_data[unnorm]['history'].get('test_accs'):
            test_accs = epoch_data[unnorm]['history']['test_accs']
            if test_accs: # Check if list is not empty
                epochs = range(1, len(test_accs) + 1)
                max_epochs1 = max(max_epochs1, len(test_accs))
                ax1.plot(epochs, test_accs, linestyle='-', label=unnorm, color=colors1[color_idx])
                has_data_1 = True
        else: print(f"    Missing history data for {unnorm} at noise {noise_rate_to_plot}")

        if norm in epoch_data and epoch_data[norm] and 'history' in epoch_data[norm] and epoch_data[norm]['history'].get('test_accs'):
            test_accs = epoch_data[norm]['history']['test_accs']
            if test_accs: # Check if list is not empty
                epochs = range(1, len(test_accs) + 1)
                max_epochs1 = max(max_epochs1, len(test_accs))
                ax1.plot(epochs, test_accs, linestyle='--', label=norm, color=colors1[color_idx + 1])
                has_data_1 = True
        else: print(f"    Missing history data for {norm} at noise {noise_rate_to_plot}")

    if has_data_1:
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax1.set_title(f'Robustness via Normalization (Standard Losses)\nNoise Rate {noise_rate_to_plot} ({noise_type})', fontsize=14)
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.legend(fontsize=10, loc='best', frameon=True)
        ax1.set_ylim(bottom=0)
        if max_epochs1 > 0: ax1.set_xlim(left=1, right=max_epochs1)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        fig1.tight_layout()
        output_path1 = output_dir_path / f'paper_fig1-2_style_std_{noise_rate_to_plot}_{noise_type.lower()}.png'
        fig1.savefig(output_path1, dpi=300)
        plt.close(fig1)
        print(f"  Saved: {output_path1.name}")
    else:
        plt.close(fig1)
        print("  Skipping standard loss normalization plot: No data found.")

    # --- Plot 2: Robust vs Normalized Robust ---
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    has_data_2 = False
    colors2 = plt.cm.plasma(np.linspace(0.2, 0.9, 4))
    max_epochs2 = 0

    for i, pair in enumerate(pairs_robust):
        robust, norm_robust = pair
        color_idx = i * 2

        if robust in epoch_data and epoch_data[robust] and 'history' in epoch_data[robust] and epoch_data[robust]['history'].get('test_accs'):
            test_accs = epoch_data[robust]['history']['test_accs']
            if test_accs: # Check if list is not empty
                epochs = range(1, len(test_accs) + 1)
                max_epochs2 = max(max_epochs2, len(test_accs))
                ax2.plot(epochs, test_accs, linestyle='-', label=robust, color=colors2[color_idx])
                has_data_2 = True
        else: print(f"    Missing history data for {robust} at noise {noise_rate_to_plot}")

        if norm_robust in epoch_data and epoch_data[norm_robust] and 'history' in epoch_data[norm_robust] and epoch_data[norm_robust]['history'].get('test_accs'):
            test_accs = epoch_data[norm_robust]['history']['test_accs']
            if test_accs: # Check if list is not empty
                epochs = range(1, len(test_accs) + 1)
                max_epochs2 = max(max_epochs2, len(test_accs))
                ax2.plot(epochs, test_accs, linestyle='--', label=norm_robust, color=colors2[color_idx+1])
                has_data_2 = True
        else: print(f"    Missing history data for {norm_robust} at noise {noise_rate_to_plot}")

    if has_data_2:
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax2.set_title(f'Effect of Normalizing Robust Losses\nNoise Rate {noise_rate_to_plot} ({noise_type})', fontsize=14)
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.legend(fontsize=10, loc='best', frameon=True)
        ax2.set_ylim(bottom=0)
        if max_epochs2 > 0: ax2.set_xlim(left=1, right=max_epochs2)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        fig2.tight_layout()
        output_path2 = output_dir_path / f'paper_fig1-2_style_robust_{noise_rate_to_plot}_{noise_type.lower()}.png'
        fig2.savefig(output_path2, dpi=300)
        plt.close(fig2)
        print(f"  Saved: {output_path2.name}")
    else:
        plt.close(fig2)
        print("  Skipping robust loss normalization plot: No data found.")


def plot_accuracy_heatmap(results, loss_functions_metadata, float_noise_rates, noise_type, output_dir_path):
    """Create heatmap visualization of test accuracy for all loss functions at all noise rates."""
    print("Plotting: Accuracy Heatmap...") # No changes needed here

    accuracy_data = {}
    loss_order = []

    for category in ['standard', 'normalized', 'robust', 'apl', 'other', 'unknown']:
        for loss_name, metadata in sorted(loss_functions_metadata.items()):
            if metadata.get('type', 'unknown') == category:
                if any(loss_name in results.get(str(rate), {}) for rate in float_noise_rates):
                     # final_test_acc now refers to last epoch's test accuracy
                     row_acc = [results.get(str(rate), {}).get(loss_name, {}).get('final_test_acc', np.nan) for rate in float_noise_rates]
                     if not np.isnan(row_acc).all():
                        accuracy_data[loss_name] = row_acc
                        if loss_name not in loss_order:
                             loss_order.append(loss_name)

    if not accuracy_data:
        print("  Skipping: No data available for heatmap.")
        return

    df = pd.DataFrame(accuracy_data, index=float_noise_rates).T
    df = df.reindex(index=loss_order, fill_value=np.nan)

    fig_height = max(6, len(loss_order) * 0.45)
    fig_width = max(8, len(float_noise_rates) * 1.2)

    plt.style.use('default')
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='viridis_r', linewidths=0.5, linecolor='lightgray',
                cbar_kws={'label': 'Final Test Accuracy (%)'}, annot_kws={"size": 8}) # Updated Label
    plt.title(f'Final Test Accuracy Heatmap ({noise_type} Noise)', fontsize=16, pad=20) # Updated Title
    plt.xlabel("Noise Rate", fontsize=12)
    plt.ylabel("Loss Function", fontsize=12)
    xtick_labels = [f'{r:.2f}' for r in df.columns]
    plt.xticks(ticks=np.arange(len(df.columns)) + 0.5, labels=xtick_labels, rotation=0, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    output_path = output_dir_path / f'accuracy_heatmap_{noise_type.lower()}.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_training_stability(results, loss_functions_metadata, float_noise_rates, noise_type, output_dir_path):
    """Plot training stability (std dev of test accuracy over last N epochs)."""
    print("Plotting: Training Stability...") # No changes needed here

    N_EPOCHS_STABILITY = 10
    noise_rate_to_use = None
    for rate in sorted(float_noise_rates, reverse=True):
        if str(rate) in results and results[str(rate)]:
             noise_rate_to_use = rate
             break

    if noise_rate_to_use is None:
        print("  Skipping: No results found at any noise rate for stability plot.")
        return

    rate_str = str(noise_rate_to_use)
    rate_data = results[rate_str]
    stability_data = []

    for loss_name, metadata in loss_functions_metadata.items():
        if loss_name in rate_data and rate_data[loss_name]:
            history = rate_data[loss_name].get('history')
            if history and 'test_accs' in history:
                test_accs = history['test_accs']
                num_epochs = len(test_accs)
                if num_epochs > 1:
                    lookback = min(N_EPOCHS_STABILITY, num_epochs)
                    stability_std = np.std(test_accs[-lookback:])
                    stability_data.append({
                        'loss_name': loss_name,
                        'loss_type': metadata.get('type', 'unknown'),
                        'stability': stability_std
                    })

    if not stability_data:
        print(f"  Skipping: No stability data calculable at noise rate {noise_rate_to_use}.")
        return

    stability_df = pd.DataFrame(stability_data).sort_values('stability')
    bar_colors = [LOSS_TYPE_COLORS.get(row['loss_type'], LOSS_TYPE_COLORS['other']) for _, row in stability_df.iterrows()]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(max(10, len(stability_df)*0.6), 7))
    bars = plt.bar(stability_df['loss_name'], stability_df['stability'], color=bar_colors)
    mean_stability = np.mean(stability_df['stability'])
    plt.axhline(y=mean_stability, color='k', linestyle='--', linewidth=1, alpha=0.7, label=f'Average Std Dev ({mean_stability:.2f})')

    plt.xlabel('Loss Function', fontsize=12)
    plt.ylabel(f'Std Dev of Test Acc (last {N_EPOCHS_STABILITY} epochs)', fontsize=12)
    plt.title(f'Training Stability (Noise Rate {noise_rate_to_use}, {noise_type})', fontsize=15)
    plt.xticks(rotation=60, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    output_path = output_dir_path / f'training_stability_noise_{noise_rate_to_use}_{noise_type.lower()}.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Saved: {output_path.name}")


# === MODIFIED FUNCTION ===
def plot_overfitting_analysis(results, loss_functions_metadata, float_noise_rates, noise_type, output_dir_path):
    """Analyze and visualize overfitting gap (Train Acc - Test Acc) at the FINAL epoch."""
    print("Plotting: Overfitting Analysis (Final Epoch)...") # Indicate final epoch
    noise_rate_to_use = None
    for rate in sorted(float_noise_rates, reverse=True):
         if str(rate) in results and results[str(rate)]:
              noise_rate_to_use = rate
              break
    if noise_rate_to_use is None:
         print("  Skipping: No results found for overfitting plot.")
         return

    rate_str = str(noise_rate_to_use)
    rate_data = results[rate_str]
    gap_data = []

    for loss_name, metadata in loss_functions_metadata.items():
         if loss_name in rate_data and rate_data[loss_name]:
             history = rate_data[loss_name].get('history')
             # --- REMOVED best_epoch logic ---

             if history and history.get('train_accs') and history.get('test_accs'):
                  # Use the LAST epoch's data
                  if history['train_accs'] and history['test_accs']: # Check lists aren't empty
                      train_acc_final = history['train_accs'][-1]
                      test_acc_final = history['test_accs'][-1]

                      if not (np.isnan(train_acc_final) or np.isnan(test_acc_final)):
                           gap = train_acc_final - test_acc_final
                           gap_data.append({
                               'loss_name': loss_name,
                               'loss_type': metadata.get('type', 'unknown'),
                               'gap': gap
                           })
                  else:
                      print(f"    Empty history for {loss_name} at noise {noise_rate_to_use}")

    if not gap_data:
        print(f"  Skipping: No overfitting data calculable at noise rate {noise_rate_to_use}.")
        return

    gap_df = pd.DataFrame(gap_data).sort_values('gap') # Sort low gap to high gap
    bar_colors = [LOSS_TYPE_COLORS.get(row['loss_type'], LOSS_TYPE_COLORS['other']) for _, row in gap_df.iterrows()]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(max(10, len(gap_df)*0.6), 7))
    bars = plt.bar(gap_df['loss_name'], gap_df['gap'], color=bar_colors)

    plt.xlabel('Loss Function', fontsize=12)
    # --- UPDATED Y-Label ---
    plt.ylabel('Train - Test Accuracy Gap (%) [at Final Epoch]', fontsize=12)
    plt.title(f'Overfitting Analysis (Noise Rate {noise_rate_to_use}, {noise_type})', fontsize=15)
    plt.xticks(rotation=60, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.7) # Line at zero gap
    plt.tight_layout()
    output_path = output_dir_path / f'overfitting_gap_noise_{noise_rate_to_use}_{noise_type.lower()}.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Saved: {output_path.name}")
# === END MODIFIED FUNCTION ===


def plot_training_time(results, loss_functions_metadata, float_noise_rates, noise_type, output_dir_path):
    """Compare and visualize average epoch training time."""
    print("Plotting: Training Time...") # No changes needed here
    time_data = []
    for loss_name, metadata in loss_functions_metadata.items():
        all_avg_times = []
        for rate in float_noise_rates:
             rate_str = str(rate)
             if loss_name in results.get(rate_str, {}) and results[rate_str][loss_name]:
                  avg_epoch_time = results[rate_str][loss_name].get('avg_epoch_time_s')
                  if avg_epoch_time is not None and avg_epoch_time > 0:
                       all_avg_times.append(avg_epoch_time)

        if all_avg_times:
            time_data.append({
                'loss_name': loss_name,
                'loss_type': metadata.get('type', 'unknown'),
                'avg_epoch_time': np.mean(all_avg_times)
            })

    if not time_data:
         print("  Skipping: No training time data found.")
         return

    time_df = pd.DataFrame(time_data).sort_values('avg_epoch_time')
    bar_colors = [LOSS_TYPE_COLORS.get(row['loss_type'], LOSS_TYPE_COLORS['other']) for _, row in time_df.iterrows()]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(max(10, len(time_df)*0.6), 7))
    bars = plt.bar(time_df['loss_name'], time_df['avg_epoch_time'], color=bar_colors)
    mean_time = np.mean(time_df['avg_epoch_time'])
    plt.axhline(y=mean_time, color='k', linestyle='--', linewidth=1, alpha=0.7, label=f'Average ({mean_time:.2f} s)')

    plt.xlabel('Loss Function', fontsize=12)
    plt.ylabel('Average Epoch Time (s) [Across Noise Rates]', fontsize=12)
    plt.title(f'Training Time Comparison ({noise_type})', fontsize=15)
    plt.xticks(rotation=60, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    output_path = output_dir_path / f'training_time_{noise_type.lower()}.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_comprehensive_results(results, loss_functions_metadata, noise_rates, output_dir, symmetric=True, experiment_name="Experiment"):
    """
    Main function to generate all relevant plots for the experiment results.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    noise_type = "Symmetric" if symmetric else "Asymmetric"
    # Convert noise_rates (passed as float list) to strings for lookup, filter out rates with no results
    results_str_keys = {str(k): v for k, v in results.items() if float(k) in noise_rates and v}
    float_noise_rates_with_data = sorted([float(r) for r in results_str_keys.keys()])

    if not float_noise_rates_with_data:
        print("No noise rates found with data in results. Cannot generate plots.")
        return

    # Filter metadata to include only losses present in results
    present_losses_names = set()
    for rate_key in results_str_keys:
        present_losses_names.update(results_str_keys[rate_key].keys())

    plottable_loss_metadata = {k: v for k, v in loss_functions_metadata.items() if k in present_losses_names}
    if not plottable_loss_metadata:
         print("No loss functions found with results to plot.")
         return

    print(f"\n--- Starting Comprehensive Plot Generation ({experiment_name}) ---")
    print(f"Output directory: {output_dir_path}")
    print(f"Noise Type: {noise_type}, Rates with data: {float_noise_rates_with_data}")
    print(f"Losses to plot: {list(plottable_loss_metadata.keys())}")

    # Pass the filtered float rates list and metadata
    plot_accuracy_vs_noise(results_str_keys, plottable_loss_metadata, float_noise_rates_with_data, noise_type, output_dir_path)
    plot_apl_analysis(results_str_keys, plottable_loss_metadata, float_noise_rates_with_data, noise_type, output_dir_path)
    plot_learning_curves(results_str_keys, plottable_loss_metadata, float_noise_rates_with_data, noise_type, output_dir_path)

    # Select appropriate noise rate for normalization plot
    relevant_rates = [r for r in float_noise_rates_with_data if (symmetric and r > 0.4 and r < 0.9) or (not symmetric and r > 0.2)]
    noise_rate_for_norm_plot = max(relevant_rates) if relevant_rates else (max(float_noise_rates_with_data) if float_noise_rates_with_data else None)
    if noise_rate_for_norm_plot is not None:
         plot_normalization_effect(results_str_keys, plottable_loss_metadata, noise_rate_for_norm_plot, noise_type, output_dir_path)

    plot_accuracy_heatmap(results_str_keys, plottable_loss_metadata, float_noise_rates_with_data, noise_type, output_dir_path)
    plot_training_stability(results_str_keys, plottable_loss_metadata, float_noise_rates_with_data, noise_type, output_dir_path)
    plot_overfitting_analysis(results_str_keys, plottable_loss_metadata, float_noise_rates_with_data, noise_type, output_dir_path) # Calls the modified function
    plot_training_time(results_str_keys, plottable_loss_metadata, float_noise_rates_with_data, noise_type, output_dir_path)

    print(f"--- Comprehensive Plot Generation Complete ---")


def generate_text_reports(results_dict, loss_functions_metadata, float_noise_rates, output_dir_path, noise_type):
    """
    Generates text-based summary reports (like console output and CSVs)
    from loaded results data.

    Args:
        results_dict (dict): The 'results' part loaded from JSON.
        loss_functions_metadata (dict): Metadata about the losses run.
        float_noise_rates (list): List of noise rates (float) found in results.
        output_dir_path (Path): Path object for the *main* output directory for this run
                                (e.g., .../cifar10_symmetric_...). Reports saved here.
        noise_type (str): "Symmetric" or "Asymmetric".
    """
    print(f"\n--- Generating Text Reports ({noise_type}) ---")
    loss_names_run = list(loss_functions_metadata.keys())

    # --- Final Summary Table ---
    print(f"\n\n=== Final Test Accuracy (%) - {noise_type.upper()} ===")
    summary_data = {}
    for loss_name in loss_names_run:
        row = {}
        # Use string keys for accessing results_dict
        for rate in float_noise_rates:
            rate_str = str(rate)
            acc = results_dict.get(rate_str, {}).get(loss_name, {}).get('final_test_acc', float('nan'))
            row[rate] = acc # Use float rate as column header
        summary_data[loss_name] = row

    try:
        df = pd.DataFrame.from_dict(summary_data, orient='index')
        df = df.reindex(columns=sorted(float_noise_rates), fill_value=float('nan')) # Ensure columns sorted
        df.index.name = 'Loss Function'
        print(df.round(2).to_string())
        summary_csv_path = output_dir_path / 'summary_final_test_accuracy.csv'
        df.round(2).to_csv(summary_csv_path)
        print(f"Summary table saved to: {summary_csv_path}")
    except Exception as e:
         print(f"Error creating or saving summary dataframe: {e}")

    # --- Best Performers ---
    print(f"\n--- Best Loss Function per Noise Rate ({noise_type}, Final Test Acc) ---")
    best_performers = {}
    for rate in float_noise_rates:
        rate_str = str(rate)
        loss_results = results_dict.get(rate_str, {})
        valid_results = {ln: r['final_test_acc'] for ln, r in loss_results.items() if r and 'final_test_acc' in r and not np.isnan(r['final_test_acc'])}
        if valid_results:
             best_loss_name = max(valid_results, key=valid_results.get)
             best_acc = valid_results[best_loss_name]
             print(f"Noise Rate {rate_str}: {best_loss_name} ({best_acc:.2f}%)")
             best_performers[rate_str] = {'loss': best_loss_name, 'acc': best_acc}
        else:
             print(f"Noise Rate {rate_str}: No successful runs found.")
             best_performers[rate_str] = None

    best_perf_path = output_dir_path / 'best_performers_final_epoch.json'
    try:
        with open(best_perf_path, 'w') as f: json.dump(best_performers, f, indent=4)
        print(f"Best performers saved to: {best_perf_path}")
    except Exception as e: print(f"Error saving best performers: {e}")

    # --- Performance Report (Optional but useful) ---
    print("\n--- Performance Summary ---")
    perf_data = []
    total_time_all = 0
    for rate in float_noise_rates:
        rate_str = str(rate)
        loss_results = results_dict.get(rate_str, {})
        for loss_name, result in loss_results.items():
             if result: # Check if result is not None (i.e., run didn't fail)
                 total_time_all += result.get('total_training_time_s', 0)
                 perf_data.append({
                     'noise_rate': rate,
                     'loss_function': loss_name,
                     'avg_epoch_time_s': result.get('avg_epoch_time_s', 0),
                     'total_train_time_s': result.get('total_training_time_s', 0),
                     'final_test_acc': result.get('final_test_acc', 0),
                     # 'device' might not be in saved results, get from config if needed
                 })
    try:
        perf_df = pd.DataFrame(perf_data)
        if not perf_df.empty: # Avoid error if no performance data found
            perf_df = perf_df.sort_values(by=['noise_rate', 'loss_function'])
            print(f"Total training time across all runs in this set: {total_time_all / 3600:.2f} hours")
            print(perf_df[['noise_rate', 'loss_function', 'avg_epoch_time_s', 'final_test_acc']].round(2).to_string(index=False))
            perf_csv_path = output_dir_path / 'performance_report.csv'
            perf_df.round(3).to_csv(perf_csv_path, index=False)
            print(f"Performance report saved to: {perf_csv_path}")
        else:
             print("No performance data found to generate report.")
    except Exception as e:
         print(f"Error creating or saving performance report dataframe: {e}")

    print(f"--- Text Report Generation Complete ({noise_type}) ---")


# --- Modified main function to load results and generate plots AND reports ---
def create_visualization_report(results_path, plot_output_dir_name="plots", report_output_dir_name="."):
    """
    Loads saved results and generates all standard plots and text reports.

    Args:
        results_path (str or Path): Path to the saved experiment results JSON file.
        plot_output_dir_name (str): Name of subdir for plots (relative to results file dir).
        report_output_dir_name (str): Name of subdir (or '.' for same dir) for text reports
                                     (relative to results file dir). Default is same dir.
    """
    print(f"Creating visualization report from: {results_path}")
    results_path = Path(results_path)
    base_results_dir = results_path.parent # Directory containing the results JSON

    # Define output paths relative to the results directory
    plot_output_dir_path = base_results_dir / plot_output_dir_name
    report_output_dir_path = base_results_dir / report_output_dir_name # Can be the same as base_results_dir if '.'

    plot_output_dir_path.mkdir(parents=True, exist_ok=True)
    if report_output_dir_name != ".": # Only create if it's a subdirectory
        report_output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Plots will be saved to: {plot_output_dir_path}")
    print(f"Text reports will be saved to: {report_output_dir_path}")


    # Load results JSON
    data = {}; results_dict = None; config = {}
    try:
        if results_path.suffix == '.json':
            with open(results_path, 'r') as f: data = json.load(f)
        else: raise ValueError(f"Unsupported results file format: {results_path.suffix}. Expecting .json")

        results_dict = data.get('results'); config = data.get('config_run', data.get('config', {})) # Prioritize config_run

        if not isinstance(results_dict, dict) or not results_dict: print(f"Error: 'results' key invalid in '{results_path}'."); return
        if not config: print(f"Warning: Configuration ('config_run' or 'config') not found.")
    except Exception as e: print(f"Error loading results file '{results_path}': {e}"); import traceback; traceback.print_exc(); return

    # Get noise rates and type from data/config
    try: float_noise_rates_present = sorted([float(k) for k in results_dict.keys() if results_dict[k]])
    except (ValueError, TypeError) as e: print(f"Error converting noise rate keys: {e}."); return
    if not float_noise_rates_present: print("No valid noise rate data found."); return

    symmetric = config.get('symmetric', True); noise_type = "Symmetric" if symmetric else "Asymmetric"
    num_epochs_str = config.get('num_epochs', 'N'); experiment_name = f"CIFAR10_{noise_type.lower()}_ep{num_epochs_str}"

    results_str_keys = {str(k): v for k, v in results_dict.items()} # Keep using string keys for lookups if needed

    # --- Infer loss function metadata ---
    loss_functions_metadata = {}; inferred_loss_names = set().union(*(results_str_keys.get(k, {}).keys() for k in results_str_keys))
    for loss_name in sorted(list(inferred_loss_names)):
         l_type, category, active, passive = 'unknown', 'unknown', None, None
         if '+' in loss_name: parts = loss_name.split('+'); l_type = 'apl'; category = 'combined'; active = parts[0] if len(parts)>0 else '?'; passive=parts[1] if len(parts)>1 else '?'
         elif loss_name.startswith('N') and len(loss_name)>1 and loss_name!='NLNL': l_type='normalized'; category='active' if loss_name in ['NCE','NFL'] else ('passive' if loss_name in ['NMAE','NRCE'] else '?')
         elif loss_name in ['CE','FL','GCE']: l_type='standard'; category='active'
         elif loss_name in ['MAE','RCE']: l_type='robust'; category='passive'
         elif loss_name in ['SCE','NLNL']: l_type='other'; category='other'
         else: l_type='other'; category='unknown'
         loss_functions_metadata[loss_name] = {'type': l_type, 'category': category};
         if active: loss_functions_metadata[loss_name]['active'] = active
         if passive: loss_functions_metadata[loss_name]['passive'] = passive


    # --- Generate Plots ---
    plot_comprehensive_results(
        results=results_str_keys, # Pass dict with string keys
        loss_functions_metadata=loss_functions_metadata,
        noise_rates=float_noise_rates_present, # Pass float list
        output_dir=plot_output_dir_path, # Pass the specific plots dir
        symmetric=symmetric,
        experiment_name=experiment_name
    )

    # --- Generate Text Reports ---
    generate_text_reports(
        results_dict=results_dict, # Pass dict with original keys might be easier here
        loss_functions_metadata=loss_functions_metadata,
        float_noise_rates=float_noise_rates_present,
        output_dir_path=report_output_dir_path, # Save reports here
        noise_type=noise_type
    )

    print(f"\nVisualization report generation complete.")


# <<< Modified Entry point >>>
if __name__ == "__main__":
    parser_vis = argparse.ArgumentParser(description="Generate plots and text reports from saved results JSON.")
    parser_vis.add_argument("results_file", help="Path to the saved experiment_results.json file")
    # Changed output args slightly
    parser_vis.add_argument("--plot_dir", default="plots", help="Subdirectory name for plots (relative to results file location, default: 'plots')")
    parser_vis.add_argument("--report_dir", default=".", help="Subdirectory name for text reports (relative to results file location, default: '.', meaning same directory as results file)")
    args_vis = parser_vis.parse_args()

    create_visualization_report(args_vis.results_file, args_vis.plot_dir, args_vis.report_dir)