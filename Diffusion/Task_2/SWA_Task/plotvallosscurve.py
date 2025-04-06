"""
Parses a DiT training log file (log.txt) and plots the
training and validation loss curves vs. training steps/epochs.
"""

import argparse
import os
import re # Regular expressions for parsing
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file_path):
    """Parses the log file to extract training and validation losses."""

    if not os.path.isfile(log_file_path):
        print(f"!!! ERROR: Log file not found at: {log_file_path}")
        return None, None

    # Regular expressions to find relevant lines
    # Handles variations in step formatting (e.g., leading zeros)
    train_loss_regex = re.compile(r"Epoch\s+(\d+)\s+Step\s+(\d+):\s+TrainLoss=([\d.]+)")
    # More specific regex for validation loss to avoid matching other averages
    val_loss_regex = re.compile(r"Average Validation Loss:\s+([\d.]+)")
    # Regex to find the epoch number *just before* the validation summary
    epoch_finish_regex = re.compile(r"------ Epoch\s+(\d+)\s+Finished")


    train_steps = []
    train_losses = []
    val_epochs = []
    val_losses = []

    current_epoch_for_val = -1 # Track the epoch number for the next validation loss

    print(f">>> Parsing log file: {log_file_path}")
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Match Training Loss
                train_match = train_loss_regex.search(line)
                if train_match:
                    # epoch = int(train_match.group(1)) # We plot vs step mainly
                    step = int(train_match.group(2))
                    loss = float(train_match.group(3))
                    train_steps.append(step)
                    train_losses.append(loss)
                    continue # Move to next line once matched

                # Match Epoch Finish line to know the context for the validation loss
                epoch_finish_match = epoch_finish_regex.search(line)
                if epoch_finish_match:
                    current_epoch_for_val = int(epoch_finish_match.group(1))
                    continue # Move to next line

                # Match Validation Loss (needs context from epoch finish)
                val_match = val_loss_regex.search(line)
                if val_match and current_epoch_for_val != -1:
                    loss = float(val_match.group(1))
                    val_epochs.append(current_epoch_for_val) # Store epoch number
                    val_losses.append(loss)
                    # Reset context after capturing validation loss for this epoch
                    # current_epoch_for_val = -1 # Keep it until next epoch finish? Let's keep it.
                    continue # Move to next line


    except Exception as e:
        print(f"!!! ERROR parsing log file: {e}")
        return None, None

    print(f">>> Found {len(train_losses)} training loss points and {len(val_losses)} validation loss points.")

    # Convert lists to numpy arrays for easier handling
    train_data = np.array([train_steps, train_losses]).T if train_steps else np.empty((0, 2))
    val_data = np.array([val_epochs, val_losses]).T if val_epochs else np.empty((0, 2))

    # Sort by step/epoch just in case logs are out of order (unlikely but safe)
    train_data = train_data[train_data[:, 0].argsort()]
    val_data = val_data[val_data[:, 0].argsort()]

    return train_data, val_data


def plot_losses(train_data, val_data, output_filename="loss_curve.png", smooth_train=0):
    """Plots the training and validation loss curves."""

    if train_data.shape[0] == 0 and val_data.shape[0] == 0:
        print("!!! ERROR: No loss data found to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Plot Training Loss vs. Step ---
    if train_data.shape[0] > 0:
        train_steps = train_data[:, 0]
        train_losses = train_data[:, 1]

        # Optional smoothing (simple moving average)
        if smooth_train > 1 and len(train_losses) >= smooth_train:
            kernel = np.ones(smooth_train) / smooth_train
            train_losses_smooth = np.convolve(train_losses, kernel, mode='valid')
            # Adjust steps axis for smoothed data
            steps_smooth = train_steps[smooth_train // 2 : -(smooth_train // 2) + (0 if smooth_train % 2 == 1 else 1) ][:len(train_losses_smooth)]
            ax1.plot(steps_smooth, train_losses_smooth, label=f'Training Loss (Smoothed, window={smooth_train})', color='tab:blue', alpha=0.9)
            ax1.plot(train_steps, train_losses, label='Training Loss (Raw)', color='lightblue', alpha=0.4, linestyle=':') # Show raw data faintly
        else:
            ax1.plot(train_steps, train_losses, label='Training Loss', color='tab:blue')

        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Training Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        # Set y-limit based on training data, potentially starting above 0
        # ax1.set_ylim(bottom=0, top=max(1.0, np.max(train_losses[int(len(train_losses)*0.05):]) * 1.2) ) # Ignore first 5% for limit
        ax1.set_ylim(bottom=0) # Start y-axis at 0 for loss

    # --- Plot Validation Loss vs. Epoch on a secondary axis ---
    if val_data.shape[0] > 0:
        val_epochs = val_data[:, 0]
        val_losses = val_data[:, 1]

        # Create a secondary x-axis for epochs if training steps are plotted
        if train_data.shape[0] > 0:
            ax2 = ax1.twiny() # Share the y-axis, different x-axis
            # Plot validation loss vs Epoch on the secondary axis
            ax2.plot(val_epochs, val_losses, label='Validation Loss', color='tab:red', marker='o', linestyle='--', markersize=4)
            ax2.set_xlabel('Validation Epoch', color='tab:red')
            ax2.tick_params(axis='x', labelcolor='tab:red')
            # Align epoch ticks reasonably with step ticks if possible (can be tricky)
            # ax2.set_xlim(ax1.get_xlim()) # This doesn't work well, need conversion
        else: # If no training data, just plot validation vs epoch on primary axis
             ax1.plot(val_epochs, val_losses, label='Validation Loss', color='tab:red', marker='o', linestyle='--')
             ax1.set_xlabel('Validation Epoch')
             ax1.set_ylabel('Validation Loss', color='tab:red')
             ax1.tick_params(axis='y', labelcolor='tab:red')
             ax1.set_ylim(bottom=0)


    # --- Final Touches ---
    # Add a single legend
    lines, labels = ax1.get_legend_handles_labels()
    if 'ax2' in locals() and ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    else:
         ax1.legend(loc='upper right')

    plt.title('Training and Validation Loss Curves')
    fig.tight_layout() # Adjust plot to prevent labels overlapping

    # Save the plot
    try:
        plt.savefig(output_filename, dpi=150)
        print(f">>> Loss curve plot saved to: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"!!! ERROR saving plot: {e}")

    # plt.show() # Uncomment to display the plot interactively


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training/validation loss from DiT log file.")
    parser.add_argument(
        "log_file",
        type=str,
        help="Path to the log.txt file."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="loss_curve.png",
        help="Filename for the output plot image."
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=10, # Default smoothing window size for training loss
        help="Window size for smoothing the training loss curve (moving average). Set to 0 or 1 to disable smoothing."
    )
    args = parser.parse_args()

    train_data, val_data = parse_log_file(args.log_file)

    if train_data is not None or val_data is not None:
        plot_losses(train_data, val_data, args.output, args.smooth)