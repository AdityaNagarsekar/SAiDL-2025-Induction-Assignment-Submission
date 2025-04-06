# main.py
"""Main script to run CIFAR-10 noise robustness experiments."""

# === Import Statements ===
# --- Core PyTorch & NN ---
import torch
# --- Numerical & Utilities ---
import numpy as np
import os               # File system operations
from pathlib import Path # Object-oriented paths
# --- Experiment Configuration & Reporting ---
import argparse         # Command-line argument parsing
from datetime import datetime # Timestamping output directories
# --- Performance ---
# Import AMP utilities correctly based on PyTorch version (>=1.6)
from torch.amp import autocast, GradScaler # Keep here? Only used in training.py - maybe remove
# --- Local Modules ---
from utils import set_seed
from experiment import run_experiment_set


# === Main Execution Block ===
def main():
    """Parses arguments and orchestrates the running of experiment sets."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run Noise Robustness Experiments (CIFAR-10)')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate for SGD.')
    parser.add_argument('--num_epochs', type=int, default=120, help='Number of training epochs.')
    # Reproducibility & Setup
    parser.add_argument('--random_state', type=int, default=42, help='Seed for random number generators.')
    parser.add_argument('--base_output_dir', type=str, default='results_final_run', help='Base directory to save all results.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader worker processes.')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='Use pinned memory for DataLoader (GPU only).') # Default changed to True
    # Performance & Device
    parser.add_argument('--use_amp', action='store_true', default=True, help='Enable Automatic Mixed Precision (CUDA only).') # Default changed to True
    parser.add_argument('--force_cpu', action='store_true', default=False, help='Force execution on CPU even if CUDA is available.')
    # Experiment Scope Control
    parser.add_argument('--subset_losses', action='store_true', default=False, help='Run only a small subset of loss functions for quick testing.')
    parser.add_argument('--subset_noise_rates', action='store_true', default=False, help='Run only a small subset of noise rates for quick testing.')
    # --- End Argument Parsing ---

    args = parser.parse_args() # Parse command-line arguments

    # Apply random seed globally using the utility function
    set_seed(args.random_state)

    # --- Device Setup ---
    # Design Choice: Auto-select CUDA if available, allow forcing CPU. Enable TF32 for performance on CUDA.
    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        try: # Guard against older PyTorch versions
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except AttributeError:
            print("TF32 flags not available in this PyTorch version.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")
        # Disable AMP if running on CPU
        if args.use_amp:
            print("Warning: AMP requested but CPU selected. Disabling AMP.")
            args.use_amp = False
    # --- End Device Setup ---

    # --- Define Noise Rates ---
    # Design Choice: Allow running full or subset of rates via CLI flag.
    if args.subset_noise_rates:
        # Define smaller sets for quick tests
        symmetric_rates = [0.2, 0.6] # Example subset
        asymmetric_rates = [0.1, 0.4] # Example subset
        print("Running with SUBSET of noise rates.")
    else:
        # Define the full sets used in the papers/report
        symmetric_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
        asymmetric_rates = [0.1, 0.2, 0.3, 0.4] # Note: 0.0 noise often omitted for asymmetric as it's same as clean
        print("Running with FULL set of noise rates.")
    # --- End Noise Rates ---

    # --- Execute Experiment Sets ---
    # Design Choice: Run symmetric and asymmetric experiments sequentially.
    # Can comment out one or the other as needed.

    # Run Symmetric Noise Experiments
    print("\n>>> Starting Symmetric Noise Experiments <<<")
    run_experiment_set(
        symmetric=True, # Flag for noise type
        noise_rates_to_run=symmetric_rates, # Noise levels for this set
        device=device, # Determined device
        base_output_dir_root=args.base_output_dir, # Parent output dir
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        random_state=args.random_state,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_amp=args.use_amp,
        subset_losses=args.subset_losses # Pass subset flag
    )

    # Run Asymmetric Noise Experiments
    print("\n>>> Starting Asymmetric Noise Experiments <<<")
    run_experiment_set(
        symmetric=False, # Flag for noise type
        noise_rates_to_run=asymmetric_rates, # Noise levels for this set
        device=device,
        base_output_dir_root=args.base_output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        random_state=args.random_state, # Use same base seed, noise function uses it internally
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_amp=args.use_amp,
        subset_losses=args.subset_losses # Pass subset flag
    )
    # --- End Experiment Execution ---

    print("\n\n--- ALL EXPERIMENTS COMPLETED ---")

# === Entry Point Guard ===
if __name__ == '__main__':
    # This makes the core training script runnable even without the plotting dependency,
    # useful for testing or running in environments where plotting libraries aren't installed.
    # NOTE: This dummy creation logic is now less critical as experiment.py handles the import gracefully,
    # but keeping it here provides an explicit fallback file if needed.
    plotting_utils_path = Path('plotting_utils.py')
    if not plotting_utils_path.exists():
         print(f"Warning: '{plotting_utils_path}' not found. Creating dummy plotting file.")
         try:
             with open(plotting_utils_path, 'w') as f:
                 # Write minimal Python code for a dummy function
                 f.write("import os\n")
                 f.write("from pathlib import Path\n")
                 f.write("import numpy as np\n") # Added for potential use in dummy
                 f.write("import pandas as pd\n") # Added for potential use in dummy
                 f.write("import json\n") # Added for potential use in dummy
                 f.write("try:\n") # Make matplotlib import optional
                 f.write("    import matplotlib.pyplot as plt\n")
                 f.write("    MPL_AVAILABLE = True\n")
                 f.write("except ImportError:\n")
                 f.write("    MPL_AVAILABLE = False\n\n")
                 f.write("# Dummy function signature matching the original (approximated)\n")
                 f.write("def plot_comprehensive_results(results, loss_functions_metadata, noise_rates, output_dir, symmetric=True, experiment_name='Experiment'):\n")
                 f.write("    print('[Plotting Utils] Dummy plot function called. No plots generated.')\n")
                 f.write("    output_dir = Path(output_dir)\n")
                 f.write("    os.makedirs(output_dir, exist_ok=True)\n")
                 f.write("    # Create a dummy file to indicate it ran\n")
                 f.write("    (output_dir / f'dummy_plot_ran_{experiment_name}.txt').touch()\n")
                 f.write("    print(f'Dummy plot indicator file created in {output_dir}')\n")

                 f.write("# Dummy function signature matching the original (approximated)\n")
                 f.write("def create_visualization_report(results_path, plot_output_dir_name='plots', report_output_dir_name='.'):\n")
                 f.write("    print('[Plotting Utils] Dummy create_visualization_report called.')\n")
                 f.write("    # Add basic logic to check input path and potentially create dummy output dirs\n")
                 f.write("    results_path = Path(results_path)\n")
                 f.write("    if not results_path.is_file():\n")
                 f.write("        print(f'  Warning: Results file not found at {results_path}')\n")
                 f.write("        return\n")
                 f.write("    base_dir = results_path.parent\n")
                 f.write("    plot_dir = base_dir / plot_output_dir_name\n")
                 f.write("    report_dir = base_dir / report_output_dir_name\n")
                 f.write("    plot_dir.mkdir(parents=True, exist_ok=True)\n")
                 f.write("    if report_output_dir_name != '.': report_dir.mkdir(parents=True, exist_ok=True)\n")
                 f.write("    print(f'  (Dummy report: would save plots to {plot_dir}, reports to {report_dir})')\n")

                 f.write("# Dummy function signature matching the original (approximated)\n")
                 f.write("def generate_text_reports(*args, **kwargs):\n")
                 f.write("    print('[Plotting Utils] Dummy generate_text_reports called. No reports generated.')\n")

         except Exception as e_write:
              print(f"Error writing dummy plotting_utils.py: {e_write}")

    # Call the main function to start the process
    main()