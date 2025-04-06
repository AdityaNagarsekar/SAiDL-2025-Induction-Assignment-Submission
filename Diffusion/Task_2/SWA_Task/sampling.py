"""
Evaluate All DiT Checkpoints Script

This script generates sample image grids from ALL checkpoints (.pt files) found
within the 'checkpoints' subdirectories of one or more specified DiT experiment
directories created by the accompanying training script (e.g., swa.py).
"""

import os
import sys
import argparse
import subprocess
import glob
import re # Import regex for sorting

# --- Helper Functions (Adapted from swa.py) ---

def print_header(text):
    """Prints a formatted header."""
    print("\n" + "=" * 80)
    print(f"=== {text.upper()} ")
    print("=" * 80)

def print_error(text):
    """Prints an error message."""
    print(f"\n!!! ERROR: {text}", file=sys.stderr)

def print_warning(text):
    """Prints a warning message."""
    print(f"\n!!! WARNING: {text}")

def print_info(text):
    """Prints an info message."""
    print(f"\n>>> {text}")

def run_command(command_list, cwd=None, description=""):
    """Runs a shell command using subprocess, captures output."""
    if description:
        print_info(f"Running: {description}...")
    # Mask sensitive paths if necessary for cleaner logging, e.g., replace long ckpt path
    logged_command = list(command_list)
    for i, item in enumerate(logged_command):
        if item.endswith(".pt"):
            logged_command[i] = os.path.basename(item) # Show only filename in log
    print(f"Executing: {' '.join(logged_command)}")

    try:
        process = subprocess.run(command_list, capture_output=True, text=True, check=True, cwd=cwd)
        # Only print significant output/stderr if needed, FID/sampling often verbose
        # Minimal output on success:
        # if process.stdout:
        #     print("--- Command Output ---")
        #     print(process.stdout)
        #     print("----------------------")
        # if process.stderr:
        #     print("--- Command Stderr ---")
        #     print(process.stderr)
        #     print("--------------------")
        print_info(f"{description or 'Command'} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description or 'Command'} failed with return code {e.returncode}")
        if e.stdout:
            print("--- Captured STDOUT ---")
            print(e.stdout)
            print("-----------------------")
        if e.stderr:
            print("--- Captured STDERR ---")
            print(e.stderr, file=sys.stderr) # Print stderr to stderr
            print("-----------------------")
        return False
    except Exception as e:
        print_error(f"An unexpected error occurred while running command: {e}")
        return False

def natural_sort_key(s):
    """Key for sorting filenames containing numbers naturally."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# --- Main Evaluation Logic ---

def evaluate_all_checkpoints(args):
    """Finds and evaluates all checkpoints in specified experiment directories."""
    print_header("Starting Checkpoint Evaluation")

    repo_dir = os.path.abspath(args.repo_dir)
    if not os.path.isdir(repo_dir):
        print_error(f"DiT repository directory not found: {repo_dir}")
        return False
    sample_script_path = os.path.join(repo_dir, "sample.py")
    if not os.path.isfile(sample_script_path):
        print_error(f"sample.py script not found in repository: {sample_script_path}")
        return False

    output_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        print_info(f"Output directory: {output_dir}")
    except OSError as e:
        print_error(f"Could not create output directory {output_dir}: {e}")
        return False

    total_success_count = 0
    total_fail_count = 0

    # --- Iterate Through Experiment Directories ---
    for exp_dir_path in args.exp_dir:
        exp_dir_abs = os.path.abspath(exp_dir_path)
        exp_name = os.path.basename(os.path.normpath(exp_dir_abs)) # Get a name like '000-DiT-S-4...'
        print_header(f"Processing Experiment Directory: {exp_name}")

        # --- Validate Paths for this experiment ---
        if not os.path.isdir(exp_dir_abs):
            print_error(f"Experiment directory not found: {exp_dir_abs}")
            total_fail_count += 1 # Count as failure if dir doesn't exist
            continue
        checkpoints_subdir = os.path.join(exp_dir_abs, "checkpoints")
        if not os.path.isdir(checkpoints_subdir):
            print_error(f"Checkpoints subdirectory not found inside experiment directory: {checkpoints_subdir}")
            total_fail_count += 1 # Count as failure if subdir doesn't exist
            continue

        # --- Discover Checkpoints ---
        checkpoint_paths = glob.glob(os.path.join(checkpoints_subdir, '*.pt'))
        if not checkpoint_paths:
            print_warning(f"No checkpoints (.pt files) found in: {checkpoints_subdir}")
            continue # Don't count as failure, just nothing to process

        # Sort checkpoints naturally (e.g., ckpt_500 before ckpt_10000, best/final appropriately)
        checkpoint_paths.sort(key=lambda x: natural_sort_key(os.path.basename(x)))

        print_info(f"Found {len(checkpoint_paths)} checkpoints to evaluate in {exp_name}.")

        # --- Iterate Through Found Checkpoints ---
        for ckpt_path in checkpoint_paths:
            ckpt_basename = os.path.basename(ckpt_path)
            print("-" * 60)
            print(f"Evaluating: {ckpt_basename}")

            # Construct output filename using experiment name and checkpoint name
            ckpt_name_no_ext, _ = os.path.splitext(ckpt_basename)
            # Sanitize exp_name slightly if needed (replace slashes, etc.) - usually basename is fine
            sanitized_exp_name = exp_name.replace('/', '_').replace('\\', '_')
            output_filename = f"samples_{sanitized_exp_name}_{ckpt_name_no_ext}_seed{args.seed}.png"
            output_image_path = os.path.join(output_dir, output_filename)
            print_info(f"Output grid will be saved to: {output_image_path}")

            # Build command
            cmd = [
                sys.executable,
                sample_script_path,
                "--ckpt", ckpt_path,
                "--num-samples", str(args.num_samples),
                "--num-sampling-steps", str(args.sampling_steps),
                "--seed", str(args.seed),
                "--output-path", output_image_path
            ]

            # Run the command
            if run_command(cmd, cwd=repo_dir, description=f"Generating samples for {ckpt_basename} from {exp_name}"):
                if os.path.isfile(output_image_path):
                    print_info(f"Successfully generated sample grid: {output_image_path}")
                    total_success_count += 1
                else:
                    print_error(f"Command ran but output file was not found: {output_image_path}")
                    total_fail_count += 1
            else:
                print_error(f"Sample generation failed for checkpoint: {ckpt_basename}")
                total_fail_count += 1

    print_header("Evaluation Summary")
    print_info(f"Successfully generated grids for {total_success_count} checkpoint(s).")
    if total_fail_count > 0:
        print_warning(f"Failed to generate grids for {total_fail_count} checkpoint(s).")

    return total_fail_count == 0

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ALL DiT checkpoints in specified experiment directories by generating sample grids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--exp-dir", type=str, required=True, nargs='+',
                        help="One or more paths to the DiT experiment directories containing 'checkpoints' subdirs.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the generated sample grids.")
    parser.add_argument("--repo-dir", type=str, default="dit_workspace/DiT",
                        help="Path to the cloned DiT repository containing sample.py.")
    parser.add_argument("--num-samples", type=int, default=16,
                        help="Number of samples per grid.")
    parser.add_argument("--sampling-steps", type=int, default=1000,
                        help="Number of DDPM/DDIM steps for sampling.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampling (for reproducibility).")

    args = parser.parse_args()

    if not evaluate_all_checkpoints(args):
        sys.exit(1) # Exit with error code if any evaluation failed
    else:
        sys.exit(0) # Exit successfully