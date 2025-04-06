"""
Standalone FID Calculation Script

This script calculates the Frechet Inception Distance (FID) between a set of
real images (presumed to be already resized) and one or more sets of generated
sample images (provided as either directories containing images or .npz files).
"""
import argparse
import os
import sys
import subprocess
import time

# --- Helper Functions ---

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
    print(f"Executing: {' '.join(command_list)}")
    try:
        process = subprocess.run(command_list, capture_output=True, text=True, check=True, cwd=cwd)
        if process.stdout:
            print("--- FID Output ---")
            print(process.stdout.strip()) # Strip whitespace for cleaner FID score output
            print("------------------")
        if process.stderr:
            # pytorch-fid often prints stats/progress to stderr, treat as info unless return code != 0
            print("--- FID Stderr Stream (Info/Progress) ---")
            print(process.stderr.strip(), file=sys.stderr)
            print("-----------------------------------------")
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

def validate_paths(real_path, generated_path, name):
    """Validates the paths needed for FID calculation."""
    print_info(f"Validating paths for '{name}'...")
    paths_valid = True

    # Validate Real Path
    if not os.path.isdir(real_path):
        print_error(f"Real images path is not a directory: {real_path}")
        paths_valid = False
    else:
        try:
            if not os.listdir(real_path):
                print_error(f"Real images directory is empty: {real_path}")
                paths_valid = False
            else:
                 print_info(f"Real images directory found and is not empty: {real_path}")
        except OSError as e:
            print_error(f"Error accessing real images directory {real_path}: {e}")
            paths_valid = False

    # Validate Generated Path (can be file or dir)
    is_valid_gen_path = False
    if os.path.isfile(generated_path) and generated_path.endswith('.npz'):
        is_valid_gen_path = True
        print_info(f"Generated samples NPZ file found: {generated_path}")
    elif os.path.isdir(generated_path):
        try:
            if os.listdir(generated_path):
                is_valid_gen_path = True
                print_info(f"Generated samples directory found and is not empty: {generated_path}")
            else:
                print_error(f"Generated samples directory is empty: {generated_path}")
        except OSError as e:
            print_error(f"Error accessing generated samples directory {generated_path}: {e}")
    elif not os.path.exists(generated_path):
         print_error(f"Generated samples path does not exist: {generated_path}")

    if not is_valid_gen_path:
        paths_valid = False

    if not paths_valid:
        print_error(f"Path validation failed for '{name}'. Cannot calculate FID.")
        return False

    print_info(f"Path validation successful for '{name}'.")
    return True


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Calculate FID between resized real images and generated samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--workspace_dir", type=str, required=True,
        help="Path to the main workspace directory created by the training script."
    )
    parser.add_argument(
        "--real_images_dir", type=str, default="real_images_fid_resized",
        help="Directory containing the RESIZED real images (relative to workspace_dir or absolute)."
    )
    parser.add_argument(
        "--samples",
        action='append',  # Allows specifying multiple times
        nargs=2,          # Expects two arguments: name and path
        metavar=('NAME', 'PATH'),
        required=True,
        help="Specify a generated sample set: provide a NAME (e.g., 'Full Attn') and the PATH "
             "(absolute or relative to workspace_dir) to the directory containing generated images OR the .npz file."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device to use for FID calculation."
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of workers for FID calculation data loading."
    )

    args = parser.parse_args()

    print_header("FID Calculation Setup")
    print(f"Workspace Directory: {args.workspace_dir}")
    print(f"Device: {args.device}")
    print(f"Num Workers: {args.num_workers}")

    # Resolve real images path
    if os.path.isabs(args.real_images_dir):
        real_path_abs = args.real_images_dir
    else:
        real_path_abs = os.path.abspath(os.path.join(args.workspace_dir, args.real_images_dir))
    print(f"Absolute Real Images Path: {real_path_abs}")

    # --- Loop through specified sample sets ---
    all_successful = True
    for sample_info in args.samples:
        name, gen_path_rel_or_abs = sample_info
        print_header(f"Calculating FID for: {name}")

        # Resolve generated samples path
        if os.path.isabs(gen_path_rel_or_abs):
            gen_path_abs = gen_path_rel_or_abs
        else:
            gen_path_abs = os.path.abspath(os.path.join(args.workspace_dir, gen_path_rel_or_abs))
        print(f"Absolute Generated Samples Path: {gen_path_abs}")

        # Validate Paths before running FID
        if not validate_paths(real_path_abs, gen_path_abs, name):
            all_successful = False
            continue # Skip to the next sample set if paths are invalid

        # Construct the pytorch-fid command
        # Use sys.executable to ensure the same Python environment is used
        cmd = [
            sys.executable, "-m", "pytorch_fid",
            real_path_abs,
            gen_path_abs,
            "--device", args.device,
            "--num-workers", str(args.num_workers)
        ]

        # Run the command
        success = run_command(cmd, description=f"Calculating FID for {name}")
        if not success:
            all_successful = False
            print_error(f"FID calculation failed for {name}.")
        else:
            print_info(f"FID calculation apparently successful for {name} (check output above for score).")

    print_header("FID Calculation Summary")
    if all_successful:
        print("All requested FID calculations attempted successfully (check logs for scores).")
    else:
        print_warning("One or more FID calculations failed. Please review the logs.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()