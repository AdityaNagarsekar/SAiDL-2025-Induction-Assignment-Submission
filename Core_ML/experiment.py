# experiment.py
"""Contains functions to orchestrate model training and experiment sets."""

import os
import gc
import time
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP # Only needed for isinstance check

# Local imports from other modules in the project
from models import CNN_From_Diagram
from losses import (
    CrossEntropyLoss, FocalLoss, MAELoss, RCELoss,
    NCELoss, NFLLoss, NMAELoss, NRCELoss, APLLoss
)
from training import train, evaluate
from data_setup import get_data_loaders

# Attempt to import plotting utility, handle if missing
try:
    from plotting_utils import plot_comprehensive_results
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: plotting_utils not found or cannot be imported. Plot generation will be skipped.")
    PLOTTING_AVAILABLE = False
    # Define a dummy function if plotting is not available
    def plot_comprehensive_results(*args, **kwargs):
        print("[Dummy Plot Func] Plotting skipped as plotting_utils is unavailable.")
        output_dir = kwargs.get('output_dir', Path('./plots_dummy'))
        os.makedirs(output_dir, exist_ok=True)
        # Optionally create a small file to indicate it ran
        (output_dir / "_plotting_skipped.txt").touch()


# === Model Training Orchestration ===
# Reports performance based on the final epoch. Adds timing. Saves final model state.
def train_model(model, train_loader, test_loader, criterion,
                optimizer, scheduler, num_epochs, device, save_dir=None,
                loss_name=None, use_amp=True):
    """
    Orchestrates the training and evaluation of a model for a fixed number of epochs.
    Reports loss/accuracy based on the final epoch's test set evaluation.
    No validation or early stopping is performed in this version.
    """
    # Create model saving directory if specified
    if save_dir: os.makedirs(save_dir, exist_ok=True)

    # Initialize GradScaler for AMP (enabled based on use_amp flag and device type)
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')

    # History dictionary to store metrics per epoch (simplified: no validation)
    history = {
        'train_losses': [], 'train_accs': [],
        'test_losses': [], 'test_accs': [],
        'learning_rates': [], 'epoch_times': []
    }

    print(f"Starting training for {loss_name}...") # Indicate start
    # --- Main Training Loop ---
    for epoch in range(num_epochs):
        epoch_start_time = time.time() # Track epoch duration

        # --- Train for one epoch ---
        # Pass scheduler only if it's batch-based (handled inside train)
        batch_scheduler = scheduler if isinstance(scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)) else None
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device,
                                      batch_scheduler, use_amp, scaler)

        # --- Evaluate on the Test Set ---
        # No validation evaluation in this iteration
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, use_amp, desc="Testing")

        # --- Step Learning Rate Scheduler (Epoch-wise) ---
        # Step scheduler if it's not batch-based (e.g., CosineAnnealingLR, StepLR)
        if scheduler is not None and not isinstance(scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
            scheduler.step()

        # --- Record History ---
        current_lr = optimizer.param_groups[0]['lr'] # Get current learning rate
        epoch_time = time.time() - epoch_start_time # Calculate epoch duration
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc) # Recording test accuracy
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)
        # --- End Record History ---

        # --- Console Output ---
        # Print summary for the current epoch (Train Loss/Acc, Test Loss/Acc, LR, Time)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] '
            f'Time: {epoch_time:.1f}s | '
            f'LR: {current_lr:.6f} | '
            f'Train L/A: {train_loss:.4f}/{train_acc:.2f}% | '
            f'Test L/A: {test_loss:.4f}/{test_acc:.2f}%' # Reporting test accuracy
        )
        # --- End Console Output ---

    # --- End Training Loop ---

    # --- Final Result Calculation (Based on Last Epoch) ---
    final_test_acc = history['test_accs'][-1] if history['test_accs'] else -1.0
    final_test_loss = history['test_losses'][-1] if history['test_losses'] else -1.0

    # --- Prepare Result Dictionary ---
    result = {
        'history': history, # Full history for potential later analysis/plotting
        'final_test_acc': final_test_acc, # Primary metric: Test accuracy after num_epochs
        'final_test_loss': final_test_loss, # Test loss after num_epochs
        'avg_epoch_time_s': np.mean(history['epoch_times']) if history['epoch_times'] else 0,
        'total_training_time_s': sum(history['epoch_times']) if history['epoch_times'] else 0
    }

    # --- Save Final Model State (Optional) ---
    if save_dir and loss_name:
         # Handle potential DDP model wrapping if DDP were fully implemented
         final_model_state = (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()).copy()
         final_model_path = os.path.join(save_dir, f"{loss_name}_final.pth")
         torch.save(final_model_state, final_model_path)
         print(f"Saved final model state to {final_model_path}")

    # --- Final Print Summary ---
    print(f"\n{loss_name}: Training Complete.")
    print(f"{loss_name}: Final Test Acc (Epoch {num_epochs}): {final_test_acc:.2f}%")

    return result # Return the dictionary containing history and final results

# === Experiment Set Orchestration Function ===
# (all losses, all noise rates) for a single noise type (symmetric or asymmetric)
# to keep the main function cleaner and manage resources effectively.
def run_experiment_set(
    # --- Configuration passed from main ---
    symmetric,              # bool: True for symmetric, False for asymmetric
    noise_rates_to_run,     # list: Noise rates to iterate through
    device,                 # torch.device: CPU or CUDA device
    base_output_dir_root,   # str: Parent directory for saving results (e.g., './results_all')
    num_epochs,             # int: Number of training epochs
    learning_rate,          # float: Initial learning rate
    batch_size,             # int: Batch size
    random_state,           # int: Random seed
    num_workers,            # int: DataLoader workers
    pin_memory,             # bool: DataLoader pin_memory setting
    use_amp,                # bool: Enable Automatic Mixed Precision
    subset_losses,          # bool: Flag to run only a subset of losses (for quick tests)
    loss_defs = None        # dict, optional: Pre-defined loss function dictionary to override defaults
    ):
    """
    Runs a complete set of experiments (multiple noise rates, multiple losses)
    for a specific noise type (symmetric or asymmetric). Handles setup, execution,
    resource management, results saving, and plotting invocation.
    """
    # --- Setup Run Directory and Logging ---
    noise_type = "symmetric" if symmetric else "asymmetric"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a unique name and directory for this specific run set
    exp_name = f"cifar10_{noise_type}_noise_epochs{num_epochs}_{timestamp}"
    run_output_dir = Path(base_output_dir_root) / exp_name # Use pathlib for robustness
    run_output_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT SET: {noise_type.upper()}")
    print(f"Output Directory: {run_output_dir}")
    print(f"Noise Rates: {noise_rates_to_run}")
    print(f"Device: {device}")
    print(f"AMP Enabled: {use_amp}")
    print(f"{'='*80}\n")
    # --- End Setup ---

    # --- Define Loss Functions for this Run ---
    # Design Choice: Define losses within the run function or allow passing them in.
    # Allows flexibility while having defaults aligned with the paper/report.
    if loss_defs is None:
        num_classes = 10
        reduction = 'mean' # Use mean reduction for training
        # Default parameters aligned with Ma et al./Nagarsekar report
        fl_gamma, rce_A, apl_alpha, apl_beta = 0.5, -4.0, 1.0, 1.0

        # Define base losses using corrected/stabilized classes from Iteration 6
        base_loss_functions_defs = {
            'CE': {'loss_fn': CrossEntropyLoss(), 'type': 'standard', 'category': 'active'},
            'FL': {'loss_fn': FocalLoss(gamma=fl_gamma, reduction=reduction), 'type': 'standard', 'category': 'active'},
            'MAE': {'loss_fn': MAELoss(num_classes=num_classes, reduction=reduction), 'type': 'robust', 'category': 'passive'},
            'RCE': {'loss_fn': RCELoss(num_classes=num_classes, A=rce_A, reduction=reduction), 'type': 'robust', 'category': 'passive'},
            'NCE': {'loss_fn': NCELoss(reduction=reduction), 'type': 'normalized', 'category': 'active'},
            'NFL': {'loss_fn': NFLLoss(gamma=fl_gamma, reduction=reduction), 'type': 'normalized', 'category': 'active'},
            'NMAE': {'loss_fn': NMAELoss(num_classes=num_classes, reduction=reduction), 'type': 'normalized', 'category': 'passive'},
            'NRCE': {'loss_fn': NRCELoss(num_classes=num_classes, A=rce_A, reduction=reduction), 'type': 'normalized', 'category': 'passive'}
        }

        # Optional: Select only a subset of base losses if flag is set (for quick tests)
        base_losses_subset = ['CE', 'MAE', 'NCE', 'RCE', 'NFL'] # Example subset
        if subset_losses:
            print(f"Running SUBSET of base losses: {base_losses_subset}")
            base_loss_functions = {name: base_loss_functions_defs[name] for name in base_losses_subset if name in base_loss_functions_defs}
        else:
            base_loss_functions = base_loss_functions_defs

        # Define Active/Passive groups based on the selected base losses
        active_losses = {name: config for name, config in base_loss_functions.items() if config['category'] == 'active'}
        passive_losses = {name: config for name, config in base_loss_functions.items() if config['category'] == 'passive'}

        # Define standard APL combinations using available active/passive components
        apl_combinations = {}
        # Standard combinations tested in the papers
        apl_pairs_to_test = [('NCE', 'MAE'), ('NCE', 'RCE'), ('NFL', 'MAE'), ('NFL', 'RCE')]
        for active_name, passive_name in apl_pairs_to_test:
            # Only create the combination if both components are in the selected base_loss_functions
            if active_name in active_losses and passive_name in passive_losses:
                comb_name = f"{active_name}+{passive_name}"
                apl_combinations[comb_name] = {
                    'loss_fn': APLLoss(
                        active_loss=active_losses[active_name]['loss_fn'],
                        passive_loss=passive_losses[passive_name]['loss_fn'],
                        alpha=apl_alpha, beta=apl_beta, loss_name=comb_name
                    ),
                    'type': 'apl', 'category': 'combined',
                    'active': active_name, 'passive': passive_name, # Store components for analysis
                    'alpha': apl_alpha, 'beta': apl_beta
                }

        # Combine base losses and APL combinations into the final dictionary for this run
        loss_functions_to_run = {**base_loss_functions, **apl_combinations}
    else:
         # Use externally provided loss definitions if passed
         loss_functions_to_run = loss_defs

    print(f"Loss functions included in this run set: {list(loss_functions_to_run.keys())}")
    # --- End Loss Definition ---

    # --- Main Experiment Loop (Noise Rates -> Losses) ---
    results = defaultdict(lambda: defaultdict(dict)) # Nested dict for results[noise_rate][loss_name]
    all_noise_statistics = {}                         # Store noise stats per rate

    # Performance: Clear CUDA cache before starting the potentially long loops
    if device.type == 'cuda': torch.cuda.empty_cache()

    # Outer loop: Iterate through specified noise rates
    for noise_rate in noise_rates_to_run:
        print(f"\n--- Starting Noise Rate Loop: {noise_rate:.2f} ({noise_type} noise) ---")
        # Get data loaders and actual noise statistics for this rate
        train_loader, test_loader, noise_stats = get_data_loaders(
            noise_rate, symmetric, batch_size, random_state, num_workers, pin_memory
        )
        # Store noise stats, converting float rate key to string for JSON compatibility
        all_noise_statistics[str(noise_rate)] = noise_stats
        print(f"DataLoaders created. Effective Noise Rate: {noise_stats.get('actual_noise_rate', 'N/A'):.4f}")

        # Define directory for saving models specific to this noise rate
        models_dir = run_output_dir / 'models' / f"noise_{noise_rate:.1f}"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Inner loop: Iterate through all loss functions defined for this run
        for loss_name, config in loss_functions_to_run.items():
            print(f"\n..... Training Model with Loss: {loss_name} | Noise Rate: {noise_rate:.2f} .....")
            # --- Prepare for Single Model Training ---
            # Instantiate the loss function and move it to the device
            criterion = config['loss_fn'].to(device)
            # Instantiate a *new* model for each loss function run and move to device
            # Design Choice: Fresh model per run ensures fair comparison, avoids transfer effects.
            model = CNN_From_Diagram(num_classes=10).to(device)

            # Instantiate optimizer (SGD) and LR scheduler (Cosine Annealing)
            # Design Choice: Using standard SGD with momentum and weight decay (1e-4) aligned with practice.
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            # --- End Preparation ---

            # --- Run Training and Evaluation for this Model/Loss/Noise ---
            try:
                # Call the main training loop function
                result = train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_epochs=num_epochs,
                    device=device,
                    save_dir=models_dir, # Pass directory to save final model state
                    loss_name=loss_name,  # Pass loss name for saving file
                    use_amp=use_amp       # Pass AMP flag
                )
                # Store the results dictionary (contains history, final metrics, times)
                # Use string key for noise rate for JSON compatibility
                results[str(noise_rate)][loss_name] = result
            # --- Error Handling ---
            except Exception as e:
                # Catch potential errors during training (e.g., from NaN loss in NFL)
                print(f"CRITICAL ERROR during training: {loss_name} @ {noise_rate} ({noise_type}). Run failed.")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc() # Print full stack trace
                # Store None to indicate failure for this run
                results[str(noise_rate)][loss_name] = None # Mark as failed
            # --- Resource Cleanup (CRITICAL for large sweeps) ---
            finally:
                # Ensure cleanup happens even if errors occur
                try: del model, optimizer, scheduler, criterion # Delete large objects
                except NameError: pass # Ignore if they weren't defined due to early error
                gc.collect() # Force Python garbage collection
                if device.type == 'cuda':
                    torch.cuda.empty_cache() # Release unused cached memory on GPU
            # --- End Resource Cleanup ---
        # --- End Loss Loop ---

        print(f"--- Finished Noise Rate Loop: {noise_rate:.2f} ({noise_type} noise) ---")

        # --- Resource Cleanup (DataLoaders) ---
        try: del train_loader, test_loader
        except NameError: pass
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # --- End Loader Cleanup ---
    # --- End Noise Rate Loop ---

    # --- Save Comprehensive Results ---
    results_filepath = run_output_dir / 'experiment_results.json'
    try:
        # Helper function to make NumPy/Path types JSON serializable
        def json_default(o):
            if isinstance(o,(np.int_,np.intc,np.intp,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64)): return int(o)
            elif isinstance(o,(np.float_,np.float16,np.float32,np.float64)): return float(o)
            elif isinstance(o,np.ndarray): return o.tolist()
            elif isinstance(o, Path): return str(o) # Handle pathlib.Path object
            elif isinstance(o, torch.device): return str(o) # Handle device object
            return f'<not serializable: {type(o).__name__}>' # Fallback for unknown types

        # Save run configuration arguments separately for traceability
        run_config = {
            'symmetric': symmetric, 'noise_rates_run': noise_rates_to_run,
            'num_epochs': num_epochs, 'learning_rate': learning_rate,
            'batch_size': batch_size, 'random_state': random_state,
            'num_workers': num_workers, 'pin_memory': pin_memory,
            'use_amp': use_amp, 'subset_losses': subset_losses,
            'device': str(device) # Convert device object to string
        }
        config_path = run_output_dir / 'config_run.json'
        with open(config_path, 'w') as f:
            json.dump(run_config, f, indent=4)

        # Prepare the final output dictionary, ensuring contents are serializable
        # Convert results/stats dicts using json.loads(json.dumps(...)) trick with the default handler
        final_output = {
            'config_run': run_config,
            'results': json.loads(json.dumps(results, default=json_default)),
            'noise_statistics': json.loads(json.dumps(all_noise_statistics, default=json_default)),
            'loss_functions_run': list(loss_functions_to_run.keys()) # List of losses actually used
        }
        # Save the main results JSON file
        with open(results_filepath, 'w') as f:
            json.dump(final_output, f, indent=4)
        print(f"\n{noise_type.upper()} comprehensive results saved to {results_filepath}")

    except Exception as e:
        print(f"\nError saving {noise_type} results JSON: {e}")
        import traceback; traceback.print_exc()
    # --- End Saving ---

    # --- Generate Final Summary Table ---
    print(f"\n\n=== Final Test Accuracy (%) - {noise_type.upper()} ===")
    summary_data = {}
    # Extract final test accuracy for each loss and noise rate
    for loss_name in loss_functions_to_run.keys():
        row_data = {}
        for noise_rate_str in results: # Iterate through string keys ('0.0', '0.2'...)
             # Safely get result, defaulting to NaN if run failed or metric missing
             acc = results[noise_rate_str].get(loss_name, {}).get('final_test_acc', np.nan) \
                   if results[noise_rate_str].get(loss_name) is not None else np.nan
             row_data[float(noise_rate_str)] = acc # Use float for column sorting
        summary_data[loss_name] = row_data

    try:
        # Create pandas DataFrame
        df = pd.DataFrame.from_dict(summary_data, orient='index')
        # Ensure columns are sorted numerically by noise rate
        df = df.reindex(columns=sorted(noise_rates_to_run), fill_value=np.nan)
        df.index.name = 'Loss Function' # Label the index
        # Print formatted table to console
        print(df.round(2).to_string())
        # Save table to CSV file
        summary_csv_path = run_output_dir / 'summary_final_test_accuracy.csv'
        df.round(2).to_csv(summary_csv_path)
        print(f"Summary table saved to {summary_csv_path}")
    except Exception as e:
        print(f"Error creating or saving {noise_type} summary table: {e}")
    # --- End Summary Table ---

    # --- Invoke External Plotting Function ---
    if PLOTTING_AVAILABLE:
        try:
            plot_dir = run_output_dir / 'plots' # Define specific plots subdirectory
            # Rework loss_functions_to_run slightly to match plotting util expectations (needs metadata, not loss_fn object)
            plotting_metadata = {}
            for name, config in loss_functions_to_run.items():
                plotting_metadata[name] = {k:v for k,v in config.items() if k != 'loss_fn'}

            # Call the function, passing necessary data
            plot_comprehensive_results(
                results=results, # The main results dictionary
                loss_functions_metadata=plotting_metadata, # Pass metadata without function objects
                noise_rates=noise_rates_to_run, # List of noise rates
                output_dir=plot_dir, # Directory to save plots
                symmetric=symmetric, # Noise type flag
                experiment_name=f"CIFAR10_{noise_type}" # Experiment name for titles etc.
            )
            print(f"\n{noise_type.upper()} plots saved to {plot_dir}")
        except Exception as e:
            # Catch errors during plotting
            print(f"\nError during {noise_type} plot generation: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"\nSkipping plot generation for {noise_type.upper()} as plotting utility is unavailable.")
    # --- End Plotting ---

    print(f"\n--- COMPLETED EXPERIMENT SET: {noise_type.upper()} ---")