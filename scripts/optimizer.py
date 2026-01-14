#!/usr/bin/env python3
import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
import subprocess
import re
import sys
import os
import time
import argparse
import random
import signal
import functools

# --- CONFIGURATION ---
PYTHON_CMD = "python3"
SCRIPT_PATH = "/app/image-aa/scripts/image_trainer.py" # Adjust if running locally vs docker
# For local testing, we might assume relative path
if not os.path.exists(SCRIPT_PATH):
    SCRIPT_PATH = "scripts/image_trainer.py" # Fallback for local run

# Regex to capture loss
# Log format: "steps:  21%|██▏       | 77/360 [05:52<21:35,  4.58s/it, Average key norm=0.0255, Keys Scaled=0, avr_loss=0.14]"
LOSS_REGEX = re.compile(r"avr_loss=([0-9\.]+)")
STEP_REGEX = re.compile(r"steps:\s+(\d+)%\|") # Capture percentage or raw step?
# Actually raw step is better: "77/360" -> match group

def objective(trial, model, dataset_zip):
    # 1. Suggest Parameters
    gamma = trial.suggest_float("gamma", 0.5, 10.0)
    prior = trial.suggest_float("prior", 0.1, 2.0)
    norm = trial.suggest_float("norm", 0.0, 100.0)
    d_coef = trial.suggest_float("d_coef", 0.5, 2.0)
    epochs = trial.suggest_int("epochs", 20, 50)
    
    # Generate a unique Task ID for this trial
    task_id = f"optuna_trial_{trial.number}_{int(time.time())}"
    
    # 2. Construct Command
    # Model and dataset are passed as arguments from main
    cmd = [
        PYTHON_CMD, SCRIPT_PATH,
        "--task-id", task_id,
        "--model", model,
        "--dataset-zip", dataset_zip,
        "--model-type", "sdxl",
        "--expected-repo-name", f"optuna_{trial.number}",
        "--hours-to-complete", "1",
        # Optimization Overrides
        "--opt-gamma", str(gamma),
        "--opt-prior", str(prior),
        "--opt-norm", str(norm),
        "--opt-d-coef", str(d_coef),
        "--opt-epochs", str(epochs)
    ]
    
    print(f"\n[Trial {trial.number}] Starting with: Gamma={gamma:.2f}, Prior={prior:.2f}, Norm={norm:.2f}, DCoef={d_coef:.2f}, Epochs={epochs}")
    
    # 3. Run Process & Monitor
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    final_loss = float('inf')
    
    try:
        current_step = 0
        for line in process.stdout:
            print(line, end="") # Verbose output enabled for debugging
            
            # Parse Loss
            loss_match = LOSS_REGEX.search(line)
            if loss_match:
                loss = float(loss_match.group(1))
                final_loss = loss
                
                # Report to Optuna
                # We report every time we find a loss update
                current_step += 1
                trial.report(loss, current_step)
                
                # Check for Pruning
                if trial.should_prune():
                    print(f"[Trial {trial.number}] Pruned at step {current_step} with loss {loss}")
                    raise optuna.exceptions.TrialPruned()
                    
    except optuna.exceptions.TrialPruned:
        process.kill()
        raise
    except Exception as e:
        print(f"[Trial {trial.number}] Error: {e}")
        process.kill()
        raise
    finally:
        if process.poll() is None:
            process.kill()
            
    return final_loss

if __name__ == "__main__":
    # Create Study
    # MedianPruner: Prunes if the trial's best intermediate result is worse than 
    # the median of the intermediate results of previous trials at the same step.
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)
    
    # Parse CLI Args FIRST to pass to objective
    parser = argparse.ArgumentParser(description="Image AA Optimizer")
    parser.add_argument("--model", type=str, default="cagliostrolab/animagine-xl-4.0", help="Model ID to optimize")
    parser.add_argument("--dataset-zip", type=str, required=True, help="URL of the dataset zip")
    args = parser.parse_args()

    study = optuna.create_study(
        direction="minimize", 
        pruner=pruner,
        study_name="sdxl_optimization"
    )
    
    print("--- STARTING AUTOMATED OPTIMIZATION ---")
    print(f"Target Model: {args.model}")
    print(f"Target Dataset: {args.dataset_zip}")
    print("Optimization target: Minimize `avr_loss`")
    print("Pruner: MedianPruner (Aggressive early stopping)")
    
    # Wrap objective with partial to pass args
    optimization_func = functools.partial(objective, model=args.model, dataset_zip=args.dataset_zip)

    start_time = time.time()
    try:
        study.optimize(optimization_func, n_trials=50, timeout=3600*6) # 6 Hours or 50 trials
    except KeyboardInterrupt:
        print("\nOptimization stopped by user.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
        
    print("\n--- OPTIMIZATION RESULTS ---")
    print(f"Total Execution Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Loss: {study.best_value}")
    print("Best Params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
        
    # Save best results
    with open("best_params.json", "w") as f:
        import json
        json.dump(study.best_trial.params, f, indent=4)
