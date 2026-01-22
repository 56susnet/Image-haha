#!/usr/bin/env python3

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time
import yaml
import toml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# --- EMERGENCY TEMP DIR FIX ---
# Ensure local tmp directory exists and is used by libraries (torch, transformers, etc.)
LOCAL_TMP = os.path.join(project_root, "tmp")
os.makedirs(LOCAL_TMP, exist_ok=True)
os.environ["TMPDIR"] = LOCAL_TMP
os.environ["TEMP"] = LOCAL_TMP
os.environ["TMP"] = LOCAL_TMP
# ------------------------------

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType
from trainer.utils.trainer_downloader import download_image_dataset

# --- Model Categorization (Kasta) ---
# --- Model Categorization (Kasta) ---
# Phase 5: Mass Model Governance
# We define "Generals" (Proxies) and "Soldiers" (Target Models).

MODELS_ANIME = [
    # General: cagliostrolab/animagine-xl-4.0
    "cagliostrolab/animagine-xl-4.0",
    "zenless-lab/sdxl-aam-xl-anime-mix",
    "John6666/nova-anime-xl-pony-v5-sdxl",
    "zenless-lab/sdxl-anima-pencil-xl-v5",
    "recoilme/colorfulxl",
    "zenless-lab/sdxl-anything-xl",
    "stablediffusionapi/protovision-xl-v6.6", # Often stylistic
    "OnomaAIResearch/Illustrious-xl-early-release-v0",
    "John6666/hassaku-xl-illustrious-v10style-sdxl",
    "KBlueLeaf/Kohaku-XL-Zeta",
    "zenless-lab/sdxl-blue-pencil-xl-v7",
    "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16",
    "mhnakif/fluxunchained-dev", # Anime style flux
    "dataautogpt3/FLUX-MonochromeManga"
]

MODELS_REALISTIC = [
    # General: dataautogpt3/CALAMITY
    "dataautogpt3/CALAMITY",
    "misri/leosamsHelloworldXL_helloworldXL70",
    "GraydientPlatformAPI/albedobase2-xl",
    "dataautogpt3/ProteusV0.5",
    "dataautogpt3/ProteusSigma",
    "fluently/Fluently-XL-Final",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "openart-custom/DynaVisionXL",
    "Lykon/dreamshaper-xl-1-0",
    "mann-e/Mann-E_Dreams",
    "Corcelio/mobius",
    "femboysLover/RealisticStockPhoto-fp16",
    "ehristoforu/Visionix-alpha",
    "ifmain/UltraReal_Fine-Tune",
    "Lykon/art-diffusion-xl-0.9",
    "stablediffusionapi/omnium-sdxl",
    "misri/zavychromaxl_v90",
    "dataautogpt3/TempestV0.1",
    "GraydientPlatformAPI/realism-engine2-xl",
    "SG161222/RealVisXL_V4.0",
    "bghira/terminus-xl-velocity-v2",
    "rayonlabs/FLUX.1-dev", # Realistic Flux
    "mikeyandfriends/PixelWave_FLUX.1-dev_03" 
]

# Anything else falls to "General" (Treated as Realistic Safety First)


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path

# --- Phase 5: Mass Model Governance ---
# We define "Generals" (Proxies) and "Soldiers" (Target Models).

MODELS_ANIME = [
    # General: cagliostrolab/animagine-xl-4.0
    "cagliostrolab/animagine-xl-4.0",
    "zenless-lab/sdxl-aam-xl-anime-mix",
    "John6666/nova-anime-xl-pony-v5-sdxl",
    "zenless-lab/sdxl-anima-pencil-xl-v5",
    "recoilme/colorfulxl",
    "zenless-lab/sdxl-anything-xl",
    "stablediffusionapi/protovision-xl-v6.6", 
    "OnomaAIResearch/Illustrious-xl-early-release-v0",
    "John6666/hassaku-xl-illustrious-v10style-sdxl",
    "KBlueLeaf/Kohaku-XL-Zeta",
    "zenless-lab/sdxl-blue-pencil-xl-v7",
    "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16",
    "mhnakif/fluxunchained-dev", 
    "dataautogpt3/FLUX-MonochromeManga"
]

MODELS_REALISTIC = [
    # General: dataautogpt3/CALAMITY
    "dataautogpt3/CALAMITY",
    "misri/leosamsHelloworldXL_helloworldXL70",
    "GraydientPlatformAPI/albedobase2-xl",
    "dataautogpt3/ProteusV0.5",
    "dataautogpt3/ProteusSigma",
    "fluently/Fluently-XL-Final",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "openart-custom/DynaVisionXL",
    "Lykon/dreamshaper-xl-1-0",
    "mann-e/Mann-E_Dreams",
    "Corcelio/mobius",
    "femboysLover/RealisticStockPhoto-fp16",
    "ehristoforu/Visionix-alpha",
    "ifmain/UltraReal_Fine-Tune",
    "Lykon/art-diffusion-xl-0.9",
    "stablediffusionapi/omnium-sdxl",
    "misri/zavychromaxl_v90",
    "dataautogpt3/TempestV0.1",
    "GraydientPlatformAPI/realism-engine2-xl",
    "SG161222/RealVisXL_V4.0",
    "bghira/terminus-xl-velocity-v2",
    "rayonlabs/FLUX.1-dev", 
    "mikeyandfriends/PixelWave_FLUX.1-dev_03" 
]
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

    return merged if merged else None

def count_images_in_directory(directory_path: str) -> int:
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    count = 0
    
    try:
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}", flush=True)
            return 0
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.startswith('.'):
                    continue
                
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    count += 1
    except Exception as e:
        print(f"Error counting images in directory: {e}", flush=True)
        return 0
    
    return count

def load_size_based_config(model_type: str, is_style: bool, dataset_size: int) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")
    
    if model_type == "flux":
        return None
    elif is_style:
        config_file = os.path.join(config_dir, "size_style.json")
    else:
        config_file = os.path.join(config_dir, "size_person.json")
    
    try:
        with open(config_file, 'r') as f:
            size_config = json.load(f)
        
        size_ranges = size_config.get("size_ranges", [])
        for size_range in size_ranges:
            min_size = size_range.get("min", 0)
            max_size = size_range.get("max", float('inf'))
            
            if min_size <= dataset_size <= max_size:
                print(f"Using size-based config for {dataset_size} images (range: {min_size}-{max_size})", flush=True)
                return size_range.get("config", {})
        
        default_config = size_config.get("default", {})
        if default_config:
            print(f"Using default size-based config for {dataset_size} images", flush=True)
        return default_config
        
    except Exception as e:
        print(f"Warning: Could not load size-based config from {config_file}: {e}", flush=True)
        return None

def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None, optimization_overrides: dict | None = None):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        return config_path
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        lrs_config = load_lrs_config(model_type, is_style)

        if lrs_config:
            model_hash = hash_model(model_name)
            lrs_settings = get_config_for_model(lrs_config, model_hash)

            if lrs_settings:
                for optional_key in [
                    "max_grad_norm",
                    "prior_loss_weight",
                    "max_train_epochs",
                    "train_batch_size",
                    "max_train_steps",
                    "network_dim",
                    "network_alpha",
                    "optimizer_args",
                    "unet_lr",
                    "text_encoder_lr",
                    "lr_warmup_steps",
                    "network_dropout",
                    "min_snr_gamma",
                    "seed",
                    "noise_offset",
                    "lr_scheduler",
                    "save_every_n_epochs",
                    "scale_weight_norms",
                    "loss_type",
                ]:
                    if optional_key in lrs_settings:
                        config[optional_key] = lrs_settings[optional_key]
            else:
                print(f"Warning: No LRS configuration found for model '{model_name}'", flush=True)
        else:
            print("Warning: Could not load LRS configuration, using default values", flush=True)

        # --- Phase 5: Mass Model Governance (Kasta System) ---
        # Apply "Class Profile" physics if model is in a known kasta.
        # This runs BEFORE Size Config, so Size Config's "Physics Transparency" allows this to shine through.
        
        if model_name in MODELS_ANIME:
            print(f"👻 Model '{model_name}' detected as ANIME Class (General: Animagine). Applying Anime Physics.", flush=True)
            # ANIME PHYSICS (Refined for Illustrious-Class Robustness)
            config["min_snr_gamma"] = 1.0
            config["prior_loss_weight"] = 0.50
            config["scale_weight_norms"] = 10.0
            config["optimizer_args"] = [
                "decouple=True", "d_coef=1.5", "weight_decay=0.01", "use_bias_correction=True", "safeguard_warmup=True"
            ]

        elif model_name in MODELS_REALISTIC:
            print(f"📸 Model '{model_name}' detected as REALISTIC Class (General: Calamity). Applying Realistic Physics.", flush=True)
            # REALISTIC PHYSICS (The Disciplined Strategy)
            config["min_snr_gamma"] = 5.0 
            config["prior_loss_weight"] = 0.80
            config["scale_weight_norms"] = 1.0 
            config["optimizer_args"] = [
                "decouple=True", "d_coef=1.0", "weight_decay=0.005", "use_bias_correction=True", "safeguard_warmup=True"
            ]
            
        # --- End Phase 5 ---
        
        # --- PHASE 4: Automated Optimization Overrides ---
        if optimization_overrides:
            print(f"Applying Optuna overrides: {optimization_overrides}", flush=True)
            for key, value in optimization_overrides.items():
                if value is not None:
                    # Special handling for optimizer_args to avoid lobotomizing the optimizer
                    if key == "optimizer_args" and isinstance(value, list) and len(value) > 0:
                        # Extract d_coef if it's there
                        d_coef_val = None
                        for item in value:
                            if item.startswith("d_coef="):
                                d_coef_val = item
                                break
                        
                        if d_coef_val:
                            # Update existing d_coef or append it
                            existing_args = config.get("optimizer_args", [])
                            new_args = []
                            found = False
                            for arg in existing_args:
                                if arg.startswith("d_coef="):
                                    new_args.append(d_coef_val)
                                    found = True
                                else:
                                    new_args.append(arg)
                            if not found:
                                new_args.append(d_coef_val)
                            config["optimizer_args"] = new_args
                    else:
                        config[key] = value
                        if key in ["max_train_epochs", "save_every_n_epochs", "network_dim", "network_alpha", "train_batch_size"]:
                             config[key] = int(value) 
                    print(f"  [Optuna] Applied/Merged {key}", flush=True)
        # System now relies purely on LRS and Size-based scaling.

        network_config_person = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 467,
            "stablediffusionapi/protovision-xl-v6.6": 467,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 467,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 699,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 467,
            "dataautogpt3/TempestV0.1": 456,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 467,
            "fluently/Fluently-XL-Final": 228,
            "mann-e/Mann-E_Dreams": 456,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 228,
            "recoilme/colorfulxl": 228,
            "zenless-lab/sdxl-aam-xl-anime-mix": 456,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
            "zenless-lab/sdxl-anything-xl": 228,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
            "Corcelio/mobius": 228,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 467,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 228
        }

        network_config_style = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 235,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 235,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 235,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 235,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 235,
            "dataautogpt3/TempestV0.1": 228,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 235,
            "fluently/Fluently-XL-Final": 235,
            "mann-e/Mann-E_Dreams": 235,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 235,
            "recoilme/colorfulxl": 235,
            "zenless-lab/sdxl-aam-xl-anime-mix": 235,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 235,
            "zenless-lab/sdxl-anything-xl": 235,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
            "Corcelio/mobius": 235,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 235
        }

        config_mapping = {
            228: {
                "network_dim": 32,
                "network_alpha": 32,
                "network_args": []
            },
            235: {
                "network_dim": 32,
                "network_alpha": 32,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            456: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": []
            },
            467: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            699: {
                "network_dim": 96,
                "network_alpha": 96,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
        }

        config["pretrained_model_name_or_path"] = model_path
        config["train_data_dir"] = train_data_dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        if model_type == "sdxl":
            if is_style:
                network_config = config_mapping[network_config_style[model_name]]
            else:
                network_config = config_mapping[network_config_person[model_name]]

            config["network_dim"] = network_config["network_dim"]
            config["network_alpha"] = network_config["network_alpha"]
            config["network_args"] = network_config["network_args"]


        dataset_size = 0
        if os.path.exists(train_data_dir):
            dataset_size = count_images_in_directory(train_data_dir)
            if dataset_size > 0:
                print(f"Counted {dataset_size} images in training directory", flush=True)

        if dataset_size > 0:
            size_config = load_size_based_config(model_type, is_style, dataset_size)
            if size_config:
                print(f"Applying size-based tactical zone for {dataset_size} images", flush=True)
                for key, value in size_config.items():
                    # Size-based config (Sniper/Turbo) has high priority
                    # Implement Adaptive Multiplier Logic
                    if isinstance(value, str) and value.startswith("*"):
                        try:
                            multiplier = float(value[1:])
                            if key in config:
                                base_value = config[key]
                                if isinstance(base_value, (int, float)):
                                    new_value = base_value * multiplier
                                    # Ensure logical types (int for epochs/dim, float for others if needed)
                                    if key in ["max_train_epochs", "save_every_n_epochs", "network_dim", "network_alpha", "lr_warmup_steps", "train_batch_size"]:
                                        config[key] = int(new_value)
                                    else:
                                        config[key] = new_value
                                    print(f"  [Adaptive] Scaled {key}: {base_value} -> {config[key]} (x{multiplier})", flush=True)
                                else:
                                    print(f"  [Adaptive] Warning: Cannot multiply {key} (base value not number). Overwriting instead.", flush=True)
                                    config[key] = value
                            else:
                                print(f"  [Adaptive] Warning: {key} not in base config. Cannot multiply. Ignoring.", flush=True)
                        except ValueError:
                            print(f"  [Adaptive] Error parsing multiplier {value} for {key}. Overwriting instead.", flush=True)
                            config[key] = value
                    else:
                        # Standard Override
                        config[key] = value
        
        # --- PHASE 4: Automated Optimization Overrides ---
        if optimization_overrides:
            print(f"Applying Optuna overrides: {optimization_overrides}", flush=True)
            for key, value in optimization_overrides.items():
                if value is not None:
                    config[key] = value
                    if key in ["max_train_epochs", "save_every_n_epochs", "network_dim", "network_alpha", "train_batch_size"]:
                         config[key] = int(value) # Ensure appropriate types
                    print(f"  [Optuna] Override {key} = {config[key]}", flush=True)

        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"config is {config}", flush=True)
        print(f"Created config at {config_path}", flush=True)
        return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        training_command = [
            "python3",
            os.path.join(project_root, "scripts", "ai-toolkit", "run.py"),
            config_path
        ]
    else:
        # Standardize folder to 'sd-script' as observed in repo
        sd_script_dir = os.path.join(project_root, "scripts", "sd-script")
        
        if model_type == "sdxl":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                os.path.join(sd_script_dir, f"{model_type}_train_network.py"),
                "--config_file", config_path
            ]
        elif model_type == "flux":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                os.path.join(sd_script_dir, f"{model_type}_train_network.py"),
                "--config_file", config_path
            ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    
    # Optuna Args
    parser.add_argument("--opt-gamma", type=float, help="Optuna Override: min_snr_gamma")
    parser.add_argument("--opt-prior", type=float, help="Optuna Override: prior_loss_weight")
    parser.add_argument("--opt-norm", type=float, help="Optuna Override: scale_weight_norms")
    parser.add_argument("--opt-d-coef", type=float, help="Optuna Override: d_coef")
    parser.add_argument("--opt-epochs", type=int, help="Optuna Override: max_train_epochs")
    parser.add_argument("--opt-lr", type=float, help="Optuna Override: unet_lr/text_encoder_lr")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    # --- PHASE 4: Automated Download ---
    # If file is missing, fetch it using the downloader logic
    print(f"Checking for dataset zip: {args.dataset_zip}", flush=True)
    await download_image_dataset(args.dataset_zip, args.task_id, train_cst.CACHE_DATASETS_DIR)

    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    optimization_overrides = {
        "min_snr_gamma": args.opt_gamma,
        "prior_loss_weight": args.opt_prior,
        "scale_weight_norms": args.opt_norm,
        "max_train_epochs": args.opt_epochs,
        "unet_lr": args.opt_lr,
        "text_encoder_lr": args.opt_lr,
        "optimizer_args": [f"d_coef={args.opt_d_coef}"] if args.opt_d_coef else None
    }

    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        trigger_word=args.trigger_word,
        optimization_overrides=optimization_overrides
    )

    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
