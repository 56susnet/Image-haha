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
import random
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

# --- PHASE 6: Standard Global Physics (Pro Optimized) ---

# STYLE Anchors (Pro Source)
# UNet 5e-5, TE 5e-6, min_snr_gamma=6, prior_loss_weight=0.612, max_grad_norm=1.314
STANDARD_MULTIPLIERS_STYLE = {
    "under_10": {"unet": 4.0, "te": 2.0}, # -> 2e-4 / 1e-5
    "under_20": {"unet": 2.0, "te": 1.6}, # -> 1e-4 / 8e-6
    "under_30": {"unet": 1.6, "te": 1.0}, # -> 8e-5 / 5e-6
    "above_30": {"unet": 1.0, "te": 1.0}  # -> 5e-5 / 5e-6
}

# PERSON Anchors (Pro Source)
# Prodigy base d_coef=1.0, prior_loss_weight=0.7, min_snr_gamma=6
STANDARD_PHYSICS_PERSON = {
    "under_10": {"d_coef": 2.4, "te_ratio": 1.5},
    "under_20": {"d_coef": 1.8, "te_ratio": 1.3},
    "under_30": {"d_coef": 1.4, "te_ratio": 1.2},
    "above_30": {"d_coef": 1.0, "te_ratio": 1.0}
}

def get_jittered_value(base_val, multiplier, jitter_range=0.1):
    """Apply multiplier and a small random jitter (+/- 10% default)."""
    target = base_val * multiplier
    jitter = random.uniform(1 - jitter_range, 1 + jitter_range)
    return target * jitter
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

# load_size_based_config removed - Consolidation to unified JSON architecture.

def get_dataset_size_category(dataset_size: int) -> str:
    """Map dataset size to category labels used in LRS config."""
    if dataset_size <= 10:
        cat = "under_10"
    elif dataset_size <= 20:
        cat = "under_20"
    elif dataset_size <= 30:
        cat = "under_30"
    else:
        cat = "above_30"
    
    print(f"DEBUG_LRS: Image count {dataset_size} mapped to category -> [{cat.upper()}]", flush=True)
    return cat

def get_config_for_model(lrs_config: dict, model_hash: str, dataset_size: int = None, raw_model_name: str = None) -> dict:
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})
    
    target_config = None

    # Sanitize input name if provided
    clean_name = raw_model_name.strip().strip("'").strip('"') if raw_model_name else None

    # 1. Try Hash Lookup
    if isinstance(data, dict):
        if model_hash in data:
            target_config = data.get(model_hash)
            print(f"DEBUG_LRS: MATCH [HASH] -> {model_hash}", flush=True)
            
        # 2. Try Raw Name Lookup (Fallback)
        elif clean_name:
             # Direct lookup
             if clean_name in data:
                 target_config = data.get(clean_name)
                 print(f"DEBUG_LRS: MATCH [DIRECT KEY] -> {clean_name}", flush=True)
             else:
                 # Iterative lookup (scan 'model_name' field)
                 for key, val in data.items():
                     if isinstance(val, dict) and val.get("model_name") == clean_name:
                         target_config = val
                         print(f"DEBUG_LRS: MATCH [FIELD SCAN] -> {clean_name} (Key: {key})", flush=True)
                         break
        
        if not target_config and clean_name:
             print(f"DEBUG_LRS: FAIL lookup for '{clean_name}'. Hash was '{model_hash}'", flush=True)

    if target_config:
        # If dataset_size provided and model_config has size categories, merge them
        if dataset_size is not None and isinstance(target_config, dict):
            size_category = get_dataset_size_category(dataset_size)
            
            # Check if model_config has size-specific settings
            if size_category in target_config:
                size_specific_config = target_config.get(size_category, {})
                # Merge Config
                base_model_config = {k: v for k, v in target_config.items() if k not in ["under_10", "under_20", "under_30", "above_30"]}
                merged = merge_model_config(default_config, base_model_config)
                print(f"DEBUG_LRS: Merged Size Config ({size_category})", flush=True)
                return merge_model_config(merged, size_specific_config)
        
        return merge_model_config(default_config, target_config)

    if default_config:
        print("DEBUG_LRS: Using Default Config", flush=True)
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
            228: {"network_dim": 32, "network_alpha": 32, "network_args": ["conv_dim=8", "conv_alpha=8", "algo=locon"]},
            235: {"network_dim": 32, "network_alpha": 32, "network_args": ["conv_dim=8", "conv_alpha=8", "algo=locon"]},
            456: {"network_dim": 64, "network_alpha": 64, "network_args": ["conv_dim=16", "conv_alpha=16", "algo=locon"]},
            467: {"network_dim": 64, "network_alpha": 64, "network_args": ["conv_dim=16", "conv_alpha=16", "algo=locon"]},
            699: {"network_dim": 96, "network_alpha": 96, "network_args": ["conv_dim=32", "conv_alpha=32", "algo=locon"]},
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

        size_category = get_dataset_size_category(dataset_size)
        
        if is_style:
            # STYLE: Pro Adaptive Physics (AdamW)
            anchor_unet = 5e-5
            anchor_te = 5e-6
            mults = STANDARD_MULTIPLIERS_STYLE.get(size_category)
            
            config["min_snr_gamma"] = 6
            config["prior_loss_weight"] = 0.612
            config["max_grad_norm"] = 1.314
            config["noise_offset"] = 0.0411
            config["seed"] = 2951032221
            
            print(f"Applying Pro STYLE Physics (AdamW) for category [{size_category.upper()}]", flush=True)
            if mults:
                config["unet_lr"] = get_jittered_value(anchor_unet, mults["unet"])
                config["text_encoder_lr"] = get_jittered_value(anchor_te, mults["te"])
                print(f"  [Physics] Set UNet LR: {config['unet_lr']:.2e}, TE LR: {config['text_encoder_lr']:.2e}", flush=True)
        else:
            # PERSON: Pro Adaptive Physics (Prodigy)
            anchor_d_coef = 1.0
            physics = STANDARD_PHYSICS_PERSON.get(size_category)
            
            config["min_snr_gamma"] = 6
            config["prior_loss_weight"] = 0.7
            
            print(f"Applying Pro PERSON Physics (Prodigy) for category [{size_category.upper()}]", flush=True)
            if physics:
                config["unet_lr"] = 1.0
                config["text_encoder_lr"] = physics["te_ratio"]
                final_d_coef = get_jittered_value(anchor_d_coef, physics["d_coef"])
                
                new_args = []
                # Pro optimizer args baseline
                pro_base_args = {
                    "decouple": "True",
                    "weight_decay": "0.005",
                    "use_bias_correction": "True",
                    "safeguard_warmup": "True"
                }
                
                # Rebuild optimizer_args with d_coef and pro baselines
                new_args.append(f"d_coef={final_d_coef:.2f}")
                for k, v in pro_base_args.items():
                    new_args.append(f"{k}={v}")
                
                config["optimizer_args"] = new_args
                print(f"  [Physics] Set d_coef: {final_d_coef:.2f}, TE Ratio: {config['text_encoder_lr']}", flush=True)

        # --- PHASE 7: Pro Fine-Tuning Parameters ---
        # "Hacking" the loss for better numerical results
        config["loss_type"] = "huber"
        config["huber_schedule"] = "snr" # Robust loss weighting
        config["huber_c"] = 0.1         # Standard Huber threshold
        # config["gradient_accumulation_steps"] = 1 # Already default but key for stable loss
        print(f"  [Pro] Physics Enabled: Huber Loss (SNR Schedule)", flush=True)
            
        # --- TIERED JSON RESOLUTION (Champion Tier) ---
        # Note: dataset_size already counted above in Phase 6.

        lrs_settings = None
        lrs_config = load_lrs_config(model_type, is_style)
        if lrs_config:
            # Sanitize model name for robust hashing
            clean_model_name = model_name.strip().strip("'").strip('"')
            model_hash = hash_model(clean_model_name)
            lrs_settings = get_config_for_model(lrs_config, model_hash, dataset_size, clean_model_name)

        # Apply overrides with Adaptive Multiplier logic
        def apply_overrides(name, overrides):
            for key, value in overrides.items():
                if key in ["under_10", "under_20", "under_30", "above_30"]:
                    continue
                
                if isinstance(value, str) and value.startswith("*"):
                    try:
                        multiplier = float(value[1:])
                        if key in config:
                            base_value = config[key]
                            if isinstance(base_value, (int, float)):
                                new_value = base_value * multiplier
                                if key in ["max_train_epochs", "save_every_n_epochs", "network_dim", "network_alpha", "lr_warmup_steps", "train_batch_size"]:
                                    config[key] = int(new_value)
                                else:
                                    config[key] = new_value
                                print(f"  [Adaptive] Scaled {key}: {base_value} -> {config[key]} (x{multiplier}) from {name}", flush=True)
                            else:
                                print(f"  [Adaptive] Warning: Cannot multiply {key}. Overwriting with {value}", flush=True)
                                config[key] = value
                        else:
                            print(f"  [Adaptive] Warning: {key} not in base config. Ignoring.", flush=True)
                    except ValueError:
                        config[key] = value
                else:
                    config[key] = value

        if lrs_settings:
            print(f"Applying Champion LRS overrides for {model_name}", flush=True)
            apply_overrides("Champion LRS", lrs_settings)
        
        # --- PHASE 4: Automated Optimization Overrides ---
        if optimization_overrides:
            print(f"Applying Optuna overrides: {optimization_overrides}", flush=True)
            apply_overrides("Optuna", optimization_overrides)

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
