import torch
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from utils import load_ckpt_model, extract_features
from dataset import get_test_loader
import argparse

def test_model(config):
    """Perform testing on patch_test and gene_test, calculate Pearson correlation, MAE and MSE."""
    # load test data
    test_loader = get_test_loader(config)

    # load checkpoint
    diffusion, processor, plip, fusion_net = load_ckpt_model(config)
    plip.eval()
    diffusion.eval()
    fusion_net.eval()
    
    # Move models to device
    plip = plip.to(config["device"])
    diffusion = diffusion.to(config["device"])
    fusion_net = fusion_net.to(config["device"])

    # Check if c is correctly restored
    if config.get("c_learnable", False):
        print(f"Restored c value: {fusion_net.c.item()}")
    else:
        print(f"Restored c value (fixed): {fusion_net.c.item()}")

    # record outputs
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for patches, genes in test_loader:
            # preprocess test data
            plip_patches = processor(images=patches, return_tensors="pt")["pixel_values"]
            diffusion_patches = patches / 127.5 - 1 
            
            # move to specific device
            plip_patches = plip_patches.to(config["device"])
            diffusion_patches = diffusion_patches.to(config["device"])
            genes = genes.to(config["device"])

            # get diffusion features and plip features
            diffusion_features = extract_features(diffusion, x=diffusion_patches, noise_label=config["noise_label"], genes=None)
            plip_features = plip.get_image_features(plip_patches)
            
            # Compute predicted genes
            predicted_genes = fusion_net([
                diffusion_features['28x28_block3'],
                diffusion_features['56x56_block3'],
                diffusion_features['112x112_block3'],
                diffusion_features['224x224_block3'],
            ], plip_features)

            # record predicted genes
            all_predictions.append(predicted_genes.cpu().numpy())
            all_ground_truths.append(genes.cpu().numpy())

    # convert to numpy arrays
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_ground_truths = np.concatenate(all_ground_truths, axis=0).flatten()

    # compute evaluation metrics
    pearson_corr, p_value = pearsonr(all_predictions, all_ground_truths)
    mae = np.mean(np.abs(all_predictions - all_ground_truths))
    mse = np.mean((all_predictions - all_ground_truths) ** 2)

    print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
    print(f"P-value: {p_value:.4e}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

    return mae, mse, pearson_corr, p_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Plip gene prediction model")
    parser.add_argument('--config_paths', nargs='+', required=True, help="List of paths to the config files")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to save the CSV results")
    args = parser.parse_args()

    results = []
    
    for config_path in args.config_paths:
        # Load config
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        print(f"Testing using configuration: {config_path}")
        mae, mse, pearson_corr, p_value = test_model(config)
        
        results.append({
            'Config': '+plip',
            'MAE': mae,
            'MSE': mse,
            'Pearson_Correlation': pearson_corr,
            'P_value': p_value
        })
    
    # Append results to CSV file
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, mode='a', header=not pd.io.common.file_exists(args.output_csv), index=False)
    print(f"Results appended to {args.output_csv}")
