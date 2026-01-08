import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from dataset import get_loader
from utils import load_model, save_ckpt, extract_features


def get_loss(diffusion, fusion_net, plip, diffusion_patches, plip_patches, noise_label, genes):
    criterion = nn.MSELoss()

    # Extract features from diffusion
    diffusion_features = extract_features(diffusion, x=diffusion_patches, noise_label=noise_label, genes=None)

    # Extract features from plip
    with torch.no_grad():
        plip_features = plip.get_image_features(plip_patches)

    # Compute predicted genes
    predicted_genes = fusion_net([
        diffusion_features['28x28_block3'],
        diffusion_features['56x56_block3'],
        diffusion_features['112x112_block3'],
        diffusion_features['224x224_block3'],
    ], plip_features)

    # Calculate loss
    loss = criterion(predicted_genes, genes)
    return loss

def valid(diffusion, fusion_net, processor, plip, valid_loader, config, writer=None, epoch=None):
    """
    Validate the model and compute multiple evaluation metrics:
    - MSE Loss
    - MAE Loss
    - Pearson Correlation
    """
    diffusion.eval()  
    fusion_net.eval()
    total_mse_loss = 0.0
    total_mae_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for patches, genes in valid_loader:
            # Preprocess patches
            plip_patches = processor(images=patches, return_tensors="pt")['pixel_values']
            diffusion_patches = patches / 127.5 - 1 

            # Move to specific device
            plip_patches = plip_patches.to(config["device"])
            diffusion_patches = diffusion_patches.to(config["device"])
            genes = genes.to(config["device"])

            # Get features
            plip_features = plip.get_image_features(plip_patches)
            diffusion_features = extract_features(diffusion, x=diffusion_patches, noise_label=config["noise_label"], genes=None)

            # Compute predicted genes
            predicted_genes = fusion_net([
                diffusion_features['28x28_block3'],
                diffusion_features['56x56_block3'],
                diffusion_features['112x112_block3'],
                diffusion_features['224x224_block3'],
            ], plip_features)

            # Compute MSE and MAE loss
            mse_loss = F.mse_loss(predicted_genes, genes).item()
            mae_loss = F.l1_loss(predicted_genes, genes).item()
            total_mse_loss += mse_loss
            total_mae_loss += mae_loss

            # Collect predictions and targets for Pearson Correlation
            all_preds.extend(predicted_genes.cpu().numpy())
            all_targets.extend(genes.cpu().numpy())

    # Compute average losses
    avg_mse_loss = total_mse_loss / len(valid_loader)
    avg_mae_loss = total_mae_loss / len(valid_loader)

    # Compute Pearson Correlation
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    pearson_corr, _ = pearsonr(all_preds, all_targets)

    # Log metrics to TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/valid_mse', avg_mse_loss, epoch)
        writer.add_scalar('Loss/valid_mae', avg_mae_loss, epoch)
        writer.add_scalar('Metric/valid_pearson', pearson_corr, epoch)

    diffusion.train()  
    fusion_net.train()
    return avg_mse_loss, avg_mae_loss, pearson_corr

def main(config):
    if not os.path.exists(config["tensorboard_dir"]):
        os.makedirs(config["tensorboard_dir"])

    # tensorboard logging
    writer = SummaryWriter(log_dir=config["tensorboard_dir"])

    # Load model, optimizer and dataloaders
    diffusion, processor, plip, fusion_net = load_model(config)
    # Move models to device
    plip = plip.to(config["device"])
    diffusion = diffusion.to(config["device"])
    fusion_net = fusion_net.to(config["device"])

    optimizer = optim.AdamW(
        fusion_net.parameters(), 
        lr=config["lr"], 
        weight_decay=config["weight_decay"],
        )

    # Define StepLR scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_loader, valid_loader = get_loader(config)

    # Initialize variables for tracking best model
    best_valid_pearson = -1.0  # Pearson ranges from -1 to 1
    best_epoch = -1
    best_valid_mse = float('inf')  
    best_valid_mae = float('inf')

    # train
    for epoch in range(config["epochs"]):
        fusion_net.train()
        epoch_loss = 0.0

        for i, (patches, genes) in enumerate(train_loader):
            # preprocess patches
            plip_patches = processor(images=patches, return_tensors="pt")['pixel_values']
            diffusion_patches = patches / 127.5 - 1 
            # move to specific device
            plip_patches = plip_patches.to(config["device"])
            diffusion_patches = diffusion_patches.to(config["device"])
            genes = genes.to(config["device"])
            
            # compute loss
            optimizer.zero_grad()
            loss = get_loss(diffusion, fusion_net, plip, diffusion_patches=diffusion_patches, plip_patches=plip_patches, noise_label=config["noise_label"], genes=genes)
            # backward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # verbose output
            if (i + 1) % config["log_interval"] == 0:
                print(f"Epoch [{epoch+1}/{config['epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Average Loss: {avg_epoch_loss:.4f}")
        # log to tensorboard
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)

        # Step the scheduler
        # scheduler.step()

        # validation
        valid_interval = config.get("valid_interval", 5)  # Add default value
        start_valid = config.get("start_valid", 5)  # Add default value
        if (epoch % valid_interval == 0 and epoch >= start_valid) or epoch == (config["epochs"] - 1):
            valid_mse, valid_mae, valid_pearson = valid(
                diffusion, fusion_net, processor, plip, valid_loader, config, writer, epoch
            )
            print(f"Epoch [{epoch+1}/{config['epochs']}], Validation MSE: {valid_mse:.4f}, MAE: {valid_mae:.4f}, Pearson: {valid_pearson:.4f}")

            # Save checkpoint if Pearson is highest and MSE is lowest
            # if valid_pearson > best_valid_pearson and valid_mae < best_valid_mae:
            if valid_pearson > best_valid_pearson:
                best_valid_pearson = valid_pearson
                best_epoch = epoch
            # if valid_mae < best_valid_mae:
            #     best_valid_mae = valid_mae
            #     best_epoch = epoch
                # Save checkpoint
                save_ckpt(fusion_net, optimizer, best_epoch, config)
                
    print(f"Training complete. Best model found at epoch {best_epoch+1} with Pearson: {best_valid_pearson:.4f}")
    writer.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the gene prediction model")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    # load config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # print config
    print("Loaded configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)

    main(config)
