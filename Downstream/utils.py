import pickle
import torch
import torch.nn.functional as F
import yaml
import os

from model import FeatureFusionNetWithPLIP

def load_model(config):
    import pickle
    from transformers import CLIPModel, CLIPProcessor

    # pretrained diffusion
    with open(config['diffusion_path'], 'rb') as file:
        data = pickle.load(file)
        diffusion = data['ema']

    # freeze diffusion
    for param in diffusion.parameters():
        param.requires_grad = False

    # pretrained Plip and image processor
    plip = CLIPModel.from_pretrained("vinid/plip")
    processor = CLIPProcessor.from_pretrained("vinid/plip")

    # freeze plip
    for param in plip.parameters():
        param.requires_grad = False

    # Feature Fusion Network with PLIP
    fusion_net = FeatureFusionNetWithPLIP(
        out_dim=config["out_dim"],
        fusion_method=config["fusion_method"],
        c=config["c"],  # Initial value of c
        c_learnable=config.get("c_learnable", False)  # Whether c is learnable (default to False if not specified)
    )

    return diffusion, processor, plip, fusion_net

def save_ckpt(fusion_net, optimizer, epoch, config):
    ckpt_dir = config["checkpoint_dir"]
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    filename = f"checkpoint_best.pth"
    checkpoint_path = os.path.join(ckpt_dir, filename)

    # Prepare checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "fusion_net_state_dict": fusion_net.state_dict(),  # Automatically includes self.c
        "optimizer_state_dict": optimizer.state_dict()
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}!")

def load_ckpt_model(config):
    """
    Load the diffusion model and FusionNet from a checkpoint file.
    """
    # Load models and freeze diffusion and plip
    diffusion, processor, plip, fusion_net = load_model(config)

    # Load checkpoint if specified
    fusion_net_ckpt_path = config.get("fusion_net_path", None)
    assert fusion_net_ckpt_path is not None, "FusionNet checkpoint path not provided in the config."

    # Load model weights
    fusion_net_ckpt = torch.load(fusion_net_ckpt_path, map_location=config["device"])
    fusion_net.load_state_dict(fusion_net_ckpt["fusion_net_state_dict"])

    # freeze fusion net
    for param in fusion_net.parameters():
        param.requires_grad = False

    print(f"FusionNet Checkpoint loaded from {fusion_net_ckpt_path}!")

    return diffusion, processor, plip, fusion_net

def extract_features(diffusion, x, noise_label, genes=None):
    with torch.no_grad():
        B = x.shape[0]
        device = x.device

        # U-net
        unet = diffusion.unet
        features = {}

        # sigma
        sigma = torch.tensor([noise_label] * B, device=device).view(B, 1, 1, 1)

        # Preconditioning weights
        sigma_data = diffusion.sigma_data
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # noise and gene conditions
        noise_emb = unet.emb_noise(unet.noise_emb_fourier(c_noise))
        if genes is None:
            emb = noise_emb
        else:
            genes = genes.to(device)
            gene_emb = unet.emb_gene(genes)
            emb = F.silu(0.5 * noise_emb + 0.5 * gene_emb)

        # Precond formula
        noise = torch.randn_like(x, device=device) * sigma
        x = x + noise

        # scale and bias
        x = (c_in * x).to(torch.float32)
        x = torch.cat([x, torch.ones_like(x[:, :1], device=device)], dim=1)

        # encoder and skip connect
        skips = []
        for name, block in unet.enc.items():
            if 'conv' in name:
                x = block(x)
            else:
                x = block(x, emb)
            skips.append(x)

        # decoder
        for name, block in unet.dec.items():
            if 'block' in name:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)

            if name == '28x28_block3':
                features['28x28_block3'] = x
            elif name == '56x56_block3':
                features['56x56_block3'] = x
            elif name == '112x112_block3':
                features['112x112_block3'] = x
            elif name == '224x224_block3':
                features['224x224_block3'] = x

        return features

if __name__ == "__main__":
    with open("./config.yaml", "r") as file:
        config = yaml.safe_load(file)

    diffusion, processor, plip, fusion_net = load_model(config)

    x = torch.randn(1, 3, 224, 224)
    noise_label = config["noise_label"]
    features = extract_features(diffusion, x, noise_label=noise_label, genes=None)

    predicted_genes = fusion_net([
        features['28x28_block3'],
        features['56x56_block3'],
        features['112x112_block3'],
        features['224x224_block3']
    ], plip_features=torch.randn(1, 512))

    print(f"Predicted genes shape: {predicted_genes.shape}")