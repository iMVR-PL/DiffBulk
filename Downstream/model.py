import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=128, num_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x

class FeatureFusionNetWithPLIP(nn.Module):
    def __init__(self, out_dim=100, fusion_method='concat', c=0.5, c_learnable=False, dropout_p=0.1):
        super().__init__()
        self.c_learnable = c_learnable

        # Feature Fusion Blocks
        self.block1 = FeatureFusionBlock(512, 128, num_groups=32)
        self.block2 = FeatureFusionBlock(384, 128, num_groups=32)
        self.block3 = FeatureFusionBlock(256, 128, num_groups=32)
        self.block4 = FeatureFusionBlock(128, 128, num_groups=32)
        self.dropout = nn.Dropout(dropout_p)


        # Fusion method
        self.fusion_method = fusion_method

        # Initialize c
        if self.c_learnable:
            # c is a learnable parameter, initialized to the provided value
            self.c = nn.Parameter(torch.tensor(c))
        else:
            # c is a fixed value
            self.register_buffer('c', torch.tensor(c))

        # FiLM Parameters (for Feature-wise Linear Modulation)
        if self.fusion_method == 'film':
            self.film_gamma = nn.Linear(512, 512)  # Generate γ from diffusion features
            self.film_beta = nn.Linear(512, 512)   # Generate β from diffusion features

        # Gated Residual Fusion Parameters
        if self.fusion_method == 'gated_residual':
            self.gate = nn.Sequential(
                nn.Linear(512 + 512, 512),  # Concatenate fusion_output and plip_features
                nn.Sigmoid()                # Output gate value g ∈ [0, 1]
            )
            self.fusion_proj = nn.Linear(512, 512)

        # Final Combination Layer
        if fusion_method == 'concat':
            self.out = nn.Linear(512 + 512, out_dim)  # 128 + 512
        elif fusion_method == 'add':
            self.out = nn.Linear(512, out_dim)
        elif fusion_method == 'film':
            self.out = nn.Linear(512, out_dim)
        elif fusion_method == 'gated_residual':
            self.out = nn.Linear(512, out_dim)  # Output dimension matches num_genes
        else:
            raise ValueError("fusion_method must be 'concat', 'add', 'film', or 'gated_residual'")

    def forward(self, features, plip_features):
        f28, f56, f112, f224 = features

        # Feature Fusion
        f28_up = F.interpolate(f28, size=f56.shape[2:], mode='bilinear', align_corners=False)
        out1 = self.block1(f28)
        out1 = self.dropout(out1)

        out2 = self.block2(f56)
        out2 = self.dropout(out2)

        out3 = self.block3(f112)
        out3 = self.dropout(out3)

        out4 = self.block4(f224)
        out4 = self.dropout(out4)

        # Flatten the outputs to [B, C]
        out1 = out1.view(out1.size(0), -1)  # [B, 128]
        out2 = out2.view(out2.size(0), -1)  # [B, 128]
        out3 = out3.view(out3.size(0), -1)  # [B, 128]
        out4 = out4.view(out4.size(0), -1)  # [B, 128]

        # Pooling
        fusion_output = torch.cat([out1, out2, out3, out4], dim=1)
        # fusion_output = self.fusion_norm(fusion_output)  # [B, 512]

        # PLIP Features Processing
        plip_output = plip_features  # [B, 512]

        # Apply sigmoid to c if it is learnable to ensure it is between 0 and 1
        if isinstance(self.c, nn.Parameter):
            c = torch.sigmoid(self.c)
        else:
            c = self.c

        # Combine Features
        if self.fusion_method == 'concat':
            combined_output = torch.cat([plip_output, c * fusion_output], dim=1)
        elif self.fusion_method == 'add':
            combined_output = c * plip_output + (1 - c) * fusion_output
        elif self.fusion_method == 'film':
            # FiLM: Generate γ and β from diffusion features
            gamma = self.film_gamma(fusion_output)  # [B, 512]
            beta = self.film_beta(fusion_output)    # [B, 512]
            # Apply Feature-wise Linear Modulation to plip_output
            combined_output = (1. + gamma) * plip_output + beta  # [B, 512]
        elif self.fusion_method == 'gated_residual':
            # Gated Residual Fusion
            gate_input = torch.cat([plip_output, fusion_output], dim=-1)  # [B, 128 + 512]
            g = self.gate(gate_input)  # [B, 512]
            # Project g to [B, 512] to match plip_output
            fusion_output_proj = self.fusion_proj(fusion_output)  # [B, 128] -> [B, 512]
            # Dynamic fusion
            combined_output = plip_output + g * fusion_output_proj  # [B, 512]

        final_output = self.out(combined_output)

        return final_output
