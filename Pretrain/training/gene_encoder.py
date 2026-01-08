import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneProcessorBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.dropout(x)
        x += residual
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x += residual
        return x

class AttentionAggregator(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x):
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        attn_output, _ = self.attn(query, x, x)
        return attn_output.squeeze(1)

class GeneModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_gene_blocks=2, dropout=0.1):
        super().__init__()
        # Multi-Embeddings
        self.vocabs = [460, 541, 538]
        
        # Register the embeddings as regular parameters instead of directly assigning
        for num_genes in self.vocabs:
            self.register_parameter(
                f'embedding_{num_genes}', 
                nn.Parameter(torch.zeros(num_genes, embed_dim))
            )
        
        # Initialize embeddings consistently after parameter registration
        for num_genes in self.vocabs:
            embedding = getattr(self, f'embedding_{num_genes}')
            with torch.no_grad():
                # Use a deterministic initialization method
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                nn.init.xavier_uniform_(embedding)
        
        self.processor = nn.ModuleList([
            GeneProcessorBlock(embed_dim, num_heads, dropout)
            for _ in range(num_gene_blocks)
        ])
        self.aggregator = AttentionAggregator(embed_dim, num_heads)

    def forward(self, x):
        num_genes = x.size(1)
        if num_genes in self.vocabs:
            embedding = getattr(self, f'embedding_{num_genes}')
        else:
            raise ValueError(f"{num_genes} gene embeddings are not defined!")
        
        # value * embedding
        x = x.unsqueeze(-1) * embedding  # [B, num_genes, D]
        
        # Gene embedding block
        for block in self.processor:
            x = block(x)
            
        return self.aggregator(x)
        
# test
if __name__ == "__main__":
    # 460 genes
    genes_460 = torch.randint(0, 460, (10, 460)).float()
    model = GeneModel()
    output = model(genes_460)
    print("The shape of 460 gene embeddings:", output.shape)  # torch.Size([10, 256])
    
    # 541 genes
    genes_541 = torch.randint(0, 541, (10, 541)).float()
    output = model(genes_541)
    print("The shape of 541 gene embeddings:", output.shape)  # torch.Size([10, 256])
