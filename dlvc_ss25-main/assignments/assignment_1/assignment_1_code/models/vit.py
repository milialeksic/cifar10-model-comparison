import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self,in_channels = 3,patch_size = 16,emb_dim = 768, img_size =224):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size//patch_size)**2
        self.proj = nn.Conv2d(in_channels,emb_dim,kernel_size = patch_size,stride = patch_size)

    def forward(self,x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,emb_dim,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert emb_dim%num_heads == 0 

        self.qkv = nn.Linear(emb_dim,emb_dim*3)
        self.out = nn.Linear(emb_dim,emb_dim)

    def forward(self,x):
        B,N,D = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,1,3,4)
        q,k,v = qkv[0],qkv[1],qkv[2]

        scores = (q @ k.transpose(-2,-1))/math.sqrt(self.head_dim)
        attn = scores.softmax(dim = -1)
        out = (attn @ v).transpose(1,2).reshape(B,N,D)
        return self.out(out)
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self,emb_dim,num_heads,mlp_ratio,dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadSelfAttention(emb_dim,num_heads)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim,int(emb_dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(emb_dim*mlp_ratio),emb_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        x =x + self.dropout(self.attn(self.norm1(x)))
        x =x + self.dropout(self.mlp(self.norm2(x)))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self,img_size,patch_size,emb_dim,depth,num_heads,num_classes,mlp_ratio,dropout):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=3,patch_size=patch_size,emb_dim=emb_dim,img_size=img_size)
        self.cls_tokens = nn.Parameter(torch.zeros(1,1,emb_dim))
        self.n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1,1 + self.n_patches,emb_dim))

        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(emb_dim, num_heads,mlp_ratio=mlp_ratio,dropout=dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])