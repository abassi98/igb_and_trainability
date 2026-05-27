"""Vision Transformer components used by the repository's custom ViT model."""

import math
import torch
from torch import nn
import numpy as np
from src.models import SuperModule

class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project them to token space."""
    def __init__(self, image_size, patch_size, num_channels, hidden_size, sigma_w, sigma_b):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        # init
        std =  sigma_w * 1.0 / math.sqrt(self.projection.kernel_size[0] * self.projection.kernel_size[1] * self.projection.in_channels)
        nn.init.normal_(self.projection.weight, mean=0.0, std=std)
        nn.init.normal_(self.projection.bias, mean=0.0, std=sigma_b)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        return x
 

class Embeddings(nn.Module):
    """Combine patch embeddings with a class token and positional embeddings."""
    def __init__(self, image_size, patch_size, num_channels, hidden_size, sigma_w, sigma_b):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_embeddings = PatchEmbeddings(image_size, patch_size, num_channels, hidden_size, sigma_w, sigma_b)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches+1, hidden_size)/np.sqrt(hidden_size)
        )

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = (x + self.position_embeddings)
        return x

class AttentionHead(nn.Module):
    """Single self-attention head with custom variance-scaled initialization."""
    def __init__(self, hidden_size, attention_head_size, bias, sigma_w, sigma_b, num_patches):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        # init
        std = sigma_w * np.sqrt(1.0/(math.sqrt(2)*self.query.in_features)) * np.log(num_patches)**(1/4)
        nn.init.normal_(self.query.weight, mean=0.0, std=std)
        nn.init.normal_(self.key.weight, mean=0.0, std=std)
        nn.init.normal_(self.value.weight, mean=0.0, std=std)   
        if bias: 
            nn.init.normal_(self.query.bias, mean=0.0, std=sigma_b)
            nn.init.normal_(self.key.bias, mean=0.0, std=sigma_b)
            nn.init.normal_(self.value.bias, mean=0.0, std=sigma_b)


    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, value)
        return attention_output, attention_probs


class MultiHeadAttention(nn.Module):
    """Reference multi-head attention built from explicit per-head modules."""
    def __init__(self, hidden_size, num_attention_heads, qkv_bias, sigma_w, sigma_b, num_patches):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = qkv_bias
        self.heads = nn.ModuleList([
            AttentionHead(hidden_size, self.attention_head_size, qkv_bias,sigma_w, sigma_b, num_patches)
            for _ in range(num_attention_heads)
        ])
        self.output_projection = nn.Linear(self.all_head_size, hidden_size)
        std = sigma_w * np.sqrt(1.0/self.output_projection.in_features)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=std)
        if qkv_bias:
            nn.init.normal_(self.output_projection.bias, mean=0.0, std=sigma_b) 
    
    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([out for out, _ in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        if not output_attentions:
            return attention_output, None
        attention_probs = torch.stack([p for _, p in attention_outputs], dim=1)
        return attention_output, attention_probs


class FasterMultiHeadAttention(nn.Module):
    """Vectorized multi-head attention using a joint QKV projection."""
    def __init__(self, hidden_size, num_attention_heads, qkv_bias, sigma_w, sigma_b, num_patches):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_projection = nn.Linear(hidden_size, self.all_head_size * 3, bias=qkv_bias)
        self.output_projection = nn.Linear(self.all_head_size, hidden_size)

        # init
        #print(np.log(num_patches)**(1/2), np.log(num_patches)**(1/4))
        std = sigma_w * np.sqrt(1.0/(self.qkv_projection.in_features)) #* np.log(num_patches)**(1/4)
        nn.init.normal_(self.qkv_projection.weight, mean=0.0, std=std)
        std = sigma_w * np.sqrt(1.0/self.output_projection.in_features)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=std)

        if qkv_bias:
            nn.init.normal_(self.qkv_projection.bias, mean=0.0, std=sigma_b)
            nn.init.normal_(self.output_projection.bias, mean=0.0, std=sigma_b)

    def forward(self, x, output_attentions=False):
        qkv = self.qkv_projection(x)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        batch_size, seq_len, _ = query.size()
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        attention_output = self.output_projection(attention_output)
        if not output_attentions:
            return attention_output, None
        return attention_output, attention_probs

class MLP(nn.Module):
    """Transformer feed-forward block."""
    def __init__(self, hidden_size, intermediate_size, act, sigma_w, sigma_b):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.act = act
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)

        # init
        std = sigma_w * np.sqrt(1.0/self.dense_1.in_features)
        nn.init.normal_(self.dense_1.weight, mean=0.0, std=std)
        nn.init.normal_(self.dense_1.bias, mean=0.0, std=sigma_b)
        std = sigma_w * np.sqrt(2.0/self.dense_2.in_features)
        nn.init.normal_(self.dense_2.weight, mean=0.0, std=std)
        nn.init.normal_(self.dense_2.bias, mean=0.0, std=sigma_b)   
       
    def forward(self, x):
        x = self.dense_1(x)
        x = self.act(x)
        x = self.dense_2(x)
        return x


class Block(nn.Module):
    """Transformer encoder block with optional residual connections."""
    def __init__(self, hidden_size, num_attention_heads, qkv_bias,
                 intermediate_size, use_faster_attention, act, residual, sigma_w, sigma_b, num_patches):
        super().__init__()
        if use_faster_attention:
            self.attention = FasterMultiHeadAttention(hidden_size, num_attention_heads, qkv_bias, sigma_w, sigma_b, num_patches)
        else:
            self.attention = MultiHeadAttention(hidden_size, num_attention_heads, qkv_bias, sigma_w, sigma_b, num_patches)
        
        #self.layernorm_1 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, act, sigma_w, sigma_b)
        self.residual = residual
        #self.layernorm_2 = nn.LayerNorm(hidden_size)

    def forward(self, x, output_attentions=False):
        attention_output, attention_probs = self.attention(x, output_attentions) # x instead of self.layernorm_1(x)s
        #x = attention_output
        #flat_x = torch.flatten(x, start_dim=1)
        #print(torch.mean(flat_x, dim=0), torch.std(flat_x, dim=0))
        if self.residual:
            x = x + attention_output
        else:
            x = attention_output
        mlp_output = self.mlp(x) # x instead of self.layernorm_1(x)

        if self.residual:
            x = x + mlp_output
        else:
            x = mlp_output

        if not output_attentions:
            return x, None
        return x, attention_probs


class Encoder(nn.Module):
    """Stack of transformer blocks."""
    def __init__(self, num_hidden_layers, hidden_size, num_attention_heads, qkv_bias,
                 intermediate_size, use_faster_attention, act, residual, sigma_w, sigma_b, num_patches):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_attention_heads, qkv_bias,
                  intermediate_size, use_faster_attention, act, residual, sigma_w, sigma_b, num_patches)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attn_probs = block(x, output_attentions)
            if output_attentions:
                all_attentions.append(attn_probs)
        if not output_attentions:
            return x, None
        return x, all_attentions


class VIT(SuperModule):
    """Custom Vision Transformer used in both initialization and training experiments."""
    def __init__(self, image_size, patch_size, num_channels, hidden_size, num_hidden_layers,
                 num_attention_heads, qkv_bias,intermediate_size, n_classes, sigma_w=2.0,
                 sigma_b=0.0, lr=0.001, use_faster_attention=True, act=nn.ReLU(), init="Gaussian", output_attentions=False, residual=False):
        super().__init__(sigma_w, sigma_b, lr, n_classes)
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.embedding = Embeddings(image_size, patch_size, num_channels, hidden_size, sigma_w, sigma_b)
        self.encoder = Encoder(num_hidden_layers, hidden_size, num_attention_heads, qkv_bias, intermediate_size, use_faster_attention, act, residual, sigma_w, sigma_b, self.embedding.patch_embeddings.num_patches)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.init = init
        self.output_attentions = output_attentions
        nn.init.normal_(self.classifier.weight, mean=0.0, std=sigma_w * np.sqrt(1.0/self.classifier.in_features))
        if self.classifier.bias is not None:
            nn.init.normal_(self.classifier.bias, mean=0.0, std=sigma_b)

        # save backward hooks
        self.save_forward_hook()
        self.save_backward_hook()


    def forward(self, x):
        #print(torch.mean(flat_x, dim=0), torch.std(flat_x, dim=0))
        x = self.embedding(x)
        #print(torch.mean(x, dim=0), torch.std(x, dim=0))
        x, all_attentions = self.encoder(x, self.output_attentions)
        logits = self.classifier(x[:, 0, :])
        if not self.output_attentions:
            return logits
        return logits, all_attentions
