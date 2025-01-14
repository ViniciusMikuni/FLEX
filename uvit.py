import torch
import torch.nn as nn
import einops
import torch.utils.checkpoint
import numpy as np
import math
import sys
from common import NoScaleDropout, Base2FourierFeatures, timestep_embedding, MPFourier


def patchify(imgs, patch_size):
    """
    Divide images into patches and flatten them.

    :param imgs: Input images of shape [B, C, H, W]
    :param patch_size: The size of each patch
    :return: A tensor of shape [B, N_patches, patch_dim], where patch_dim = patch_size * patch_size * C
    """
    x = einops.rearrange(
        imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)',
        p1=patch_size, p2=patch_size
    )
    return x


def unpatchify(x, channels):
    """
    Reconstruct images from patches.

    :param x: Input tensor of shape [B, N_patches, patch_dim]
    :param channels: Number of channels in the output image
    :return: Reconstructed images of shape [B, C, H, W]
    """
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1], f"Invalid number of patches: expected {h * w}, got {x.shape[1]}"
    assert patch_size ** 2 * channels == x.shape[2], "Invalid dimensions for unpatchify"
    x = einops.rearrange(
        x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)',
        h=h, p1=patch_size, p2=patch_size
    )
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)

        qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, 'B H L D -> B L (H D)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Define layers
        self.fc1 = nn.Linear(in_features, hidden_features)
        #self.norm1 = norm_layer(hidden_features)  # LayerNorm after the first linear layer
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x):
        # Apply the first linear layer, activation, dropout, and norm
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        #x = self.norm1(x)

        # Apply the second linear layer, norm, and dropout
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., attn_drop=0.0, mlp_drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False,
                 use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop)
        self.norm2 = norm_layer(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop, norm_layer=norm_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x,  skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x,  skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    This module splits the image into patches and projects them to a vector space.
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        # Convolutional layer to extract patches and embed them
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        :param x: Input images of shape [B, C, H, W]
        :return: Patch embeddings of shape [B, N_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image size ({H}x{W}) must be divisible by patch size ({self.patch_size})"
        x = self.proj(x).flatten(2).transpose(1, 2)  # Shape: [B, N_patches, embed_dim]
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)        
        return out + residual




class Encoder(nn.Module):
    """
    Transformer-based U-Net model for diffusion denoising.

    This model uses a U-shaped architecture with skip connections between the encoder and decoder blocks.
    """
    def __init__(self, img_size=224, 
                 patch_size=16, 
                 in_chans=3,
                 in_conds=2,
                 one_hot_cond = False,
                 use_time = True,
                 embed_dim=768,                 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 attn_drop=0.0, 
                 mlp_drop=0.0, 
                 norm_layer=nn.LayerNorm,
                 mlp_time_embed=False, 
                 num_classes=-1,
                 use_checkpoint=False, 
                 conv=True,):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_chans = in_chans
        # self.ff = Base2FourierFeatures()
        # num_freq = 2
        # self.in_chans += 2*self.in_chans*num_freq

        self.in_conds = in_conds
        self.one_hot_cond = one_hot_cond
        self.use_time = use_time
        self.embed_dim = embed_dim
        self.extras = 1
        
        # Patch embedding module
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        if self.use_time:
            # Time embedding module for diffusion timesteps
            self.MPFourier = MPFourier(embed_dim)
            
            self.time_embed = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.SiLU(),
                nn.Linear(embed_dim * 2, embed_dim),
            ) if mlp_time_embed else nn.Identity()
            
                
        if self.in_conds > 0:
            self.cond_embed  = nn.Sequential(
                nn.Embedding(self.in_conds, embed_dim) if one_hot_cond else nn.Linear(self.in_conds, embed_dim),
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )

        # Positional embeddings for patches and extra tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        # Encoder blocks (first half of the U-Net)
        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                mlp_drop=mlp_drop, norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)
        ])

        # Middle block
        self.mid_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
            mlp_drop=mlp_drop, norm_layer=norm_layer, use_checkpoint=use_checkpoint)


        self.drop = NoScaleDropout(0.1)


        self.initialize_weights()
        

    def initialize_weights(self):        
        def _init_weights(m):
            # Initialize weights
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                #nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        # Initialize parameters
        nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=0.02, a=-2.0, b=2.0)
        
            
    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {'pos_embed'}

    def forward(self, x, timesteps= None,
                fluid_condition=None,
                y=None):
        
        """
        Forward pass of the UViT model.

        :param x: Input images of shape [B, C_in, H, W]
        :param timesteps: Timesteps tensor of shape [B]
        :param y: Optional class labels of shape [B]
        :return: Output images of shape [B, C_out, H, W]
        """

        
        skips = []
        # x = self.ff(x)
        x = self.patch_embed(x)  # Shape: [B, N_patches, embed_dim]
        B, L, D = x.shape

        if fluid_condition is not None:
            # Add label embedding if labels are provided
            cond = self.drop(self.cond_embed(fluid_condition.to(torch.int) if self.one_hot_cond else fluid_condition))

        if self.use_time:
            # Create time token
            #time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
            time_token = self.time_embed(self.MPFourier(timesteps))
            cond += time_token
            
        # Add positional embeddings
        cond = cond.unsqueeze(dim=1)
        x = torch.cat((cond, x), dim=1)
        x = x + self.pos_embed

        # Encoder
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)  # Store for skip connections

        # Middle block
        x = self.mid_block(x)

        return x, skips

class Decoder(nn.Module):
    """
    Transformer-based U-Net model for diffusion denoising.

    This model uses a U-shaped architecture with skip connections between the encoder and decoder blocks.
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 out_chans=3,
                 in_conds = 1,
                 one_hot_cond = False,                 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 attn_drop=0.0, 
                 mlp_drop=0.0, 
                 norm_layer=nn.LayerNorm,
                 mlp_time_embed=False, 
                 num_classes=-1,
                 use_checkpoint=False, 
                 conv=True, 
                 skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.out_chans = out_chans  # Number of output channels
        self.in_conds = in_conds
        self.one_hot_cond = one_hot_cond
        self.extras = 1
        self.MPFourier = MPFourier(embed_dim)
        
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.in_conds > 0:
            self.cond_embed  = nn.Sequential(
                nn.Embedding(self.in_conds, embed_dim) if one_hot_cond else nn.Linear(self.in_conds, embed_dim),
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )

        
        # Decoder blocks (second half of the U-Net), with optional skip connections
        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                mlp_drop=mlp_drop, norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)
        ])

        self.norm = norm_layer(embed_dim, eps=1e-6)  # Final normalization layer

        self.patch_dim = patch_size ** 2 * self.out_chans  # Dimension of each patch
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)  # Output projection


        # Final convolutional layer
        if conv:
            self.final_layer = nn.Conv2d(
                in_channels=self.out_chans, out_channels=self.out_chans, kernel_size=3, padding=1)
        else:
            self.final_layer = nn.Identity()


        self.drop = NoScaleDropout(0.1)

        self.initialize_weights()

    def initialize_weights(self):        
        def _init_weights( m):
            # Initialize weights
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                #nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
        

            
    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {'pos_embed'}

    def forward(self, x, skips,cond, cond_skips,
                timesteps,fluid_condition = None):
        
        """
        Forward pass of the UViT model.

        :param x: Input images of shape [B, C_in, H, W]
        :param timesteps: Timesteps tensor of shape [B]
        :param y: Optional class labels of shape [B]
        :return: Output images of shape [B, C_out, H, W]
        """

        # Create time token
        #time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = self.time_embed(self.MPFourier(timesteps))

        if fluid_condition is not None:
            # Add label embedding if labels are provided
            fluid_emb = self.cond_embed(fluid_condition.to(torch.int) if self.one_hot_cond else fluid_condition)
            time_token += self.drop(fluid_emb)


        #Combine middle channel
        x = x + cond
        x[:,0, :] = x[:, 0, :] + time_token
        # Decoder
        for blk in self.out_blocks:
            skip = skips.pop() + cond_skips.pop()
            x = blk(x, skip)  # Apply skip connection

        # Final normalization and projection
        x = self.norm(x)
        #remove conditioning info
        x = x[:, self.extras:, :]
        
        x = self.decoder_pred(x)  # Shape: [B, N_patches + extras, patch_dim]

        # Remove extra tokens
        #assert x.size(1) == self.extras + L, "Mismatch in sequence length after decoder_pred"
        
        # Reconstruct images from patches
        x = unpatchify(x, self.out_chans)  # Shape: [B, C_out, H, W]
        # Final convolutional layer
        x = self.final_layer(x)  # Shape: [B, C_out, H, W]
        return x

    

def UViT(image_size=256,
         in_channels=1,
         out_channels=1,
         max_factor=8,
         max_steps=3,
         model_size = 'small',
         mlp_ratio=4,
         attn_drop=0.1,
         mlp_drop=0.1,
         norm_layer=nn.LayerNorm,
         mlp_time_embed=True,
         use_checkpoint=False,
         conv=True,
         skip=True,
         ):

    if model_size == 'small':
        patch_size = 8
        embed_dim  = 384
        depth      = 13
        num_heads  = 8

    elif model_size == 'medium':
        patch_size = 8
        embed_dim  = 768
        depth      = 13
        num_heads  = 12

    elif model_size == 'big':
        patch_size = 8
        embed_dim  = 1152
        depth      = 29
        num_heads  = 16
    else:
        raise ValueError("size not found")
        
        
    base_encoder =  Encoder(
        img_size=image_size,
        patch_size=patch_size,
        in_chans=in_channels,
        in_conds = 1,
        embed_dim=embed_dim,  
        depth=depth,       
        num_heads=num_heads,    
        mlp_ratio=mlp_ratio,
        attn_drop=attn_drop,
        mlp_drop=mlp_drop,
        norm_layer=norm_layer,
        mlp_time_embed=mlp_time_embed,
        use_checkpoint=use_checkpoint,
        conv=conv,
    )

    forecast_encoder =  Encoder(
        img_size=image_size,
        patch_size=patch_size,
        in_chans=in_channels,
        in_conds = max_steps +1,
        use_time = False,
        one_hot_cond = True,
        embed_dim=embed_dim,  
        depth=depth,       
        num_heads=num_heads,    
        mlp_ratio=mlp_ratio,
        attn_drop=attn_drop,
        mlp_drop=mlp_drop,
        norm_layer=norm_layer,
        mlp_time_embed=mlp_time_embed,
        use_checkpoint=use_checkpoint,
        conv=conv,
    )

    superres_encoder =  Encoder(
        img_size=image_size,
        patch_size=patch_size,
        in_chans=in_channels,
        in_conds = max_factor +1,
        use_time = False,
        one_hot_cond = True,
        embed_dim=embed_dim,  
        depth=depth,       
        num_heads=num_heads,    
        mlp_ratio=mlp_ratio,
        attn_drop=attn_drop,
        mlp_drop=mlp_drop,
        norm_layer=norm_layer,
        mlp_time_embed=mlp_time_embed,
        use_checkpoint=use_checkpoint,
        conv=conv,
    )

    
    
    base_decoder =  Decoder(
        img_size=image_size,
        patch_size=patch_size,
        out_chans=out_channels,
        in_conds = 1,        
        embed_dim=embed_dim,  
        depth=depth,       
        num_heads=num_heads,    
        mlp_ratio=mlp_ratio,
        attn_drop=attn_drop,
        mlp_drop=mlp_drop,
        norm_layer=norm_layer,
        mlp_time_embed=mlp_time_embed,
        use_checkpoint=use_checkpoint,
        conv=conv,
        skip=skip,
    )


    return base_encoder, superres_encoder, forecast_encoder, base_decoder


if __name__ == "__main__":
    
    # Create a UViT model with specified parameters
    model = UViT(
        img_size=256,
        patch_size=16,
        in_chans=1,
        out_chans=1,
        embed_dim=128,  # Adjusted for testing purposes
        depth=4,        # Adjusted for testing purposes
        num_heads=4,    # Adjusted for testing purposes
        mlp_ratio=2.,
        attn_drop=0.1,
        mlp_drop=0.1,
        norm_layer=nn.LayerNorm,
        mlp_time_embed=True,
        use_checkpoint=False,
        conv=True,
        skip=True
    )

    # Create a batch of input images of shape [10, 2, 256, 256]
    x = torch.randn(10, 2, 256, 256)

    # Create a tensor of timesteps
    timesteps = torch.randint(0, 1000, (10,))

    # Forward pass
    output = model(x, timesteps)

    # Check the output shape
    assert output.shape == (10, 1, 256, 256), f"Expected output shape {(10, 1, 256, 256)}, got {output.shape}"
    print("Test passed. Output shape:", output.shape)
