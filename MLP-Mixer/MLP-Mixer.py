import torch
from torch import nn
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, 
                dim, 
                hidden_dim, 
                dropout = 0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, 
                dim, 
                num_patch, 
                token_dim, 
                channel_dim, 
                dropout = 0.):
        super().__init__()

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )

    def forward(self, x):
        x = x + self.token_mixer(x)

        x = x + self.channel_mixer(x)

        return x


class MLPMixer(nn.Module):
    def __init__(self,
                in_channels,
                dim,
                path_size, 
                image_size,
                depth,
                token_dim,
                channel_dim,
                num_classes, ):
        super().__init__()

        #check image_size for split patch
        assert image_size % path_size == 0, 'Image must be divided by patch size!! (이미지 크는 패치 사이즈로 나누어 떨어져야합니다!!)'
        
        self.num_path = (image_size // path_size) ** 2

        #1. split img to path size and projection
        self.path_projection = nn.Sequential(
            #1) input (1, 3, 224, 224) -> conv2d (path size : 16)-> output (1, 3, 14, 14)
            #2) input (1, 3, 14, 14) -> Rearragne -> output (1, (14 *14) ,3)
            nn.Conv2d(in_channels=in_channels,
                    out_channels=dim, 
                    kernel_size=path_size, 
                    stride=path_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        #2. define layer depth
        self.mixer_block = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_block.append(MixerBlock(dim, self.num_path, token_dim, channel_dim))
        
        self.layer_norm = nn.LayerNorm(dim)
        #3. mlp head - classifier
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )



    def forward(self, x):
        x = self.path_projection(x)
        for mixer_block in self.mixer_block:
            x = mixer_block(x)
        
        x = self.layer_norm(x)

        # global average pooling 
        x = x.mean(dim=1)

        return self.mlp_head(x)
    

# # check
# if __name__ == "__main__":
#     img = torch.ones([1, 3, 224, 224])
#     model = MLPMixer(in_channels=3,  dim=512, path_size=16, image_size=224, depth=8, token_dim=256, channel_dim=2048, num_classes=1000)

#     model(img)
