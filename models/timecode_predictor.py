import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torchvggish.vggish import VGGish
from torchvggish.audioset import vggish_input

# 定义音乐编码器
class MusicEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super(MusicEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

# 定义光照编码器
class LightingEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [batch_size, len(cuedata), 3]
        输出: [batch_size, len(cuedata), embed_dim]
        """
        B, N, C = x.shape  # B: batch_size, N: len(cuedata), C: 3
        x = self.embedding(x)  # [B, N, embed_dim]
        x = self.transformer(x) # [B, N, embed_dim]
        return x

# LAMP模型
class LAMP(nn.Module):
    def __init__(self):
        super(LAMP, self).__init__()
        self.device_cpu = torch.device("cpu") 
        self.device_gpu = torch.device("cuda") 
        self.music_encoder = MusicEncoder().to(self.device_cpu)
        self.lighting_encoder = LightingEncoder().to(self.device_gpu)

    def forward(self, audio_waveform, cue_sequence):
        audio_waveform = audio_waveform.to(self.device_gpu)  
        with torch.no_grad():
            music_features = self.music_encoder(audio_waveform) 
        music_features = music_features.to(self.device_gpu)
        if cue_sequence is not None:
            cue_sequence = cue_sequence.to(self.device_gpu)
            lighting_features = self.lighting_encoder(cue_sequence)
        else:
            lighting_features = None

        return music_features, lighting_features

def contrastive_loss(music_features,lighting_features, temperature=0.07):
    similarity = (music_features @ lighting_features.T) / temperature
    labels = torch.arange(similarity.size(0)).to(similarity.device)
    return F.cross_entropy(similarity, labels)
