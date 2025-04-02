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
    def __init__(self):
        super(MusicEncoder, self).__init__()

    def forward(self, x):
        return x  

class LightingEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)  # 投影到高维
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [batch_size, len(cuedata), 3]
        输出: [batch_size, len(cuedata), embed_dim]
        """
        B, N, C = x.shape  # B: batch_size, N: len(cuedata), C: 3
        x = self.embedding(x)  # [B, N, embed_dim]
        x = self.transformer(x)  # Transformer 处理
        return x

# LAMP 对比学习模型
class LAMP(nn.Module):
    def __init__(self):
        super(LAMP, self).__init__()
        self.device_cpu = torch.device("cpu") 
        self.device_gpu = torch.device("cuda") 
        self.music_encoder = MusicEncoder().to(self.device_cpu)
        self.lighting_encoder = LightingEncoder().to(self.device_gpu)

    def forward(self, audio_waveform, cue_sequence):
        # 1. 音频数据先移动到 CPU 进行编码
        audio_waveform = audio_waveform.to(self.device_cpu)  
        with torch.no_grad():
            music_features = self.music_encoder(audio_waveform)  # CPU 上计算

        # 2. 计算后的音频特征再转移到 GPU
        music_features = music_features.to(self.device_gpu)

        # 3. 灯光 cue 数据直接放在 GPU 上
        cue_sequence = cue_sequence.to(self.device_gpu)
        lighting_features = self.lighting_encoder(cue_sequence)  # GPU 上计算

        return music_features, lighting_features

def contrastive_loss(music_features,lighting_features, temperature=0.07):
    similarity = (music_features @ lighting_features.T) / temperature
    labels = torch.arange(similarity.size(0)).to(similarity.device)
    return F.cross_entropy(similarity, labels)
