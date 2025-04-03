import sys
sys.path.append("/data/dengjunyu/work/LightMuse/torchvggish")
sys.path.append("/data/dengjunyu/work/LightMuse")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from utils.embedding import Embedding
from utils.audio_processing import AudioProcessing
from models.timecode_predictor import LAMP, contrastive_loss
import xgboost as xgb
import pickle
from models.cue_predictor import VGG

max_len = 512

embedding = Embedding()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_path='/data/dengjunyu/work/LightMuse/data/train/audio'
cue_path='/data/dengjunyu/work/LightMuse/data/train/cue_aligned'

all_audio_features, all_num_features, lighting_data = [], [], []
cuedata_num_list, audio_data = [], []

for item in os.listdir(audio_path): 
    filename = item[:-4]
    ap = AudioProcessing(audio_path)
    if item.endswith('.wav') or item.endswith('.mp3'):
        # Mel频谱特征: [frames, 1, 96, 64]
        audio_feature = ap.to_mel(item)
        embedding = Embedding()
        vgg = VGG()
        vgg_feature = vgg(audio_feature.to(torch.device("cpu")))
        vgg_feature = vgg_feature.to(device)
        cue_feature = embedding.get_cue_embedding(os.path.join(cue_path, filename + '.json'))
        # timelist = embedding.get_timelist(os.path.join(cue_path, filename + ".json"))
        # 具体cuedata
        for cue in cue_feature:
            time = cue[0]
            cuedata_num = cue[1]
            # audio_data: [len(cue), 128]
            audio_data.append(vgg_feature[int(time)].detach().unsqueeze(0))
            # cuedatas_feature: [len(cue), len(cuedatas), 3]
            cuedata_feature = [cuedatas for cuedatas in cue[2]]

            if len(cuedata_feature) < max_len:
                cuedata_feature.extend([[0, 0, 0]] * (max_len - len(cuedata_feature))) 
            else:
                cuedata_feature = cuedata_feature[:max_len] 
            lighting_data.append(torch.tensor(cuedata_feature, dtype = torch.float32).unsqueeze(0))


audio_data = torch.cat(audio_data, dim=0).detach().to(torch.device("cuda"))
lighting_data = torch.cat(lighting_data, dim=0).detach().to(torch.device("cuda"))

model = LAMP().to(device)

class AudioLightingDataset(torch.utils.data.Dataset):
    def __init__(self, audio_data, lighting_data):
        self.audio_data = audio_data 
        self.lighting_data = lighting_data  

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        return self.audio_data[idx], self.lighting_data[idx]

# 数据加载器
dataset = AudioLightingDataset(audio_data, lighting_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=None)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

num_epochs = 100
for epoch in range(num_epochs):
    model = model.to(device)
    model.train()
    total_loss = 0

    for audio_waveform, cue_sequence in dataloader:
        audio_waveform, cue_sequence = audio_waveform.to(device), cue_sequence.to(device)
        # 前向传播
        music_features, lighting_features = model(audio_waveform, cue_sequence)

        # 计算对比损失
        loss = contrastive_loss(music_features, lighting_features.mean(dim=1))

        # 反向传播 & 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

torch.save(model.state_dict(), "./checkpoints/LAMP_model.pth")
print("Checkpoint Successful Saved")