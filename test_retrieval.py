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
import torch.nn.functional as F
from models.cue_predictor import VGG
import json

max_len = 512

embedding = Embedding()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_path='/data/dengjunyu/work/LightMuse/data/val/audio'
cue_path='/data/dengjunyu/work/LightMuse/data/val/cue_aligned'
cue_corpus_path = '/data/dengjunyu/work/LightMuse/data/cue_corpus.json'
lamp_path = '/data/dengjunyu/work/LightMuse/checkpoints/LAMP_model.pth'

cue_corpus = []

embedding = Embedding()
cue_corpus = embedding.get_cue_corpus(cue_corpus_path)
for i in range(len(cue_corpus)):
    cuedata_feature = cue_corpus[i]
    if len(cuedata_feature) < max_len:
        cuedata_feature.extend([[0, 0, 0]] * (max_len - len(cuedata_feature))) 
    else:
        cue_corpus[i] = cue_corpus[i][:max_len] 

cue_corpus = torch.tensor(cue_corpus).to(device)

model = LAMP().to(device)
model.load_state_dict(torch.load(lamp_path, map_location = "cuda"))
model.eval()

# 所有Cue库的Embedding
cue_corpus = cue_corpus.float()
batch_size = 32
cue_corpus_chunks = torch.split(cue_corpus, batch_size)
cue_corpus_list = []
with torch.no_grad():  
    for chunk in cue_corpus_chunks:
        cue_corpus_list.append(model.lighting_encoder(chunk))

cue_corpus = torch.cat(cue_corpus_list, dim=0)

def retrieve_top_k_cues(audio_waveform, cue_corpus, k=1):
    music_feature = model.music_encoder(audio_waveform) 
    cue_corpus_pooled = cue_corpus.mean(dim=1) 
    # cue_corpus_pooled = model.lighting_encoder(cue_corpus)
    similarity = torch.matmul(music_feature, cue_corpus_pooled.T)

    # 获取前K个最高的索引
    top_k_values, top_k_indices = torch.topk(similarity, k=k)

    return top_k_indices.tolist()[0]  # 返回前K个cue的索引

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

        # 整理得到metadata和audiodata
        metadata, audiodata = [], []
        timelist = embedding.get_timelist(os.path.join(cue_path, filename + ".json"))
        for time in timelist:
            time = float(time) / 30 / 0.96
            if time > len(vgg_feature):
                continue
            audiodata.append(vgg_feature[int(time)].detach().unsqueeze(0))
        for i in range(len(vgg_feature)):
            count = sum(1.0 for time in timelist if i <= float(time) / 30 / 0.96 < i + 1)
            metadata.append(count)
        
        # 开始分音频段取出
        start_idx = 0
        final_cue_idx, final_cue = [], []
        audiodata = torch.cat(audiodata, dim=0).to(device)
        for count in metadata:
            count = int(count)
            if count > 0:
                for i in range(count):
                    data = audiodata[start_idx : start_idx + 1]
                    start_idx += 1
                selected_cues = retrieve_top_k_cues(data, cue_corpus, k=count)
                for item in selected_cues:
                    final_cue_idx.append(item)
        
        cue_corpus_path = "data/cue_corpus.json"
        with open(cue_corpus_path, "r") as f:
            all_cue = json.load(f)

        # 过滤出符合条件的元素，准备输出cue
        final_cue = [cue for cue in all_cue if isinstance(cue, dict) and cue.get("Num") in final_cue_idx]
        cue_output_dir = "./outputs/cue"
        os.makedirs(cue_output_dir, exist_ok=True)

        output_path = os.path.join(cue_output_dir, f"{filename}.json")

        # 将 final_cue 写入 JSON 文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_cue, f, indent=4, ensure_ascii=False)

        print(f"Filtered cues saved to {output_path}")

        # 准备输出timecode
        timecode_output_dir = "./outputs/timecode"
        os.makedirs(timecode_output_dir, exist_ok=True)
        final_timecode = []
        for (time, timecode) in zip(timelist, final_cue_idx):
            new_timecode = {
                "time": time,
                "cue": [1, 1, timecode]
            }
            final_timecode.append(new_timecode)
        output_path = os.path.join(timecode_output_dir, f"{filename}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_timecode, f, indent=4, ensure_ascii=False)
        print(f"Final timecode saved to {output_path}")

