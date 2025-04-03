import sys
sys.path.append("/data/dengjunyu/work/LightMuse/torchvggish")
sys.path.append("/data/dengjunyu/work/LightMuse")
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from utils.embedding import Embedding
from utils.audio_processing import AudioProcessing
from models.cue_predictor import VGG, CueDataPredictor, MetaDataPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import pickle

max_len = 512
# 数据路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_path='/data/dengjunyu/work/LightMuse/data/test/audio'
cue_path='/data/dengjunyu/work/LightMuse/data/test/cue_aligned'

# 模型路径
metadata_predictor_path = "./checkpoints/metadata_model.pth"
cuedata_num_predictor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
cuedata_predictor_path = "./checkpoints/cuedata_model.pth"

metadata_predictor = MetaDataPredictor(input_dim=128, output_dim=1).to(device)
cuedata_predictor = CueDataPredictor(input_dim=128, num_cuedata=max_len).to(device)

if os.path.exists(metadata_predictor_path):
    metadata_predictor.load_state_dict(torch.load(metadata_predictor_path))
if os.path.exists(cuedata_predictor_path):
    cuedata_predictor.load_state_dict(torch.load(cuedata_predictor_path))

for item in os.listdir(audio_path): 
    filename = item[:-4]
    ap = AudioProcessing(audio_path)
    if item.endswith('.wav') or item.endswith('.mp3'):
        audio_feature = ap.to_mel(item)
        # VGG层特征: [frames, 128]
        vgg = VGG()
        vgg_feature = vgg(audio_feature.to(torch.device("cpu")))
        vgg_feature = vgg_feature.to(device)
        # print(vgg_feature)

        print("MetaData:", torch.round(metadata_predictor(vgg_feature)).int())
        # print("CueData:", torch.round(cuedata_predictor(vgg_feature)))