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
audio_path='/data/dengjunyu/work/LightMuse/data/train/audio'
cue_path='/data/dengjunyu/work/LightMuse/data/train/cue_aligned'

# 模型路径
metadata_predictor_path = "./checkpoints/metadata_predictor.pth"
cuedata_num_predictor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
cuedata_predictor_path = "./checkpoints/cuedata_predictor.pth"

# 初始化Predictor类的示例
metadata_predictor = MetaDataPredictor(input_dim=128, output_dim=1).to(device)
cuedata_predictor = CueDataPredictor(input_dim=128, num_cuedata=max_len).to(device)

# 读取模型
if os.path.exists(metadata_predictor_path):
    metadata_predictor.load_state_dict(torch.load(metadata_predictor_path))
if os.path.exists(cuedata_predictor_path):
    cuedata_predictor.load_state_dict(torch.load(cuedata_predictor_path))

# 存储cue相对应的vgg特征和cuedata数量
all_vgg_features, all_num_features, all_cuedata_features = [], [], []
cuedata_num_list, all_time_list = [], []

# 处理数据
for item in os.listdir(audio_path): 
    filename = item[:-4]
    ap = AudioProcessing(audio_path)
    if item.endswith('.wav') or item.endswith('.mp3'):
        # Me频谱特征: [frames, 1, 96, 64]
        audio_feature = ap.to_mel(item)
        # VGG层特征: [frames, 128]
        vgg = VGG()
        vgg_feature = vgg(audio_feature.to(torch.device("cpu")))
        vgg_feature = vgg_feature.to(device)

        embedding = Embedding()
        cue_feature = embedding.get_cue_embedding(os.path.join(cue_path, filename + '.json'))
        timelist = embedding.get_timelist(os.path.join(cue_path, filename + ".json"))

        # 具体cuedata
        time_list = []

        # 计算 num_feature
        num_feature = []
        for i in range(len(vgg_feature)):
            count = sum(1.0 for time in timelist if i <= float(time) / 30 / 0.96 < i + 1)
            num_feature.append(count)
        
        num_feature = torch.tensor(num_feature, device=device).unsqueeze(1)

        for cue in cue_feature:
            time = cue[0]
            cuedata_num = cue[1]
            # time_list: [len(cue), 128]
            time_list.append(vgg_feature[int(time)].detach().unsqueeze(0))
            all_time_list.append(vgg_feature[int(time)].detach().unsqueeze(0))
            cuedata_num_list.append(cuedata_num)
            # cuedatas_feature: [len(cue), len(cuedatas), 3]
            cuedata_feature = [cuedatas for cuedatas in cue[2]]
            
            # 填充或截断
            if len(cuedata_feature) < max_len:
                cuedata_feature.extend([[0, 0, 0]] * (max_len - len(cuedata_feature))) 
            else:
                cuedata_feature = cuedata_feature[:max_len] 
            all_cuedata_features.append(torch.tensor(cuedata_feature, dtype = torch.float32).unsqueeze(0))

        # 存储数据
        all_vgg_features.append(vgg_feature)
        all_num_features.append(num_feature)
        

## 训练Cuedata_Predictor
all_time_list = torch.cat(all_time_list, dim=0).detach()
all_cuedata_features = torch.cat(all_cuedata_features, dim=0).detach()

cue_dataset = torch.utils.data.TensorDataset(all_time_list, all_cuedata_features)
cue_dataloader = torch.utils.data.DataLoader(cue_dataset, batch_size=32, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(cuedata_predictor.parameters(), lr=0.001)

num_epochs = 10
print('Training Cuedata Predictor:')
best_loss = float('inf') 
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_inputs, batch_targets in cue_dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        optimizer.zero_grad()
        outputs = cuedata_predictor(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() 

    epoch_loss /= len(cue_dataloader) 
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss (cuedata_predictor): {epoch_loss:.6f}")
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(cuedata_predictor.state_dict(), cuedata_predictor_path)
        print(f"New best model saved with Loss: {best_loss:.6f}")


## 训练cuedata_num_predictor
# cuedata_num_predictor.fit(all_time_list, cuedata_num_list)
# mse = mean_squared_error(cuedata_num_predictor.predict(all_time_list), cuedata_num_list)
# mae = mean_absolute_error(cuedata_num_predictor.predict(all_time_list), cuedata_num_list)
# print(f"Cuedata Num Predictor Trained, MSE: {mse}, MAE: {mae}")

# with open(cuedata_predictor_path, "wb") as f:
#     pickle.dump(cuedata_num_predictor, f)

# print(f"Cuedata Num Predictor saved to {cuedata_predictor_path}")


## 训练metadata_predictor
if all_vgg_features and all_num_features:
    all_vgg_features = torch.cat(all_vgg_features, dim=0).detach()
    all_num_features = torch.cat(all_num_features, dim=0).detach()
    
    # 创建数据集和 DataLoader
    meta_dataset = torch.utils.data.TensorDataset(all_vgg_features, all_num_features)
    meta_dataloader = torch.utils.data.DataLoader(meta_dataset, batch_size=32, shuffle=True)

    # 定义损失函数和优化器
    metadata_criterion = torch.nn.CrossEntropyLoss()
    metadata_optimizer = torch.optim.Adam(metadata_predictor.parameters(), lr=0.001)

    # 训练 MetaDataPredictor
    num_epochs = 1000
    print("Training Metadata Predictor:")
    best_loss = float('inf') 
    for epoch in range(num_epochs):
        epoch_loss = 0  
        for batch_features, batch_labels in meta_dataloader:
            outputs = metadata_predictor(batch_features)
            loss = metadata_criterion(outputs, batch_labels)
            
            metadata_optimizer.zero_grad()
            loss.backward()
            metadata_optimizer.step()
            
            epoch_loss += loss.item() 

        epoch_loss /= len(meta_dataloader) 
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss (metadata_predictor): {epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(metadata_predictor.state_dict(), metadata_predictor_path)
            print(f"New best model saved with Loss: {best_loss:.9f}")
else:
    print("No valid data found for training.")
