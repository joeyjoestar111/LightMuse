# import sys
# sys.path.append("/data/caojiale/LightMuse/torchvggish")
# sys.path.append("/data/caojiale/LightMuse")
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# import numpy as np
# from utils.embedding import Embedding
# from utils.audio_processing import AudioProcessing
# from models.cue_predictor import VGG, CueDataPredictor, MetaDataPredictor, train_cuedata_predictor, train_metadata_predictor, WeightedBCELoss
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import xgboost as xgb
# import pickle

# max_len = 512

# # 数据路径
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# audio_path='data/train/audio'
# cue_path='data/train/cue_aligned'
# val_audio_path='data/val/audio'
# val_cue_path='data/val/cue_aligned'

# # 模型路径
# metadata_predictor_path = "./checkpoints/metadata_predictor.pth"
# cuedata_num_predictor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
# cuedata_predictor_path = "./checkpoints/cuedata_predictor.pth"

# # 初始化Predictor类的示例
# metadata_predictor = MetaDataPredictor(input_dim=128, output_dim=1).to(device)
# cuedata_predictor = CueDataPredictor(input_dim=128, num_cuedata=max_len).to(device)

# # 读取模型
# if os.path.exists(metadata_predictor_path):
#     metadata_predictor.load_state_dict(torch.load(metadata_predictor_path))
# if os.path.exists(cuedata_predictor_path):
#     cuedata_predictor.load_state_dict(torch.load(cuedata_predictor_path))

# # 存储cue相对应的vgg特征和cuedata数量
# all_vgg_features, all_num_features, all_cuedata_features = [], [], []
# cuedata_num_list, all_time_list = [], []

# # 处理数据
# for item in os.listdir(audio_path): 
#     filename = item[:-4]
#     ap = AudioProcessing(audio_path)
#     if item.endswith('.wav') or item.endswith('.mp3'):
#         # Me频谱特征: [frames, 1, 96, 64]
#         audio_feature = ap.to_mel(item)
#         # VGG层特征: [frames, 128]
#         vgg = VGG()
#         vgg_feature = vgg(audio_feature.to(torch.device("cpu")))
#         vgg_feature = vgg_feature.to(device)

#         embedding = Embedding()
#         cue_feature = embedding.get_cue_embedding(os.path.join(cue_path, filename + '.json'))
#         timelist = embedding.get_timelist(os.path.join(cue_path, filename + ".json"))

#         # 具体cuedata
#         time_list = []

#         # 计算 num_feature
#         num_feature = []
#         for i in range(len(vgg_feature)):
#             count = sum(1.0 for time in timelist if i <= float(time) / 30 / 0.96 < i + 1)
#             num_feature.append(count)
        
#         num_feature = torch.tensor(num_feature, device=device).unsqueeze(1)
#         for cue in cue_feature:
#             time = cue[0]
#             cuedata_num = cue[1]
#             # time_list: [len(cue), 128]
#             time_list.append(vgg_feature[int(time)].detach().unsqueeze(0))
#             all_time_list.append(vgg_feature[int(time)].detach().unsqueeze(0))
#             cuedata_num_list.append(cuedata_num)
#             # cuedatas_feature: [len(cue), len(cuedatas), 3]
#             cuedata_feature = [cuedatas for cuedatas in cue[2]]
            
#             # 填充或截断
#             if len(cuedata_feature) < max_len:
#                 cuedata_feature.extend([[0, 0, 0]] * (max_len - len(cuedata_feature))) 
#             else:
#                 cuedata_feature = cuedata_feature[:max_len] 
#             all_cuedata_features.append(torch.tensor(cuedata_feature, dtype = torch.float32).unsqueeze(0))

#         # 存储数据
#         all_vgg_features.append(vgg_feature)
#         all_num_features.append(num_feature)

# val_features, val_num_features = [], []

# ## 读取val数据
# for item in os.listdir(val_audio_path): 
#     filename = item[:-4]
#     ap = AudioProcessing(val_audio_path)
#     if item.endswith('.wav') or item.endswith('.mp3'):
#         # Me频谱特征: [frames, 1, 96, 64]
#         audio_feature = ap.to_mel(item)
#         # VGG层特征: [frames, 128]
#         vgg = VGG()
#         vgg_feature = vgg(audio_feature.to(torch.device("cpu")))
#         vgg_feature = vgg_feature.to(device)

#         embedding = Embedding()
#         cue_feature = embedding.get_cue_embedding(os.path.join(val_cue_path, filename + '.json'))
#         timelist = embedding.get_timelist(os.path.join(val_cue_path, filename + ".json"))
#         num_feature = []
#         for i in range(len(vgg_feature)):
#             count = sum(1.0 for time in timelist if i <= float(time) / 30 / 0.96 < i + 1)
#             num_feature.append(count)
        
#         num_feature = torch.tensor(num_feature, device=device).unsqueeze(1)
#         val_num_features.append(num_feature)
#         val_features.append(vgg_feature)

# val_num_features = torch.cat(val_num_features, dim=0).detach().to(device)
# val_features = torch.cat(val_features, dim=0).detach().to(device)


# ## 训练Cuedata_Predictor
# all_time_list = torch.cat(all_time_list, dim=0).detach()
# all_cuedata_features = torch.cat(all_cuedata_features, dim=0).detach()


# cue_dataset = torch.utils.data.TensorDataset(all_time_list, all_cuedata_features)
# cue_dataloader = torch.utils.data.DataLoader(cue_dataset, batch_size=32, shuffle=True)

# ## 训练metadata_predictor

# all_vgg_features = torch.cat(all_vgg_features, dim=0).detach().to(device)
# all_num_features = torch.cat(all_num_features, dim=0).detach().to(device)


# # 创建数据集和 DataLoader
# meta_dataset = torch.utils.data.TensorDataset(all_vgg_features, all_num_features)
# meta_dataloader = torch.utils.data.DataLoader(meta_dataset, batch_size=128, shuffle=True)
# val_dataset = torch.utils.data.TensorDataset(val_features, val_num_features)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)

# lr_list = [1e-4, 1e-5, 1e-6, 1e-7]
# momentum_list = [0.9]
# pos_weight_list = [1]

# import itertools

# def grid_search_metadata(meta_dataloader, val_dataloader, lr_list, momentum_list, pos_weight_list, device):
#     best_loss = float('inf')
#     best_hyperparams = {}

#     # 遍历所有 lr、momentum 和 pos_weight 组合
#     for lr, momentum, pos_weight in itertools.product(lr_list, momentum_list, pos_weight_list):
#         print(f"\n🔍 Testing lr={lr}, momentum={momentum}, pos_weight={pos_weight} ...")

#         # 重新初始化模型
#         metadata_criterion = WeightedBCELoss(pos_weight=pos_weight, device=device)
#         metadata_optimizer = torch.optim.Adam(metadata_predictor.parameters(), lr=lr, betas=(momentum, 0.999))

#         # 开始训练
#         num_epochs = 500
#         for epoch in range(num_epochs):
#             metadata_predictor.train()
#             for batch_features, batch_labels in meta_dataloader:
#                 batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

#                 outputs = metadata_predictor(batch_features)
#                 loss = metadata_criterion(outputs, batch_labels)

#                 metadata_optimizer.zero_grad()
#                 loss.backward()
#                 metadata_optimizer.step()

#         # 计算验证集损失
#         metadata_predictor.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch_features, batch_labels in val_dataloader:
#                 batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
#                 outputs = metadata_predictor(batch_features)
#                 loss = metadata_criterion(outputs, batch_labels)
#                 val_loss += loss.item()
#         val_loss /= len(val_dataloader)

#         print(f"✅ Finished lr={lr}, momentum={momentum}, pos_weight={pos_weight}, Validation Loss: {val_loss:.6f}")

#         # 记录最佳超参数
#         if val_loss < best_loss:
#             best_loss = val_loss
#             best_hyperparams = {'lr': lr, 'momentum': momentum, 'pos_weight': pos_weight}
#             torch.save(metadata_predictor.state_dict(), "./checkpoints/metadata_model.pth")
#             print(f"🔥 New best model saved! lr={lr}, momentum={momentum}, pos_weight={pos_weight}, Loss={best_loss:.6f}")

#     print("\n🎯 Best Hyperparameters:")
#     print(f"Learning Rate: {best_hyperparams['lr']}")
#     print(f"Momentum: {best_hyperparams['momentum']}")
#     print(f"Pos Weight: {best_hyperparams['pos_weight']}")
#     print(f"Best Validation Loss: {best_loss:.6f}")
#     return best_hyperparams

# # train_metadata_predictor(meta_dataloader)
# best_params = grid_search_metadata(meta_dataloader, val_dataloader, lr_list, momentum_list, pos_weight_list, device)


# # train_cuedata_predictor(cue_dataloader)
# ## 训练cuedata_num_predictor
# # cuedata_num_predictor.fit(all_time_list, cuedata_num_list)
# # mse = mean_squared_error(cuedata_num_predictor.predict(all_time_list), cuedata_num_list)
# # mae = mean_absolute_error(cuedata_num_predictor.predict(all_time_list), cuedata_num_list)
# # print(f"Cuedata Num Predictor Trained, MSE: {mse}, MAE: {mae}")

# # with open(cuedata_predictor_path, "wb") as f:
# #     pickle.dump(cuedata_num_predictor, f)

# # print(f"Cuedata Num Predictor saved to {cuedata_predictor_path}")

import sys
sys.path.append("/data/caojiale/LightMuse/torchvggish")
sys.path.append("/data/caojiale/LightMuse")
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from utils.embedding import Embedding
from utils.audio_processing import AudioProcessing
from models.cue_predictor import VGG, CueDataPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import pickle
from torch.utils.data import Subset

max_len = 512

# 数据路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_path = 'data/train/audio'
cue_path = 'data/train/cue_aligned'
val_audio_path = 'data/val/audio'
val_cue_path = 'data/val/cue_aligned'

# 模型路径
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
metadata_predictor_path = os.path.join(checkpoint_dir, "metadata_predictor.pth")
cuedata_predictor_path = os.path.join(checkpoint_dir, "cuedata_predictor.pth")

# 定义 MetaDataPredictor
class MetaDataPredictor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):  # 修正 input_dim 为 128
        super(MetaDataPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        return self.model(x)

# 初始化模型
metadata_predictor = MetaDataPredictor(input_dim=128, output_dim=1).to(device)  # 修正 input_dim
cuedata_predictor = CueDataPredictor(input_dim=128, num_cuedata=max_len).to(device)  # 同步修正

# 加载已有模型
if os.path.exists(metadata_predictor_path):
    metadata_predictor.load_state_dict(torch.load(metadata_predictor_path))
if os.path.exists(cuedata_predictor_path):
    cuedata_predictor.load_state_dict(torch.load(cuedata_predictor_path))

# 存储数据
all_vgg_features, all_num_features, all_cuedata_features = [], [], []
all_time_list, cuedata_num_list = [], []

# 处理训练数据
for item in os.listdir(audio_path):
    filename = item[:-4]
    ap = AudioProcessing(audio_path)
    if item.endswith('.wav') or item.endswith('.mp3'):
        audio_feature = ap.to_mel(item)
        vgg = VGG()
        vgg_feature = vgg(audio_feature.to(torch.device("cpu")))
        print(f"vgg_feature shape for {item}: {vgg_feature.shape}")  # 调试输出
        vgg_feature = vgg_feature.to(device)

        embedding = Embedding()
        cue_feature = embedding.get_cue_embedding(os.path.join(cue_path, filename + '.json'))
        timelist = embedding.get_timelist(os.path.join(cue_path, filename + ".json"))

        num_feature = []
        for i in range(len(vgg_feature)):
            count = sum(1.0 for time in timelist if i <= float(time) / 30 / 0.96 < i + 1)
            num_feature.append(count)
        
        num_feature = torch.tensor(num_feature, device=device).unsqueeze(1)
        for cue in cue_feature:
            time = cue[0]
            cuedata_num = cue[1]
            all_time_list.append(vgg_feature[int(time)].detach().unsqueeze(0))
            cuedata_num_list.append(cuedata_num)
            cuedata_feature = [cuedatas for cuedatas in cue[2]]
            if len(cuedata_feature) < max_len:
                cuedata_feature.extend([[0, 0, 0]] * (max_len - len(cuedata_feature)))
            else:
                cuedata_feature = cuedata_feature[:max_len]
            all_cuedata_features.append(torch.tensor(cuedata_feature, dtype=torch.float32).unsqueeze(0))

        all_vgg_features.append(vgg_feature)
        all_num_features.append(num_feature)

# 处理验证数据
val_features, val_num_features = [], []
for item in os.listdir(val_audio_path):
    filename = item[:-4]
    ap = AudioProcessing(val_audio_path)
    if item.endswith('.wav') or item.endswith('.mp3'):
        audio_feature = ap.to_mel(item)
        vgg = VGG()
        vgg_feature = vgg(audio_feature.to(torch.device("cpu")))
        print(f"vgg_feature shape for {item}: {vgg_feature.shape}")  # 调试输出
        vgg_feature = vgg_feature.to(device)

        embedding = Embedding()
        cue_feature = embedding.get_cue_embedding(os.path.join(val_cue_path, filename + '.json'))
        timelist = embedding.get_timelist(os.path.join(val_cue_path, filename + ".json"))
        num_feature = []
        for i in range(len(vgg_feature)):
            count = sum(1.0 for time in timelist if i <= float(time) / 30 / 0.96 < i + 1)
            num_feature.append(count)
        
        num_feature = torch.tensor(num_feature, device=device).unsqueeze(1)
        val_num_features.append(num_feature)
        val_features.append(vgg_feature)

# 合并数据
all_vgg_features = torch.cat(all_vgg_features, dim=0).detach().to(device)
all_num_features = torch.cat(all_num_features, dim=0).detach().to(device)
val_features = torch.cat(val_features, dim=0).detach().to(device)
val_num_features = torch.cat(val_num_features, dim=0).detach().to(device)
all_time_list = torch.cat(all_time_list, dim=0).detach()
all_cuedata_features = torch.cat(all_cuedata_features, dim=0).detach()

print("all_vgg_features shape:", all_vgg_features.shape)
print("all_num_features shape:", all_num_features.shape)

# 检查数据分布
print("Training set num_features mean:", all_num_features.mean().item())
print("Training set non-zero ratio:", (all_num_features > 0).float().mean().item())
print("Validation set num_features mean:", val_num_features.mean().item())
print("Validation set non-zero ratio:", (val_num_features > 0).float().mean().item())

# 调整 train/val 配比，过采样正类
def oversample_positive(dataset, num_features):
    indices = torch.arange(len(num_features), device=num_features.device)
    positive_indices = indices[num_features.squeeze() > 0]
    negative_indices = indices[num_features.squeeze() == 0]
    oversampled_positive = positive_indices.repeat(10)
    new_indices = torch.cat([oversampled_positive, negative_indices])
    return Subset(dataset, new_indices)

meta_dataset = torch.utils.data.TensorDataset(all_vgg_features, all_num_features)
meta_dataset = oversample_positive(meta_dataset, all_num_features)
meta_dataloader = torch.utils.data.DataLoader(meta_dataset, batch_size=128, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(val_features, val_num_features)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
cue_dataset = torch.utils.data.TensorDataset(all_time_list, all_cuedata_features)
cue_dataloader = torch.utils.data.DataLoader(cue_dataset, batch_size=32, shuffle=True)

# 训练 MetaDataPredictor
def train_metadata_predictor(meta_dataloader, val_dataloader, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(metadata_predictor.parameters(), lr=0.001)
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        metadata_predictor.train()
        train_loss = 0.0
        for batch_features, batch_labels in meta_dataloader:
            print(f"batch_features shape: {batch_features.shape}")  # 调试输出
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = metadata_predictor(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(meta_dataloader)

        metadata_predictor.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for batch_features, batch_labels in val_dataloader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = metadata_predictor(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                val_preds.append(outputs)
        val_loss /= len(val_dataloader)
        val_preds = torch.cat(val_preds, dim=0)
        non_zero_ratio = (val_preds > 0.1).float().mean().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Non-zero ratio: {non_zero_ratio:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(metadata_predictor.state_dict(), metadata_predictor_path)
            print(f"New best model saved with Val Loss: {best_val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

# 执行训练
print("Training MetaDataPredictor...")
train_metadata_predictor(meta_dataloader, val_dataloader, device)

# 验证模型输出
metadata_predictor.eval()
with torch.no_grad():
    val_preds = metadata_predictor(val_features)
print("Validation predictions mean:", val_preds.mean().item())
print("Validation predictions non-zero ratio:", (val_preds > 0.1).float().mean().item())