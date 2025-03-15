from torchvggish.vggish import VGGish
from torchvggish.audioset import vggish_input
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.device = torch.device("cpu")
        self.model_path = 'torchvggish/pytorch_vggish.pth'
        self.model = VGGish().to(self.device) 
        self.load_weights()  

    def load_weights(self):
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    
    def forward(self, x):
        return self.model(x)

    def eval_model(self):
        self.model.eval()
        return self

    def to_device(self):
        self.model.to(self.device)
        return self

# 定义 MetaDataPredictor
class MetaDataPredictor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
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
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

# 定义 Mapper
class Mapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mapper, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class CueDataPredictor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=3, num_cuedata=10):
        super(CueDataPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 线性层 1
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 线性层 2
        self.fc3 = nn.Linear(hidden_dim, output_dim * num_cuedata)  # 线性层 3
        self.num_cuedata = num_cuedata
        self.output_dim = output_dim
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 维度: [len(cue), num_cuedata * 3]
        x = x.view(-1, self.num_cuedata, self.output_dim)  # 重新 reshape: [len(cue), num_cuedata, 3]
        return x


# 定义 Cue Sequence Predictor
class CueSequencePredictor(nn.Module):
    def __init__(self):
        super(CueSequencePredictor, self).__init__()
        self.vggish = VGGish()
        self.classifier1 = Classifier(input_dim=128, output_dim=1)  # Flag: 是否有 cue
        self.classifier2 = Classifier(input_dim=128, output_dim=1)  # Num: cue 的数量
        self.mapper = Mapper(input_dim=512 + 2, output_dim=10)  # 假设 Cue Sequence 长度为 10

    def forward(self, x):
        # Step 1: 提取 VGGish 特征
        features = self.vggish(x)

        # Step 2: 通过 Classifier1 和 Classifier2 获取 Flag 和 Num
        flag = torch.sigmoid(self.classifier1(features))  # 输出范围在 [0, 1]
        num = F.relu(self.classifier2(features))  # 确保数量非负

        # Step 3: 将 Flag 和 Num 与 VGGish 特征拼接，输入到 Mapper
        combined = torch.cat((features, flag, num), dim=1)
        cue_sequence = self.mapper(combined)

        return cue_sequence

# 测试模型
