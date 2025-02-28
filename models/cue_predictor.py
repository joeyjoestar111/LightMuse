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

# 定义 Classifier1 和 Classifier2
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

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

# 定义 Cue Sequence Predictor
class CueSequencePredictor(nn.Module):
    def __init__(self):
        super(CueSequencePredictor, self).__init__()
        self.vggish = VGGish()
        self.classifier1 = Classifier(input_dim=512, output_dim=1)  # Flag: 是否有 cue
        self.classifier2 = Classifier(input_dim=512, output_dim=1)  # Num: cue 的数量
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
