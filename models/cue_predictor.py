from torchvggish.vggish import VGGish
from torchvggish.audioset import vggish_input
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda")

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


# 加权的BCE Loss
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=10, device=None):  # 设置正类权重
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    def forward(self, logits, targets):
        return self.loss(logits, targets)

def train_metadata_predictor(metadata_predictor, meta_dataloader, path):
        metadata_criterion = WeightedBCELoss(pos_weight=10)
        metadata_optimizer = torch.optim.Adam(metadata_predictor.parameters(), lr=0.001)

        # 训练 MetaDataPredictor
        num_epochs = 1000
        print("Training Metadata Predictor:")
        best_loss = float('inf') 
        for epoch in range(num_epochs):
            epoch_loss = 0  
            for batch_features, batch_labels in meta_dataloader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
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
                torch.save(metadata_predictor.state_dict(), path)
                print(f"New best model saved with Loss: {best_loss:.9f}")


def train_cuedata_predictor(cuedata_predictor, cue_dataloader, path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cuedata_predictor.parameters(), lr=0.005)

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
            torch.save(cuedata_predictor.state_dict(), path)
            print(f"New best model saved with Loss: {best_loss:.6f}")