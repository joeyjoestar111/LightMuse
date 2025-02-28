import sys
sys.path.append("/data/dengjunyu/work/LightMuse/torchvggish")
sys.path.append("/data/dengjunyu/work/LightMuse")
from torchvggish.vggish import VGGish
from torchvggish.audioset import vggish_input
import torch
import os
import numpy as np
from utils.embedding import Embedding
from utils.audio_processing import AudioProcessing
from models.cue_predictor import CueSequencePredictor, VGG, Mapper, Classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_path='/data/dengjunyu/work/LightMuse/data/audio'
cue_path='/data/dengjunyu/work/LightMuse/data/cue_aligned'

for item in os.listdir(audio_path): 
    filename = item[:-4]
    ap = AudioProcessing(audio_path)
    if item.endswith('.wav') or item.endswith('.mp3'):
        # Mel Spectrem Features: [frames, 1, 96, 64]
        audio_feature = ap.to_mel(item)
        # VGG Layer Features: [frames, 128]
        vgg = VGG()
        vgg_feature = vgg(audio_feature.to(torch.device("cpu")))
        classifier1 = Classifier(input_dim=128, output_dim=1)


        embedding = Embedding()
        timelist = embedding.get_timelist(os.path.join(cue_path, filename + ".json"))
        print(timelist)
        flag_feature = []
        for i in range(len(vgg_feature)):
            for time in timelist:
                framed_time = float(time) / 0.96
                if framed_time >= i and framed_time <= i + 1:
                    # print(framed_time)
                    flag_feature.append(1)
                    break
            else:
                flag_feature.append(0)
        flag_feature = torch.tensor(flag_feature).to(torch.device("cpu"))
        flag_feature = flag_feature.unsqueeze(1)
        # print(flag_feature)