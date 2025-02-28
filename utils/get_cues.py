import pandas as pd
import numpy as np
import torch
import os
import wandb

def get_cue_feature():
    cue_path = 'data/cues'
    feature_path = 'data/audio_feature'

    xlsx_files = [f for f in os.listdir(cue_path) if f.endswith('.xlsx')]

    cues = []
    features = []

    for file in xlsx_files:
        file_name = file[:-5]
        
        # 读取cue的内容，并转换为numpy数组
        df = pd.read_excel(cue_path + '/' + file)
        cue = df.to_numpy()
        cues.append(cue) 

        # 读取同名的特征向量
        feature = np.load(feature_path + '/' + file_name + '.npy')
        print(feature.shape)
        features.append(feature)
    
    return cues, features

def process_cues(cues):
    for cue in cues:
        print(cue.shape)
    return cues


if __name__ == '__main__':
    cues, features = get_cue_feature()
    cues = process_cues(cues)