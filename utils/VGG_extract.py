import sys
print(sys.path)
sys.path.append("./torchvggish")

from torchvggish.vggish import VGGish
from torchvggish.audioset import vggish_input
import torch
import os
import numpy as np
import json
from .embedding import Embedding

embedding = Embedding()

device = 'cpu'
model = VGGish()
state_dict = torch.load('torchvggish/pytorch_vggish.pth',map_location=device)

model.load_state_dict(state_dict)

model.eval().to(device)

audio_path='data/audio'

for item in os.listdir(audio_path):
        print("Processing:", item)
        input_wavfile=os.path.join(audio_path,item)
        wav_preprocess=vggish_input.wavfile_to_examples(input_wavfile)
        wav_preprocess = torch.from_numpy(wav_preprocess).unsqueeze(dim=1)
        input_wav = wav_preprocess.float().to(device)
        audio_feature_name=os.path.basename(input_wavfile)
        audio_feature_name=audio_feature_name[:-4]
        with torch.no_grad():
            output=model(input_wav)
            output=output.squeeze(0) 
            output=output.cpu().detach().numpy()
            np.save('data/audio_feature/'+audio_feature_name+'.npy', output)




# device = 'cpu'
# print(device)

# # 加载模型
# model = VGGish()
# state_dict = torch.load('torchvgg/pytorch_vggish.pth', map_location=device)
# model.load_state_dict(state_dict)
# model.eval().to(device)

# # 设置路径
# audio_path = 'data/audio'
# # audio_test_path = 'data/audio/test'

# out = []
# batch_size = 4 # 可以根据显存大小调整批次大小

# # 遍历音频文件夹
# for item in os.listdir(audio_path):
#     print(f"Processing {item}...")
#     input_wavfile = os.path.join(audio_path, item)

#     # 预处理音频文件
#     wav_preprocess = vggish_input.wavfile_to_examples(input_wavfile)  # (N, 96, 64)
#     wav_preprocess = torch.from_numpy(wav_preprocess).unsqueeze(dim=1)  # (N, 1, 96, 64)
#     print("preprocessed wave size: ", wav_preprocess.shape)
#     # 将音频分批处理
#     num_batches = (wav_preprocess.size(0) + batch_size - 1) // batch_size  # 确保能处理剩余的小批次
#     outputs = []
#     output = []

#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min(start_idx + batch_size, wav_preprocess.size(0))  # 确保最后一批次的大小不会超过实际数据量
#         input_batch = wav_preprocess[start_idx:end_idx].float().to(device)

#         # 执行模型预测
#         with torch.no_grad():
#             batch_output = model(input_batch).cpu().detach().numpy()
#         outputs.append(batch_output)
#     torch.cuda.empty_cache()

# stacked_outputs = torch.tensor(np.concatenate(outputs, axis=0))


# loss = criterion(output, labels)
# optimizer.step()
# print(f"Loss for {item}: {loss.item()}")

    # out.append(output)
    # print(f"Output shape for {item}: {output.shape}")

#     if(count >= 100):
#         break

#     if(count > 1000):
#         file = "backup.txt"
#         with open(file, "w") as f:
#             for i, output in enumerate(out):
#                 f.write(f"Audio feature {i}:\n") 
#                 np.savetxt(f, output, fmt="%.6f") 
#                 f.write("\n")

# max_shape = max(output.shape[0] for output in out)
# padded_out = [np.pad(output, ((0, max_shape - output.shape[0]), (0, 0)), mode='constant') for output in out]

# 然后再将其保存
# out_all = np.concatenate(padded_out, axis=0)
# out_all = out_all[:10, :128]
# print(out_all)

# with h5py.File("vggish.h5", "w") as f:
#     f.create_dataset("default", data=out_all)

# print("Feature extraction completed and saved to vggish.h5")


# 2. 预处理音频：将音频文件切分为多个0.96秒的片段
# def preprocess_audio(audio_path, sample_rate=16000, duration=0.96):
#     """
#     将音频文件切分为0.96秒的片段
#     """
#     y, sr = librosa.load(audio_path, sr=sample_rate)
#     segment_length = int(sample_rate * duration)
#     segments = [y[i:i+segment_length] for i in range(0, len(y), segment_length)]
    
#     # 确保每个片段的长度是相同的
#     segments = [seg for seg in segments if len(seg) == segment_length]
#     mel_spectrograms = []
#     return segments

# folder_path = 'data/audio'  # 替换为你音频文件夹的路径
# features = process_audio_folder(folder_path)

# print(features)

# # eatures = tf.gather(features, indices=[],axis = 1)

# print("Extracted features shape:", features.shape)  # (num_files, 128), 每个文件一个128维特征向量


    # if(count >= 50):
    #     break


# 计算准确率和 F1 分数
# print(test_labels, test_predictions)
# accuracy = accuracy_score(test_labels, test_predictions)
# f1 = f1_score(test_labels, test_predictions, average='weighted')  # 或者选择其他平均方法

# print(f"Test Accuracy: {accuracy:.4f}")
# print(f"Test F1-Score: {f1:.4f}")