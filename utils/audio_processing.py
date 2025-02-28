import sys
# print(sys.path)
from pydub import AudioSegment
import os
import torch
import numpy as np
from torchvggish.vggish import VGGish
from torchvggish.audioset import vggish_input

class AudioProcessing():
    def __init__(self, audio_path):
            self.audio_path = audio_path

    def mp4_to_wav(self, input_mp4):
        audio = AudioSegment.from_file(input_mp4, format="mp4")
        audio = audio.set_frame_rate(22050).set_channels(1)
        output_wav = input_mp4[:-4] + ".wav" 
        audio.export(output_wav, format="wav")
    
    def to_mel(self, item):
        if item.endswith('.wav') or item.endswith('.mp3'):  
            print("Processing:", item)
            input_wavfile = os.path.join(self.audio_path, item)
            wav_preprocess = vggish_input.wavfile_to_examples(input_wavfile)  
            wav_preprocess = torch.from_numpy(wav_preprocess).unsqueeze(dim=1)  
            input_wav = wav_preprocess.float()
            return input_wav
        
        

# AudioProcessing = AudioProcessing(audio_path)
# AudioProcessing.mp4_to_wav(audio_path + '/zitomayo.mp4')





