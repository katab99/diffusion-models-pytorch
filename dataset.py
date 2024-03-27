import os
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """
    data_dir : location of the files
    transform : transformation on the audio files
    sampling_rate : the sampling rate of the input files
    audio_length : the length of one audio file [s] - all file must have the same length
    """
    def __init__(self, data_dir, transform=None, sampling_rate=44100, audio_length=30):
        self.data_dir = data_dir
        self.transform = transform

        self.max_audio_length = sampling_rate * audio_length
        self.audio_files = self.get_audio_files()

    def __getitem__(self, index) :
        audio_path = self.audio_files[index]
        audio_waveform, sample_rate = torchaudio.load(audio_path)

        # to have the same length of each audio data
        if audio_waveform.size(1) < self.max_audio_length:
            pad_amount = self.max_audio_length - audio_waveform.size(1)
            audio_waveform = torch.nn.functional.pad(audio_waveform, (0, pad_amount))
        elif audio_waveform.size(1) > self.max_audio_length:
            audio_waveform = audio_waveform[:, :self.max_audio_length]
        
        if self.transform:
            audio_waveform = self.transform(audio_waveform)
        
        return audio_waveform
        
    def __len__(self):
        return len(self.audio_files)
    
    def get_audio_files(self):
        audio_files = []
        folders = os.listdir(self.data_dir)

        for folder in folders:
            if os.path.isdir(os.path.join(self.data_dir, folder)):
                files = os.listdir(os.path.join(self.data_dir, folder))

                for file in files:
                    if file.endswith('.wav'):
                        audio_files.append(os.path.join(self.data_dir, folder, file))

        return audio_files

