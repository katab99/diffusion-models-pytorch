import os
import torch
import torchvision
import torchaudio
from torchaudio.transforms import Resample, Spectrogram,TimeStretch, FrequencyMasking, TimeMasking, MelScale
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset import AudioDataset

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    resample_freq = 22_100
    n_fft = 1024

    transforms = torch.nn.Sequential(
        Resample(orig_freq=22100, new_freq=resample_freq), # resample to 16kHz - TODO: check if this is necessary
        Spectrogram(n_fft=n_fft), # compute spectrogram
        # torch.nn.Sequential(
        #     TimeStretch(0.8, fixed_rate=True),
        #     FrequencyMasking(freq_mask_param=80),
        #     TimeMasking(time_mask_param=80),
        #     ), # augmentation
        MelScale(n_mels=128, sample_rate=resample_freq, n_stft=n_fft // 2 + 1), # convert to mel scale
        )
    
    dataset = AudioDataset(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
