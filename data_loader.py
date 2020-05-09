import os
import numpy as np
import scipy.signal as sps
from scipy.io import wavfile
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import sr#, people, sentences, counts
from python_speech_features import mfcc
from scipy.signal import stft


# Some notes on mfcc and stft (assuming samples of len ~100ms)
# typical window for STFTing (also used in mfcc) is 25ms
# Windwo is slid by 10ms
# This results in 800 samples since sr is 32000Hz
# As a result, we get decent coverage of frequency (800/2 = 400 frequencies)
# That's why nfft is 400
# So the stft will output like a 401x8 -- we transpose this to treat frequencies as diff channels
# And MFCC pools frequency to 13 ceptstrals, to get a 9x13 (comes with correct dims)

# Option to conisder is sacrificing frequency resolution for more temporal resolution
# So we have more than 8/9 samples in time -- we could shrink our window to 10ms or smth?

# opts should have parameters like:
# window (in ms), default to 25
# slide (in ms), default to 10
# transform -- choose between 'mfcc', 'stft','raw'

class KeystrokeDataset(Dataset):
    def __init__(self, data, opts):
        self.samples = []
        self.class_counts = [0]*len(data)
        for i, key in enumerate(data):
            print(i, key)
            for audio in data:
            	if opts.transform == 'mfcc':
	            	self.samples.append((mfcc(audio, sr,winlen=opts.window/1000, winstep=opts.slide/1000, nfft=800,nfilt=40), i))
	            elif opts.transform == 'stft':
	            	self.samples.append((stft(audio, sr, nperseg=int(opts.window*sr/1000), noverlap=int(opts.slide*sr/1000))[2].T, i))
	            elif opts.transform == 'raw':
	            	self.samples.append((audio, i))
	            else:
	            	raise NotImplementedError
            self.class_counts[i] += 1
        print("Length of dataset:", len(selof.samples))

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

# Split should be handled on the caller side (recommend random split)
# We abstract that part away so that user could pull a balanced test set but have imbalanced
# train/val data if they desire
def load_data(train_data, val_data, test_data, opts):
    """Creates training and test data loaders.
    """

    train_dataset = KeystrokeDataset(train_data, opts)
    val_dataset = KeystrokeDataset(val_data, opts)
    test_dataset = KeystorkeDataset(test_data, opts)

    train_sampler = WeightedRandomSampler([1/train_dataset.class_counts[sample[1]] for sample in train_dataset.samples], len(train_dataset.samples))
    val_sampler = WeightedRandomSampler([1/val_dataset.class_counts[sample[1]] for sample in val_dataset.samples], len(val_dataset.samples))
    test_sampler = WeightedRandomSampler([1/test_dataset.class_counts[sample[1]] for sample in test_dataset.samples], len(test_dataset.samples))

    if opts.balance_classes:
        print("Using WeightedRandomSampler")
        train_dloader = DataLoader(dataset=train_dataset, batch_size=min(opts.batch_size, len(train_dataset)), sampler=train_sampler, num_workers=opts.num_workers)
        val_dloader = DataLoader(dataset=val_dataset, batch_size=min(opts.batch_size, len(val_dataset)), sampler=val_sampler, num_workers=opts.num_workers)
        test_dloader = DataLoader(dataset=test_dataset, batch_size=min(opts.batch_size, len(test_dataset)), sampler=test_sampler, num_workers=opts.num_workers)
    else:
        print("Not weighting classes")
        train_dloader = DataLoader(dataset=train_dataset, batch_size=min(opts.batch_size, len(train_dataset)), shuffle=True, num_workers=opts.num_workers)
        val_dloader = DataLoader(dataset=val_dataset, batch_size=min(opts.batch_size, len(val_dataset)), shuffle=True, num_workers=opts.num_workers)
        test_dloader = DataLoader(dataset=test_dataset, batch_size=min(opts.batch_size, len(test_dataset)), shuffle=True, num_workers=opts.num_workers)

    return train_dloader, val_dloader, test_dloader

