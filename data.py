from torch.utils.data import Dataset
import torch
import numpy as np
import os
from os import path
import librosa
from tqdm import tqdm
from IPython.display import Audio, display
from audiomentations import Compose, TimeStretch, PitchShift, Shift, PolarityInversion
import soundfile as sf
import gc

sampling_rate = 16000

class BaseDenoisingDataset(Dataset):
  def __init__(self, data_path, train = True, sr = 16000, mode = 'waveform', dual_aug = None, mono_aug = None):
    self.sr = sr
    self.path = data_path
    self.path_clean = path.join(data_path, 'train' if train else 'test', 'clean')
    self.path_noisy = path.join(data_path, 'train' if train else 'test', 'noisy')
    self.train = train
    self.files = [f for f in os.listdir(self.path_clean) if '.wav' in f]
    self.mode = mode
    self.dual_aug = dual_aug
    self.mono_aug = mono_aug
    self.no_augm_mode = False
    self.addition = 256

  def __getitem__(self, i):
    name = self.files[i]
    noisy = librosa.load(path.join(self.path_noisy, name), sr = self.sr)[0]
    clean = librosa.load(path.join(self.path_clean, name), sr = self.sr)[0]
    if self.dual_aug != None and not self.no_augm_mode:
      noisy = self.dual_aug(noisy, self.sr)
      clean = self.dual_aug(clean, self.sr)
    if self.mono_aug != None and not self.no_augm_mode:
      clean = self.mono_aug(clean, self.sr)
    if self.mode == 'waveform':
      return torch.tensor(noisy), torch.tensor(clean)
    elif self.mode == 'spectrogram':
      return self.arr_to_sp(noisy), self.arr_to_sp(clean)

  def __len__(self):
    return len(self.files)

  def display(self, i, mode='clean', autoplay = True):
    display(Audio(self[i][mode =='clean'], rate=self.sr, autoplay=autoplay, normalize=False))

  def display_augmentations(self, mode = 'clean'):
    if self.dual_aug == None:
      print('No augmentations')
    # display current?

    i = np.random.randint(len(self))

    self.no_augm_mode = True
    initial = self[i][mode == 'clean'].numpy()
    self.no_augm_mode = False

    print('Innitaial audio')
    display(Audio(initial, rate=self.sr, normalize=False))

    print('Time Stretch display')
    for rate in [0.5, 0.9, 1.1, 1.5]:
      print('Rate - ', rate)
      augmented = TimeStretch(min_rate=rate, max_rate=rate, p=1.0)(initial, self.sr)
      display(Audio(augmented, rate=self.sr, normalize=False))

    print('Pitch Shift display')

    for rate in [-3, -2, -1, 1, 2, 3]:
      print('semitones - ', rate)
      augmented = PitchShift(min_semitones=rate, max_semitones=rate, p=1)(initial, self.sr)
      display(Audio(augmented, rate=self.sr, normalize=False))

    print('Polarity Inversion display')

    for rate in ['']:
      print('Inverted sygnal is audibly unextinguishable')
      augmented = PolarityInversion(p=1)(initial, self.sr)
      display(Audio(augmented, rate=self.sr, normalize=False))


  def arr_to_sp(self, arr):
    raw_spectr = librosa.stft(arr, n_fft=512, hop_length=256, win_length=512)
    spectr = torch.stack((torch.tensor(raw_spectr.real),
                          torch.tensor(raw_spectr.imag)))
    return spectr

  def get_all_durations(self):
    cnt = {}
    for i in tqdm(range(len(self))):
      info = sf.info(path.join(self.path_noisy, self.files[i]))
      dur = round(info.duration, 1)
      cnt[dur] = cnt.get(dur, 0) + 1
    return cnt

  def collate_fn(self, batch):
    if self.mode == 'waveform':
      noisy, clean = map(list, zip(*batch))

      for i in range(len(noisy)):
        if noisy[i].shape[0] > 16_000*2:
          dur = np.random.randint(16_000, 16_000 * 2)
          beg = np.random.randint(0, noisy[i].shape[0] - dur)
          mask = torch.zeros_like(noisy[i], dtype=torch.bool)
          mask[beg: beg + dur] = 1
          noisy[i] = noisy[i][mask]
          clean[i] = clean[i][mask]

      length = max(v.shape[0] for v in noisy)
      length += (self.addition - length%self.addition)%self.addition

      noisy_pad = torch.stack([
          torch.nn.functional.pad(val, (0, length - val.shape[0]))
          for val in noisy
      ])

      clean_pad = torch.stack([
          torch.nn.functional.pad(val, (0, length - val.shape[0]))
          for val in clean
      ])
      return noisy_pad, clean_pad

    if self.mode == 'spectrogram':
      time_axis = 2
      noisy, clean = map(list, zip(*batch))

      for i in range(len(noisy)):
        if noisy[i].shape[time_axis] > 16_000*2/256:
          dur = np.random.randint(int(16_000/256), int(16_000 * 2/256))
          beg = np.random.randint(0, noisy[i].shape[time_axis] - dur)
          mask = torch.zeros_like(noisy[i], dtype=torch.bool)
# time axis not implemented
          mask[:,:,beg: beg + dur] = 1
# time axis not implemented
          old_shape = noisy[i].shape
          noisy[i] = noisy[i][mask].reshape((old_shape[0], old_shape[1], -1))
          clean[i] = clean[i][mask].reshape((old_shape[0], old_shape[1], -1))

      length = max(v.shape[time_axis] for v in noisy)
      #length += (256 - length%256)%256
# time axis not implemented
      noisy_pad = torch.stack([
          torch.nn.functional.pad(val, (0, length - val.shape[time_axis]))
          for val in noisy
      ])

      clean_pad = torch.stack([
          torch.nn.functional.pad(val, (0, length - val.shape[time_axis]))
          for val in clean
      ])
      return noisy_pad, clean_pad












standart_dual = Compose([
    PolarityInversion(p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.75),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.6),
    #Shift(p=0.5),
])


my_path = "/content/data"
#my_path = "/content/drive/MyDrive/voicebank_demand_16k"
training_dataset = BaseDenoisingDataset(my_path, sr=sampling_rate, dual_aug=standart_dual)


from torch.utils.data import random_split
training_part, evaluation_part = random_split(training_dataset, [0.9, 0.1])

from torch.utils.data import DataLoader
evaluation_data = DataLoader(evaluation_part,
                           batch_size=24,
                           shuffle=True,
                           num_workers=2,
                           pin_memory=torch.cuda.is_available(),
                           prefetch_factor=2,
                           drop_last=True,
                           collate_fn=evaluation_part.dataset.collate_fn)

train_data = DataLoader(training_part,
                           batch_size=8,
                           shuffle=True,
                           num_workers=2,
                           pin_memory=torch.cuda.is_available(),
                           prefetch_factor=2,
                           drop_last=True,
                           collate_fn=training_part.dataset.collate_fn)

train_representation, _ = random_split(training_part, [0.11, 0.89])
shortened_train_data = DataLoader(train_representation,
                                  batch_size=12,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=torch.cuda.is_available(),
                                  prefetch_factor=2,
                                  drop_last=True,
                                  collate_fn=train_representation.dataset.dataset.collate_fn)

my_path_test = "/content/data"
#my_path_test = "/content/drive/MyDrive/voicebank_demand_16k"

test_dataset = BaseDenoisingDataset(my_path_test, sr=sampling_rate, train=False)

test_data = DataLoader(test_dataset,
                        batch_size=24,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=torch.cuda.is_available(),
                        prefetch_factor=2,
                        drop_last=True,
                        collate_fn=test_dataset.collate_fn)