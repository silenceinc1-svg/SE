import torch
import torch.nn as nn
import torchaudio

class attennuate_loss(nn.Module):
    def __init__(self, sp_weight=0.5, beta = 0.5):
        super().__init__()
        self.l1 = nn.SmoothL1Loss(beta = beta)
        self.sp_weight = sp_weight
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=1024,
                                        win_length=512, hop_length=256)

    def forward(self, input, target):
        if input.dim() == 2:
            input = input.unsqueeze(1)
            target = target.unsqueeze(1)
        if self.spectrogram.window.device != input.device:
            self.spectrogram = self.spectrogram.to(input.device)
        input_sp = self.spectrogram(input)
        target_sp = self.spectrogram(target)
        return (self.l1(input, target) +
                self.sp_weight * nn.functional.mse_loss(torch.log1p(input_sp), torch.log1p(target_sp)))
