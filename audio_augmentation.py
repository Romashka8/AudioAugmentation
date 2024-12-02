import torch
import torchaudio

import numpy as np

from typing import Optional, Union


def augment_mfcc(mfcc: torch.Tensor,
                 sample_rate: Optional[int]=None, speed_factor: Optional[Union[float, int]]=None,
                 time_start_remove: Optional[int]=None, time_end_remove: Optional[int]=None,
                 freq_start_remove: Optional[int]=None, freq_end_remove: Optional[int]=None,
                 noise: Union[bool, float]=None
                 ) -> torch.Tensor:
    augmented_mfcc = mfcc.clone()
    batch_size, n_mfcc, time = augmented_mfcc.shape

    if sample_rate and speed_factor:
        new_sample_rate = int(sample_rate * speed_factor)
        resampler = torchaudio.transforms.Resample(sample_rate, new_sample_rate)
        augmented_mfcc = resampler(augmented_mfcc.reshape(-1, time)).reshape(batch_size, n_mfcc, -1)

    if time_start_remove and time_end_remove:
        assert 0 <= time_start_remove < time_end_remove <= time, "Incorrect indices(time remove)!"
        mask = torch.ones_like(augmented_mfcc, dtype=torch.bool)
        mask[:, :, time_start_remove:time_end_remove] = False
        augmented_mfcc = augmented_mfcc[mask].view(batch_size, n_mfcc, -1)

    if freq_start_remove and freq_end_remove:
        assert 0 <= freq_start_remove < freq_end_remove <= n_mfcc, "Incorrect indices(freq removal)!"
        mask = torch.ones_like(augmented_mfcc, dtype=torch.bool)
        mask[:, freq_start_remove:freq_end_remove, :] = False
        augmented_mfcc = augmented_mfcc[mask].view(batch_size, -1, augmented_mfcc.size(2))

    if noise:
        noise_std = noise if isinstance(noise, float) else 0.01
        augmented_mfcc += torch.rand_like(augmented_mfcc) * noise_std

    return augmented_mfcc


def augment_logmel_spectrogram(logmel_spec: torch.Tensor, speed_factor: Optional[int]=None,
							   time_start_remove: Optional[int]=None, time_end_remove: Optional[int]=None,
							   freq_start_remove: Optional[int]=None, freq_end_remove: Optional[int]=None,
							   noise: Union[bool, float]=None) -> torch.Tensor:
	augmented_logmel_spec = logmel_spec.clone()
	batch_size, n_mels, time = augmented_logmel_spec.shape

	if speed_factor:
		new_time = int(time * speed_factor)
		# make interpolation coordinates
		x = torch.linspace(0, time - 1, steps=new_time, device=logmel_spec.device)
		x_floor = torch.floor(x).long()
		x_ceil = torch.ceil(x).long()
		x_weight = x - x_floor.float()

		# do linear interpolation
		augmented_logmel_spec = torch.zeros((batch_size, n_mels, new_time), dtype=logmel_spec.dtype,
											device=logmel_spec.device)
		for i in range(batch_size):
			for j in range(n_mels):
				augmented_logmel_spec[i, j] = (
						(1 - x_weight) * logmel_spec[i, j, x_floor] + x_weight * logmel_spec[
					i, j, x_ceil]
				)

	if time_start_remove and time_end_remove:
		assert 0 <= time_start_remove < time_end_remove <= time, "Incorrect indices(time remove)!"
		mask = torch.ones_like(augmented_logmel_spec, dtype=torch.bool)
		mask[:, :, time_start_remove:time_end_remove] = False
		augmented_logmel_spec = augmented_logmel_spec[mask].view(batch_size, n_mels, -1)

	if freq_start_remove and freq_end_remove:
		assert 0 <= freq_start_remove < freq_end_remove <= n_mels, "Incorrect indices(freq removal)!"
		mask = torch.ones_like(augmented_logmel_spec, dtype=torch.bool)
		mask[:, freq_start_remove:freq_end_remove, :] = False
		augmented_logmel_spec = augmented_logmel_spec[mask].view(batch_size, -1, augmented_logmel_spec.size(2))

	if noise:
		noise_std = noise if isinstance(noise, float) else 0.01
		augmented_logmel_spec = augmented_logmel_spec + torch.randn_like(augmented_logmel_spec) * noise_std

	return augmented_logmel_spec
