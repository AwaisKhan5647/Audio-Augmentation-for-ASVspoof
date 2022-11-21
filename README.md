# Audio_Augmentations
This Reposatory include the audio augmentation with Python library audiomentation for the ASVspoof2019 Dataset comprises of LA and PA dataset. Inspired by albumentations. 

To implement the audio augmentation:

you have to install the following libraries as:

pip install audiomentations
https://pypi.org/project/audiomentations/

pip install librosa
https://pypi.org/project/librosa/

pip install soundfile
https://pypi.org/project/soundfile/

pip install pydub
https://pypi.org/project/pydub/

pip install tqdm
https://pypi.org/project/tqdm/

Usage Example:

from audiomentations import AddGaussianNoise

import numpy as np

augment = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)

Generate 2 seconds of dummy audio for the sake of example:

samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

Augment/transform/perturb the audio data:

augmented_samples = augment(samples=samples, sample_rate=16000)
