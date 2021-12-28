import os
import numpy as np
import librosa
import time
import h5py
from utilities import config
from utilities.utlies import *

class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor.

        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''

        self.window_size = window_size
        self.hop_size = hop_size
        self.window_size = window_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.sr = sample_rate

    def transform(self,audio):

        mel_spectrogram = librosa.feature.melspectrogram(y = audio,
                                                         sr=self.sr,
                                                         n_fft=self.window_size,
                                                         hop_length=self.hop_size,
                                                         n_mels=self.mel_bins,

                                                         fmin=self.fmin,
                                                         fmax=self.sr/2,
                                                         htk=True, norm=None)
        # Log mel spectrogram
        logmel_spectrogram = np.log(mel_spectrogram+1e-8)
        logmel_spectrogram = (logmel_spectrogram - np.min(logmel_spectrogram)) / (np.max(logmel_spectrogram) - np.min(logmel_spectrogram))
        logmel_spectrogram = logmel_spectrogram.astype(np.float32)

        return logmel_spectrogram


def calculate_feature_for_all_audio_files():
    '''Calculate feature of audio files and write out features to a hdf5 file.

    Args:
      dataset_dir: string
      workspace: string
      subtask: 'a' | 'b' | 'c'
      data_type: 'development' | 'evaluation'
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arguments & parameters
    dataset_dir = config.dataset_dir
    workspace = config.workspace
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    frames_num = config.frames_num
    total_samples = config.total_samples
    lb_to_idx = config.lb_to_idx



    audios_dir = os.path.join(dataset_dir, 'TAU-urban-acoustic-scenes-2020-mobile-development', 'audio')

    metadata_path = os.path.join(dataset_dir, 'TAU-urban-acoustic-scenes-2020-mobile-development', 'meta.csv')


    feature_path = os.path.join(workspace, 'features',
                                'logmel_{}frames_{}melbins.h5'.format(frames_per_second, mel_bins))

    create_folder(os.path.dirname(feature_path))

    # Feature extractor
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        fmin=fmin,
        fmax=fmax,
    )

    # Read metadata

    meta_dict = read_metadata(metadata_path)

    extract_time = time.time()
    # Hdf5 file for storing features and targets
    hf = h5py.File(feature_path, 'w')

    hf.create_dataset(
        name='audio_name',
        data=[audio_name.encode() for audio_name in meta_dict['audio_name']],
        dtype='S80')

    if 'scene_label' in meta_dict.keys():
        hf.create_dataset(
            name='scene_label',
            data=[scene_label.encode() for scene_label in meta_dict['scene_label']],
            dtype='S24')

    if 'identifier' in meta_dict.keys():
        hf.create_dataset(
            name='identifier',
            data=[identifier.encode() for identifier in meta_dict['identifier']],
            dtype='S24')

    if 'source_label' in meta_dict.keys():
        hf.create_dataset(
            name='source_label',
            data=[source_label.encode() for source_label in meta_dict['source_label']],
            dtype='S8')

    hf.create_dataset(
        name='feature',
        shape=(0, mel_bins,frames_num),
        maxshape=(None, mel_bins,frames_num),
        dtype=np.float32)

    for (n, audio_name) in enumerate(meta_dict['audio_name']):
        audio_path = os.path.join(audios_dir, audio_name)
        print(n, audio_path)

        # Read audio
        (audio, _) = librosa.load(audio_path,sr=sample_rate)

        # Extract feature
        feature = feature_extractor.transform(audio)

        # Remove the extra log mel spectrogram frames caused by padding zero
        feature = feature[:,0: frames_num]

        hf['feature'].resize((n + 1, mel_bins,frames_num))
        hf['feature'][n] = feature

    hf.close()

    print('Write hdf5 file to {} using {:.3f} s'.format(
        feature_path, time.time() - extract_time))


def calculate_scalar():
    '''Calculate and write out scalar of features.
    '''

    # Arguments & parameters
    workspace = config.workspace
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second



    feature_path = os.path.join(workspace, 'features',
                                'logmel_{}frames_{}melbins.h5'.format( frames_per_second, mel_bins))

    scalar_path = os.path.join(workspace, 'scalars',
                               'logmel_{}frames_{}melbins.h5'.format( frames_per_second, mel_bins))
    create_folder(os.path.dirname(scalar_path))

    # Load data
    load_time = time.time()

    with h5py.File(feature_path, 'r') as hf:
        features = hf['feature'][:]

    # Calculate scalar
    features = np.concatenate(features, axis=0)
    (mean, std) = calculate_scalar_of_tensor(features)

    with h5py.File(scalar_path, 'w') as hf:
        hf.create_dataset('mean', data=mean, dtype=np.float32)
        hf.create_dataset('std', data=std, dtype=np.float32)

    print('All features: {}'.format(features.shape))
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    print('Write out scalar to {}'.format(scalar_path))

if __name__ == '__main__':
    calculate_feature_for_all_audio_files()
    calculate_scalar()


