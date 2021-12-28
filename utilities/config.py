sample_rate = 32000
window_size = 1024
hop_size = 500  # So that there are 64 frames per second
mel_bins = 64
fmin = 0  # Hz
fmax = 24000  # Hz
num_channel = 2

'''
sample_rate = 44100
window_size = 2048
hop_size = 1024    # So that there are 64 frames per second
mel_bins = 128
fmin = 0      # Hz
fmax  = 22050   # Hz
num_channel = 2
'''
frames_per_second = sample_rate // hop_size
audio_duration = 10  # Audio recordings in DCASE2019 Task1 are all 10 seconds
frames_num = frames_per_second * audio_duration
total_samples = sample_rate * audio_duration

labels = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian',
          'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park']

lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}

dataset_dir = '/home/zjq/data/'
workspace = '/home/zjq/data/'


# dataset_dir = '/home/zjq/data/2020task1a'
# workspace = '/home/zjq/data/2020task1a'
sub_dir = 'TAU-urban-acoustic-scenes-2020-mobile-development'
#sub_dir = 'TAU-urban-acoustic-scenes-2019-mobile-development'
model_type = 'FeatureS'
batch_size = 64
cuda = True
loss='KL_WODA'
# evaluate_iteration = 8800
evaluate_iteration = 6400
# evaluate_accs = 0.7221
evaluate_accs = 0.5571
epoch = 100
sources_to_train = ['a', 'b', 'c', 's1', 's2', 's3']
sources_to_evaluate = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
# sources_to_train = ['a', 'b', 'c']
# sources_to_evaluate = ['a', 'b', 'c']
