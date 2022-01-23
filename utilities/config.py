sample_rate = 32000
window_size = 1024
hop_size = 500  # So that there are 64 frames per second
mel_bins = 64
fmin = 0  # Hz
fmax = 24000  # Hz
num_channel = 2


frames_per_second = sample_rate // hop_size
audio_duration = 10  # Audio recordings in DCASE2019 Task1 are all 10 seconds
frames_num = frames_per_second * audio_duration
total_samples = sample_rate * audio_duration

labels = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian',
          'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park']

lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}

Data_years='2020'

if Data_years=='2019':
    dataset_dir = '/home/zjq/data/'
    workspace = '/home/zjq/data/'
    sub_dir = 'TAU-urban-acoustic-scenes-2019-mobile-development'
    sources_to_train = ['a', 'b', 'c']
    sources_to_evaluate = ['a', 'b', 'c']
    evaluate_iteration = 6400
    evaluate_accs = 0.5571

elif Data_years=='2020':
    dataset_dir = '/home/zjq/data/'
    workspace = '/home/zjq/data/'
    sub_dir = 'TAU-urban-acoustic-scenes-2020-mobile-development'
    sources_to_train = ['a', 'b', 'c', 's1', 's2', 's3']
    sources_to_evaluate = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
    evaluate_iteration = 8800
    evaluate_accs = 0.7221

# Set this option to True if you want to see feature maps as they are aligned
feature_maps=False

model_type = 'FeatureS'
batch_size = 64
cuda = True
cuda_id='0'
loss='KL_WODA'
epoch = 100


