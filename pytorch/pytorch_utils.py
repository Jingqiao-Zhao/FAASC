import numpy as np
import torch
from utilities.utlies import trans

def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]


def forward(model, generate_func, cuda, return_input=False,
            return_target=False):
    '''Forward data to model in mini-batch.

    Args:
      model: object
      generate_func: function
      cuda: bool
      return_input: bool
      return_target: bool
      max_validate_num: None | int, maximum mini-batch to forward to speed up validation
    '''
    output_dict = {}

    # Evaluate on mini-batch
    for batch_data_dict in generate_func:

        # Predict

        batch_feature = move_data_to_gpu(batch_data_dict['feature'], cuda)


        with torch.no_grad():
            model.eval()
            batch_output,cla1,cla2 = model(batch_feature)

        append_to_dict(output_dict, 'output1', batch_output.data.cpu().numpy())
        append_to_dict(output_dict, 'output2', cla1.data.cpu().numpy())
        append_to_dict(output_dict, 'output3', cla2.data.cpu().numpy())

        if return_input:
            append_to_dict(output_dict, 'feature', batch_data_dict['feature'])

        if return_target:

            append_to_dict(output_dict, 'target', batch_data_dict['target'])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict

def forward2(model, generate_func, cuda, return_input=False,
            return_target=False):
    '''Forward data to model in mini-batch.

    Args:
      model: object
      generate_func: function
      cuda: bool
      return_input: bool
      return_target: bool
      max_validate_num: None | int, maximum mini-batch to forward to speed up validation
    '''
    output_dict = {}

    # Evaluate on mini-batch
    for batch_data_dict in generate_func:

        # Predict

        batch_feature = move_data_to_gpu(batch_data_dict['feature'], cuda)


        with torch.no_grad():
            model.eval()
            output = model(batch_feature)


        append_to_dict(output_dict, 'output', output.data.cpu().numpy())

        if return_input:
            append_to_dict(output_dict, 'feature', batch_data_dict['feature'])

        if return_target:

            append_to_dict(output_dict, 'target', batch_data_dict['target'])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


def change_feature(a, A):
    for i in range(a):
        A.insert(0, A.pop())
    return (A)


def random_feature(features):
    '''
    Change the input shape's time_bin

    input time_bin:[0,1,2,3,4...]
    return time_bin:[...n-1,n,n+1...]

    :param features: input_shape:batch,time_bin,frame_bin
    :return: same to input
    '''

    batch, time_step, feature_map = features.shape[0], features.shape[1], features.shape[2]

    time_list = np.linspace(0, time_step - 1, num=time_step)

    pad = np.random.choice(time_list, 1)

    slice_list = change_feature(int(pad[0]), time_list.tolist())

    for i in range(batch):
        features[i] = features[i][slice_list]

    return features
