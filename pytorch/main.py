import os
import sys
sys.path.append("..")
import librosa
import librosa.display
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from utilities.utlies import create_folder, load_scalar,create_logging
from utilities import config
from pytorch.dataset_mixup import *
from pytorch.model import *
from pytorch.evaluate import Evaluator, StatisticsContainer
from pytorch.pytorch_utils import move_data_to_gpu, forward
from tensorboardX import SummaryWriter
from pytorch.loss import nll_loss,discrepancy,klv,mmd
from pytorch.pytorch_utils import random_feature


def train():
    '''Training. Model will be saved after several iterations.

    Args:
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      data_type: 'development' | 'evaluation'
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      batch_size: int
      cuda: bool
    '''
    # Arugments & parameters
    writer = SummaryWriter('./result2')
    dataset_dir = config.dataset_dir
    workspace = config.workspace
    model_type = config.model_type
    batch_size = config.batch_size
    cuda = config.cuda and torch.cuda.is_available()

    filename = 'log'

    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    reduce_lr = True

    sources_to_evaluate = config.sources_to_evaluate
    sources_to_train = config.sources_to_train
    in_domain_classes_num = len(config.labels)
    prefix = ''
    sub_dir = config.sub_dir
    loss = config.loss

    train_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup',
                             'fold1_train.csv')

    validate_csv = os.path.join(dataset_dir, sub_dir, 'evaluation_setup',
                                'fold1_evaluate.csv')


    feature_hdf5_path = os.path.join(workspace, 'features',
                                     '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                                     '{}.h5'.format(sub_dir))

    scalar_path = os.path.join(workspace, 'scalars',
                               '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
                               '{}.h5'.format(sub_dir))

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename,
                                   'logmel_{}frames_{}melbins_{}'.format(frames_per_second, mel_bins,loss),
                                   '{}'.format(sub_dir),model_type)
    create_folder(checkpoints_dir)

    validate_statistics_path = os.path.join(workspace, 'statistics', filename,
                                            'logmel_{}frames_{}melbins'.format(frames_per_second, mel_bins),
                                            '{}'.format(sub_dir),
                                            model_type, 'validate_statistics.pickle')

    create_folder(os.path.dirname(validate_statistics_path))
    logs_dir = os.path.join(workspace, 'logs',
        'logmel_{}frames_{}melbins'.format( frames_per_second, mel_bins),
        '{}'.format(sub_dir), model_type)

    create_logging(logs_dir, 'w')

    if cuda:
        logging.info('Using GPU.')


    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    # Load scalar
    scalar = load_scalar(scalar_path)

    Model = eval(model_type)

    model = Model(in_domain_classes_num, activation='logsoftmax')

    if cuda:
        model.cuda()

    opt = optim.AdamW(model.parameters(),lr=0.001,betas=(0.9, 0.999),eps=1e-08,weight_decay=0.01,amsgrad=True)

    loss_cla = nll_loss

    loss_adv = nn.MSELoss()


    # Data generator
    data_generator = DataGenerator(
        feature_hdf5_path=feature_hdf5_path,
        train_csv=train_csv,
        evaluate_csv= validate_csv,
        scalar=scalar,
        batch_size=batch_size)

    # Evaluator
    evaluator = Evaluator(
        model=model,
        data_generator=data_generator,
        cuda=cuda)


    # Statistics
    validate_statistics_container = StatisticsContainer(validate_statistics_path)

    train_bgn_time = time.time()
    iteration = 0


    best_acc = 0
    for index, batch_data_dict in enumerate(data_generator.generate_train_domain()):
        # Evaluate
        if (iteration) % 200 == 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))
            train_fin_time = time.time()
            train_accs = []
            train_cla1=[]
            train_cla2=[]
            evaluate_accs = []
            evaluate_cla1=[]
            evaluate_cla2=[]
            loss_train = []
            loss_eva = []
            for source in sources_to_train:
                train_statistics, train_acc,cla1,cla2 ,loss= evaluator.evaluate(
                    data_type='train',
                    source=source,
                    max_iteration=None,
                    verbose=False)
                train_accs.append(train_acc)
                train_cla1.append(cla1)
                train_cla2.append(cla2)
                loss_train.append(loss)


            for source in sources_to_evaluate:
                validate_statistics, evaluate_acc,cla1,cla2,loss = evaluator.evaluate(
                    data_type='evaluate',
                    source=source,
                    max_iteration=None,
                    verbose=False)
                validate_statistics_container.append_and_dump(
                    iteration, source, validate_statistics)

                evaluate_accs.append(evaluate_acc)
                evaluate_cla1.append(cla1)
                evaluate_cla2.append(cla2)
                loss_eva.append(loss)


            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            logging.info('Average_train_acc: {}'.format(np.mean(train_accs)))
            logging.info('Average_evaluate_acc: {}'.format(np.mean(evaluate_accs)))

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()
            writer.close()

        # Save model\
            if np.mean(evaluate_accs)>best_acc and np.mean(evaluate_accs)>0.6:
                best_acc = np.mean(evaluate_accs)
                checkpoint = {
                    'iteration': iteration,
                    'model': model.state_dict(),
                    'optimizer': opt.state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_iterations_best{:.4f}.pth'.format(iteration,np.mean(evaluate_accs)))

                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))


        # Reduce learning rate

        if reduce_lr and iteration % 200 == 0 and iteration > 0 and opt.param_groups[0]['lr'] >1e-7:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.9

        for key in batch_data_dict.keys():
            if key in ['only_source_feature', 'only_source_target']:
                batch_data_dict[key] = move_data_to_gpu(batch_data_dict[key], cuda)

        model.train()
        # Move data to GPU
        for key in batch_data_dict.keys():
            if key in ['source_feature', 'target_feature', 'target']:
                batch_data_dict[key] = move_data_to_gpu(batch_data_dict[key], cuda)


        output_s,_,_ = model(random_feature(batch_data_dict['only_source_feature']))

        loss_source_cla = loss_cla(output_s,batch_data_dict['only_source_target'])

        writer.add_scalar('only_source_loss', loss_source_cla, iteration)
        loss1 = loss_source_cla

        opt.zero_grad()
        loss1.backward()
        opt.step()

        #Train for domain

        feature_s1,feature_s2,feature_s3= model(batch_data_dict['source_feature'], return_feature=True)

        feature_t1,feature_t2,feature_t3= model(batch_data_dict['target_feature'], return_feature=True)

        if config.feature_maps==True:

            if iteration %200==0:
                source_figure = batch_data_dict['source_feature'][0].cpu().detach().numpy()
                target_figure = batch_data_dict['target_feature'][0].cpu().detach().numpy()

                plt.figure()

                ax1 = plt.subplot(2, 2, 1)
                ax2 = plt.subplot(2, 2, 2)
                ax3 = plt.subplot(2, 2, 3)
                ax4 = plt.subplot(2, 2, 4)

                plt.sca(ax1)
                librosa.display.specshow(source_figure.T)
                plt.title('source_image')
                #encoder1_image

                plt.sca(ax3)
                #librosa.display.specshow(librosa.power_to_db(torch.mean(model.encoder2(model.decoder(model.encoder1(x_input))[0,:,:,:]),dim=1)[0].cpu().detach().numpy()).T)
                librosa.display.specshow(torch.mean(model(batch_data_dict['source_feature'],return_feature=True)[0],dim=0).cpu().detach().numpy().T)
                plt.title('source_feature')

                plt.sca(ax4)
                librosa.display.specshow(torch.mean(model(batch_data_dict['target_feature'],return_feature=True)[0],dim=0).cpu().detach().numpy().T)
                plt.title('target_feature')

                plt.sca(ax2)
                librosa.display.specshow(target_figure.T)
                plt.title('target_image')

                plt.savefig('./image_result/{}.png'.format(str(iteration)))


        output_t,cla1,cla2 = model(batch_data_dict['target_feature'])

        loss_feat_dal_s_t1 = loss_adv(feature_t1,feature_s1)+loss_adv(feature_t2,feature_s2)+loss_adv(feature_t3,feature_s3)


        loss_t = loss_cla(output_t, batch_data_dict['target'])
        loss_feat_dal_s_t = loss_feat_dal_s_t1
        loss2 = loss_t  + loss_feat_dal_s_t


        opt.zero_grad()
        loss2.backward()
        opt.step()


        writer.add_scalar('lr',opt.param_groups[0]['lr'],iteration)

        if index >30000:
            break

        iteration+=1



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    train()


