"""Configuration for models."""
import os

import tensorflow as tf

CONFIG = {
    'dmn_plus': {
        'msvd_qa': {
            '0': {
                'model': {
                    'batch_size':32,
                    'pretrained_embedding': '/mnt/data/yuanyitian/videoQA/dmn/data/msvd_qa/data/word_embedding.npy',
                    'word_dim': 300,
                    'video_feature_dim': 4096,
                    'video_feature_num': 20,
                    'mfcc_dim':39,
                    'num_hops':3,
                    'vocab_num': 4000,
                    'answer_num': 1000,
                    'hidden_size': 256,
                    'cap_grads': False,
                    'noisy_grads': False,
                    'max_grad_val': 10,
                    'video_conv_size':7,
                    'video_conv5_3_size':14,
                    'video_conv_feature_dim':512,
                    'attentionIsall_hidden_size': 512,
                    'attentionIsall_num_blocks': 2,
                    'attentionIsall_num_heads': 8,
                    'attentionIsall_dropout_rate': 0.1
                },
                'train': {
                    'batch_size': 32,
                    'reg_coeff': 1e-6,
                    'learning_rate': 0.001,
                    'keep_prob': 0.5
                }
            }
        },
        'msrvtt_qa': {
            '0': {
                'model': {
                    'batch_size':64,
                    'pretrained_embedding': '/mnt/data/yuanyitian/videoQA/dmn/data/msrvtt_qa/data/word_embedding.npy',
                    'word_dim': 300,
                    'video_feature_dim': 4096,
                    'video_feature_num': 20,
                    'mfcc_dim':39,
                    'num_hops':3,
                    'vocab_num': 8000,
                    'answer_num': 1000,
                    'hidden_size': 256,
                    "attentionIsall_N": 6,
                    'cap_grads': False,
                    'noisy_grads': False,
                    'max_grad_val': 10,
                    'video_conv_size':7,
                    'video_conv5_3_size':14,
                    'video_conv_feature_dim':512,
                    'attentionIsall_hidden_size': 512,
                    'attentionIsall_num_blocks': 2,
                    'attentionIsall_num_heads': 8,
                    'attentionIsall_dropout_rate': 0.1
                },
                'train': {
                    'batch_size': 64,
                    'reg_coeff': 1e-7,
                    'learning_rate': 0.001,
                    'keep_prob': 0.5
                }
            }
        }
    }
}


def get(model, dataset, config_id, gpu_list):
    """Generate configuration."""
    config = {}
    if dataset == 'msvd_qa':
        config['preprocess_dir'] = '/mnt/data/yuanyitian/videoQA/dmn/data/msvd_qa/data'
    elif dataset == 'msrvtt_qa':
        config['preprocess_dir'] = '/mnt/data/yuanyitian/videoQA/dmn/data/msrvtt_qa/data'

    config['model'] = CONFIG[model][dataset][config_id]['model']
    config['train'] = CONFIG[model][dataset][config_id]['train']

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = gpu_list
    config['session'] = sess_config

    return config
