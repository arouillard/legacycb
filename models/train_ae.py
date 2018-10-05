# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import argparse
import json
import tensorflow as tf
from keras import backend as K
from autoencoders import VAE


def main(model_config_path):
    
    ae_name_to_object = {'vae':VAE}
    
    # load model_config
    print('loading model_config...', flush=True)
    print('model_config: {0}'.format(model_config_path), flush=True)
    with open(model_config_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        model_config = json.load(fr)
    print(model_config, flush=True)
    ae_name = model_config['ae_type'].lower()
    del model_config['ae_type']
    
    # configure tensorflow session
    print('configuring tensorflow session...', flush=True)
    session_config = tf.ConfigProto(allow_soft_placement=True) # can choose any available GPU
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.95
    K.set_session(K.tf.Session(config=session_config))
    
    # train autoencoder
    print('training autoencoder...', flush=True)
    ae = ae_name_to_object[ae_name]
    ae.run(**model_config)
    
    print('done train_ae.py', flush=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train autoencoder.')
    parser.add_argument('model_config_path', help='path to .json file with configurations for autoencoder', type=str)
    args = parser.parse_args()    
    main(args.model_config_path)
