#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
Olivia Lang, ol772690, olivia.x.lang@gsk.com, olang@seas.upenn.edu
Undergraduate Student Intern, Computational Biology Department, Summer 2018

Adapted from Variational Auto-Encoder code provided by:
Jin Yao
Computational Biologist
Target Sciences
GSK
jin.8.yao@gsk.com

This (Olivia's) code slightly modified by:
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

#%% 
'''''''''''''''''''''''''''''
AUTOENCODER CLASS

Includes functionality likely 
to be shared across many types
of autoencoders.

Adapted by Olivia Lang from code by Jin Yao
'''''''''''''''''''''''''''''

import os
import json
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.layers import Input, Activation, Dense, Lambda
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

class TimedStopping(Callback):
    '''Stop training when enough time has passed.
    # Arguments
        minutes: maximum time before stopping.
        verbose: verbosity mode.
    '''
    def __init__(self, minutes=None, verbose=0):
        super(Callback, self).__init__()
        
        self.start_time = 0
        self.minutes = minutes
        self.verbose = verbose
        
    def on_train_begin(self, logs={}):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs={}):
        if (time.time() - self.start_time)/60.0 > self.minutes:
            self.model.stop_training = True
            if self.verbose:
                print('Stopping after {0!s} minutes.'.format(self.minutes))

class AE():


    def __init__(self,num_layers=5, learning_rate=0.01, batch_fraction=0.1, 
                     epochs=100, minutes=10080, latent_dim=2, activation='tanh', data_split=0.2, 
                     dim_scaling = "log", init_h_dim = 2000,
                     h_dims = None, model_type='Gaussian',
                     save_folder="../",
                     meta_file="../raw_data/TCGA/TCGA_meta.csv.gz", 
                     data_file="../raw_data/TCGA/TCGA_nanostring.csv.gz"):
                         
        # NETWORK ARCHITECTURE
        self.num_layers=int(num_layers)
        self.latent_dim=int(latent_dim)
        self.init_h_dim=int(init_h_dim)
        self.dim_scaling=dim_scaling
        self.h_dims=self.get_h_dims(h_dims)
        self.activation=activation
        
        # DATA SPECIFICATIONS
        self.meta_file=meta_file
        self.data_file=data_file
        self.model_type=model_type
        self.data_split = float(data_split)
        self._save_folder = save_folder if save_folder[-1] == '/' else save_folder + '/'
        self._train_hist = pd.DataFrame()
        self._data_train, self._data_valid, self._data, self._meta, self._df = self.load_data()
        
        # RUNNING PARAMETERS
        self.learning_rate=float(learning_rate)
        self.batch_size=max(1, int(batch_fraction*self._data_train.shape[0]))
        self.epochs=int(epochs)
        self.minutes = minutes
        
    
    def build_model(self):
        raise NotImplementedError("The build_model() method is not implemented in the parent class.")


    def get_h_dims(self, h_dims = None):
        ''' Gets dimensions of hidden layers of the autoencoder.
        
        If 'h_dims' is specified as an input, this function will return that array.
        Otherwise, the function computes either linearly decreasing or logarithmically 
        decreasing layer sizes as a function of the autoencoder's dimension scaling 
        (dim_scaling), latent dimension (latent_dim) and initial hidden dimension (init_h_dim).
        
        Example One:
            Input:
                h_dims = None
                self.num_layers = 4
                self.latent_dim = 1
                self.init_h_dim = 10,000
                self.dim_scaling = "log"
            Output:
                [10000, 1000, 100, 10]
        
        Example Two:
            Input:
                h_dims = None
                self.num_layers = 4
                self.latent_dim = 1
                self.init_h_dim = 1,000
                self.dim_scaling = "linear"
            Output:
                [1000, 750, 500, 250]
                
        Example Three:
            Input:
                h_dims = [56,90,14,30]
                self.num_layers = 4
                self.latent_dim = 1
                self.init_h_dim = 1,000
                self.dim_scaling = "linear"
            Output:
                [56,90,14,30]
        '''
        
        # Case: h_dims specified as input parameter
        print("Start")
        print(h_dims)
        if h_dims is not None:
            self.num_layers = len(h_dims)
            self.init_h_dim = h_dims[0]
            print ("Yay it used h_dims!")
            return h_dims
        
        dim_func = lambda x : x #dummy to be reset
        
        # Dimension Function is Linear
        if self.dim_scaling == "linear":
            dim_func = lambda x : int((self.latent_dim - self.init_h_dim) \
                        / (self.num_layers + 1) * x + self.init_h_dim)
        # Dimension Function is Logarithmic
        if self.dim_scaling == "log":
            dim_func = lambda x : int(self.init_h_dim * \
                                      (self.latent_dim / self.init_h_dim) \
                                      ** (x / self.num_layers))
        h_dims = [dim_func(x) for x in range(0,self.num_layers)]
        print(h_dims)
        return h_dims
                
                
    def load_data(self):
        ''' Loads and splits data from data_file and meta_file
        If model_type is Bernoulli, then scales data to [0,1] for binary loss.
        If model_type is Gaussian, then linearly scales the data between one 
            and zero for Gaussian loss.
        input = 
            Autoencoder object (self)
        output = 
            data_train = matrix of training data
            data_valid = matrix of validation data
            data = entire data matrix
            meta = meta data as pandas dataframe
            data_df = data as pandas dataframe
        '''
        data_df = pd.read_csv(self.data_file, index_col=0)
        meta = pd.read_csv(self.meta_file, low_memory=False, index_col=0)
            
        data = data_df.as_matrix()
        
        # Standarized data
        if(self.model_type=='Bernoulli'):
            data = MinMaxScaler().fit_transform(data) # Scale to [0,1] for binary loss
        else:
            data = StandardScaler().fit_transform(data) # standardization for gaussian loss
                
        ## Select a subset data to train
        data_train, data_valid = train_test_split(data, test_size=self.data_split)
        return data_train, data_valid, data, meta, data_df


    def ae_fit(self):
        ''' Fits the model to the training data. Also saves training history
            to unique folder.
        '''
        start = time.clock()
#        early_stopping = EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=500)
        timed_stopping = TimedStopping(minutes=self.minutes, verbose=True)
        model_checkpoint = ModelCheckpoint(filepath=self._save_folder+'best_model.h5', monitor='val_loss', save_best_only=True, period=100)
        result = self._model.fit(self._data_train, self._data_train, shuffle=True,
                                   epochs=self.epochs, batch_size=self.batch_size,
                                   validation_data=(self._data_valid, self._data_valid), callbacks=[early_stopping, timed_stopping, model_checkpoint])
        trained = time.clock()
        self._train_hist = pd.DataFrame(result.history)
        self._training_time = trained-start


    def ae_evaluate(self):
        ''' Fits and evaluates model based on criteria specified in model building
        '''
        self.ae_fit()
        self.evaluation = self._model.evaluate(self._data_valid, 
                                           self._data_valid,
                                           batch_size=self.batch_size)
    
    
    def delete_weights(self):
        os.remove(self._save_folder+'best_model.h5')
    
    
    # Function to sae the evaluation
    def save_evaluation(self):
        '''Saves evaluation metrics to json
        '''
        with open(self._save_folder+"output.json", 'w') as fw:
#            json.dump({k:v for k,v in zip(self.metrics_names, self.evaluation)}, fw, indent=2)
            json.dump(self.evaluation, fw, indent=2)


    # Function to save the training history
    def save_training(self):
        ''' Saves training data to csv in unique folder
        '''
        self._train_hist = self._train_hist.assign(learning_rate=self.learning_rate)
        self._train_hist = self._train_hist.assign(batch_size=self.batch_size)
        self._train_hist = self._train_hist.assign(epochs=self.epochs)
        # TODO: I took out this part about the seed. 
        #self._train_hist = self._train_hist.assign(seed=seed)
        self._train_hist = self._train_hist.assign(train_mins="%.2f"%((self._training_time)/60))
        self._train_hist.to_csv(self._save_folder+"trainLog.tsv", sep='\t')
        
    
    
    def save_and_plot_results(self):
        self._set_figure_style()
        print ("Saving Evaluation Data...")
        self.save_evaluation()
        print ("Saving Training Data...")
        self.save_training()
        print ("Plotting Training Data...")
        self.plot_training()
        print ("Plotting Learned Weights...")
        self.plot_learned_weights()
        print ("Saving the Latent Embedding...")
        self.save_latent_embedding()
        print ("Plotting the Latent Embedding...")
        self.plot_latent_embedding()
        
    
    def save_latent_embedding(self):
        ''' Saves the latent embedding to compressed file named embedding.csv.gz.
        Includes associated metadata.
        '''
        X_encoded = self._encoder.predict(self._data, batch_size=self.batch_size)
        
        # Run dimensional reduction to 2D if latent features size larger than 2
        if X_encoded.shape[1] > 2:
            pca = decomposition.PCA(n_components=2)
            X_encoded = pca.fit_transform(X_encoded)
        
        X_encoded = pd.DataFrame(X_encoded, columns=['Latent1', 'Latent2'], index=self._df.index)
        X_encoded['method'] = np.repeat('DeepVAE-H'+str(self.num_layers)+'L'+str(self.latent_dim), repeats=X_encoded.shape[0])
        
        encoded_df = X_encoded
        
        # Merge with meta data
        self._encoded_df = encoded_df.join(self._meta, how='left')
        
        ## Save latent embedding
        self._encoded_df.to_csv(self._save_folder+'embedding.csv.gz', compression='gzip', index=True)
        
    def plot_latent_embedding(self):
        '''plots the latent embedding and saves it as encoded2D.png
        '''
        plt.figure(figsize=(10,10))
        plt.scatter(x=self._encoded_df['Latent1'], 
                    y=self._encoded_df['Latent2'], 
                    marker='o',
                    s = 9,
                    alpha=0.7)
        plt.xlabel("Loss = " + str(self.evaluation[0]))
        plt.savefig(self._save_folder+'encoded2D.png', bbox_inches='tight', dpi=200)
    
    @staticmethod
    def _set_figure_style():
        '''Sets figure style
        '''
        plt.style.use('ggplot')
        sns.set(style="white", color_codes=True)
        sns.set_context("poster", rc={"font.size":14,
                                     "axes.titlesize":15,
                                     "axes.labelsize":20,
                                     'xtick.labelsize':14,
                                     'ytick.labelsize':14})
            
    def plot_training(self):
        ''' Plots the training curve with the name training_plot.png
        '''
        plt.figure(figsize=(8,8))

        # VAE loss
        plt.subplot(3,1,1)
        plt.plot(self._train_hist['loss'][2:])
        plt.plot(self._train_hist['val_loss'][2:])
        plt.legend(['training', 'validation'])
        plt.ylabel('Loss')
        # Reconstruction loss
        plt.subplot(3,1,2)
        plt.plot(self._train_hist['recon_loss'][2:])
        plt.plot(self._train_hist['val_recon_loss'][2:])
        plt.legend(['training', 'validation'])
        plt.ylabel('Reconstruction loss')
        # KL divergence
        plt.subplot(3,1,3)
        plt.plot(self._train_hist['KL_loss'][2:])
        plt.plot(self._train_hist['val_KL_loss'][2:])
        plt.legend(['training', 'validation'])
        plt.xlabel('Epochs')
        plt.ylabel('KL divergence')
        
        # save figure
        plt.savefig(self._save_folder+'training_plot.png', bbox_inches='tight', dpi=200)
        
    def plot_learned_weights(self):
        ''' Plots the learned weights and saves them to file
        learned_weights_histogram.png. 
        '''
        # Extract the weights from the encoder and decoder model
        encoder_wts = []
        for l in self._encoder.layers:
            encoder_wts.append(l.get_weights())
        decoder_wts = []
        for l in self._decoder.layers:
            decoder_wts.append(l.get_weights())
        
        plt.figure(figsize=(15, 8))
        plt.rc('xtick', labelsize=10) 
        plt.rc('ytick', labelsize=10)
        
        # For every layer
        for n in range(self.num_layers+1):
            # ENCODER
            # Each layer weight and bias are stored in one list, weight is the 1st
            weight = encoder_wts[n*3+1][0]
            plt.subplot(2,self.num_layers+1,n+1);
            plt.hist(weight.flatten(), 50);
            plt.title("%d->%d" % (weight.shape[0], weight.shape[1]), fontsize=10)
            plt.ticklabel_format(axis='y', style='sci',  scilimits=(-1,1))

            # DECODER
            # Each layer weight and bias are stored in two list of decoder weights, weight is the 1st
            weight = decoder_wts[1][n*2] 
            plt.subplot(2,self.num_layers+1,n+1+self.num_layers+1);
            plt.hist(weight.flatten(), 50);
            plt.title("%d->%d" % (weight.shape[0], weight.shape[1]), fontsize=10)
            plt.ticklabel_format(axis='y', style='sci',  scilimits=(-1,1))
        plt.savefig(self._save_folder+'learned_weights_histogram.png', bbox_inches='tight', dpi=200)
        



#%% 
'''''''''''''''''''''''''''''
VARIATIONAL AUTOENCODER CLASS

Extends autoencoder class to 
build VAE model

Written by Jin Yao
Copy and Pasted with minor 
changes by Olivia Lang
'''''''''''''''''''''''''''''

class VAE(AE):
    
    def __init__(self, num_layers=5, learning_rate=0.01, batch_fraction=0.1, 
                     epochs=100, minutes=10080, latent_dim=2, activation='tanh', 
                     model_type='Gaussian', data_split=0.2, epsilon_std = 1.0, 
                     adam_epsilon = 0.001, dim_scaling = "log", init_h_dim = 2000, # TODO should init h_dim be equal to the number of obs?
                     h_dims = None,
                     save_folder="../",
                     meta_file="../raw_data/TCGA/TCGA_meta.csv.gz", 
                     data_file="../raw_data/TCGA/TCGA_nanostring.csv.gz"):
        super().__init__(num_layers=num_layers, learning_rate=learning_rate, batch_fraction=batch_fraction,
                   epochs=epochs, minutes=minutes, latent_dim=latent_dim, activation=activation, data_split=data_split, model_type=model_type,
                   dim_scaling=dim_scaling, init_h_dim=init_h_dim, h_dims=h_dims, save_folder=save_folder,meta_file=meta_file,
                   data_file=data_file)
        
        self.epsilon_std = float(epsilon_std)
        self.adam_epsilon = float(adam_epsilon)
        self._model = self.build_model()
    
    def build_model(self): 
        n_features = self._data_train.shape[1] #input layer size
        # LOSS FUNCTIONS -- 
        # TODO: put these in new file
        # Loss function (binary data or data scaled to [0,1])
        def vae_loss_binary(x, x_decoded_mean):
                # loss is averaged across features, mutiple # of features to scale back 
                xent_loss = n_features * metrics.binary_crossentropy(x, x_decoded_mean)  
                kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - K.exp(z_log_var_encoded), axis=-1)
                return K.mean(xent_loss + kl_loss)
        def recon_loss_binary(x, x_decoded_mean):
                # loss is averaged across features, mutiple # of features to scale back
                xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
                return K.mean(xent_loss)
        
        # Loss function (real data)
        def vae_loss_gaussian(x, x_decoded_mean):
                # loss is averaged across features, mutiple # of features to scale back
                xent_loss = n_features * metrics.mean_squared_error(x, x_decoded_mean)
                kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - K.exp(z_log_var_encoded), axis=-1)
                return K.mean(xent_loss + kl_loss)
        def recon_loss_gaussian(x, x_decoded_mean):
                # loss is averaged across features, mutiple # of features to scale back
                xent_loss = metrics.mean_squared_error(x, x_decoded_mean)
                return K.mean(xent_loss)
        
        def KL_loss(x, x_decoded_mean):
                kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - K.exp(z_log_var_encoded), axis=-1)
                return K.mean(kl_loss)
            
        # ACTUAL MODEL
        # TODO -- likely worthwile to put this into its own function
        # ~~~~~~~~~~~~~~~~~~~~~~
        # ENCODER
        # ~~~~~~~~~~~~~~~~~~~~~~
        #
        ## Input layer
        
        x = Input(shape=(n_features, ))
        h = x # First layer will be used to make subsequent
        #Encoder
        for n in range(self.num_layers):
            # Hidden layers
            h_dense_linear = Dense(self.h_dims[n])(h)
            h_dense_batchnorm = BatchNormalization()(h_dense_linear)
            h = Activation(self.activation)(h_dense_batchnorm)
        
        # Latent mean
        z_mean_dense_linear = Dense(self.latent_dim)(h)
        z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
        z_mean_encoded = z_mean_dense_batchnorm
        
        
        # Latent variance
        z_log_var_dense_linear = Dense(self.latent_dim)(h)
        z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
        z_log_var_encoded = z_log_var_dense_batchnorm
        
        # return the encoded and randomly sampled z vector
        z = Lambda(self.sampling, output_shape=(self.latent_dim, ))([z_mean_encoded, z_log_var_encoded])
        
        # Encoder model encoder from inputs to latent space
        self._encoder = Model(x, z_mean_encoded) 
        
        # TODO -- likely worthwile to put this into its own function
        # ~~~~~~~~~~~~~~~~~~~~~~
        # DECODER
        # ~~~~~~~~~~~~~~~~~~~~~~
        ##Decoder
        decoder_model = Sequential()
        for n in range(self.num_layers):
            ind = self.num_layers-1-n
            if ind==self.num_layers-1: # first layer after latent layer
                decoder_model.add(Dense(self.h_dims[ind], activation=self.activation, input_dim=self.latent_dim))
            else:
                decoder_model.add(Dense(self.h_dims[ind], activation=self.activation))
        
        # Last layer
        if(self.model_type=='Bernoulli'):
            decoder_model.add(Dense(n_features, activation='sigmoid'))
        else:
            decoder_model.add(Dense(n_features))
        
        expression_reco = decoder_model(z)
        
        ## Decoder model 
        # Decoder build a generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim, ))  # can generate from any sampled z vector
        _x_decoded_mean = decoder_model(decoder_input)
        self._decoder = Model(decoder_input, _x_decoded_mean)
        
        # TODO -- likely worthwile to put this into its own function
        # ~~~~~~~~~~~~~~~~~~~~~~
        # CONNECTIONS
        # ~~~~~~~~~~~~~~~~~~~~~~
        adam = Adam(lr=self.learning_rate, epsilon=self.adam_epsilon, amsgrad=True)
        vae = Model(x, expression_reco)
        if(self.model_type=='Bernoulli'):
            recon_loss = recon_loss_binary
            recon_loss.__name__ = 'recon_loss'
            vae.compile(optimizer=adam, loss=vae_loss_binary, metrics=[recon_loss, KL_loss])
        else:
            recon_loss = recon_loss_gaussian
            recon_loss.__name__ = 'recon_loss'
            vae.compile(optimizer=adam, loss=vae_loss_gaussian, metrics=[recon_loss, KL_loss])
        return vae
        
    # Function for reparameterization trick to make model differentiable
    def sampling(self,args):
    
        # Function with args required for Keras Lambda function
        z_mean, z_log_var = args
    
        # Draw epsilon of the same shape from a standard normal distribution
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                                  stddev=self.epsilon_std)
    
        # The latent vector is non-deterministic and differentiable
        # in respect to z_mean and z_log_var
        z = z_mean + K.exp(z_log_var / 2) * epsilon
        return z
        
    # VAE RUNNING FUNCTION
    # Note: don't need to propogate defaults through all of the __init__ functions since it is done here
    # You can refactor that out.
    @staticmethod
    def run(num_layers=5, learning_rate=0.01, batch_fraction=100,
            epochs=100, minutes=10080, latent_dim=2, activation='tanh', 
            model_type='Gaussian', data_split=0.2, epsilon_std = 1.0, 
            adam_epsilon = 0.001, dim_scaling = "log", init_h_dim = 2000, 
            h_dims=None, save_folder="../",
            meta_file="../raw_data/TCGA/TCGA_meta.csv.gz", 
            data_file="../raw_data/TCGA/TCGA_nanostring.csv.gz", keep_weights=True):
        # initialize, build and train the model
        _vae = VAE(num_layers=num_layers, learning_rate=learning_rate, batch_fraction=batch_fraction,
                   epochs=epochs, minutes=minutes, latent_dim=latent_dim, activation=activation, 
                   model_type=model_type, data_split=data_split, epsilon_std=epsilon_std,
                   adam_epsilon=adam_epsilon, dim_scaling=dim_scaling, init_h_dim=init_h_dim,
                   h_dims=h_dims,save_folder=save_folder,
                   meta_file=meta_file,
                   data_file=data_file) 
        # evaluates the model on test set
        _vae.ae_evaluate()
        # make figures and files
        _vae.save_and_plot_results()
        # delete weights
        if not keep_weights:
            _vae.delete_weights()
