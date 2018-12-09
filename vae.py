#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Autoencoder

Some ideas were taken from: https://github.com/wohlert/semi-supervised-pytorch
https://github.com/timbmg/VAE-CVAE-MNIST
https://github.com/bhpfelix/Variational-Autoencoder-PyTorch


@author: alex
"""

import math
import time

import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.nn import BCEWithLogitsLoss
#from layers import GaussianSample, GaussianMerge, GumbelSoftmax
#from inference import log_gaussian, log_standard_gaussian

import visualizations
import logger as tensorboard_logger

# Distribution needed to sample the KL, maybe later put it into a separate file ->
def log_gaussian_density(samples, mu=None, sigma=None, unit_normal=False):
    """
    Returns the log pdf of a multivariate normal distribution parametrised
    by mu and sigma. It is assumed that the distribution has diagonal covariance matrix with
    sigma on diagonal.
    
    Inputs:
    --------------------
        samples: tensor ( minibatch_size, samples_num, D)
        mu, sigma: tensor ( minibatch size, D)
    
    log_var evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    
    D = samples.shape[-1] # dimensionality of latent space
    #import pdb; pdb.set_trace()
    if (mu is None or sigma is None):
        if not (unit_normal == True):
            raise ValueError("Not unit normal and None values in parameters")
        else:
            log_pdf = -0.5*samples**2 
            log_pdf = torch.sum( log_pdf, dim=-1) -0.5*D*math.log( 2 * math.pi ) # sum over dimensions. Log pi term is outside of sum over dimensions
    else:    
        mu = mu.unsqueeze(1) # add utit ddimension 1
        sigma = sigma.unsqueeze(1) # add utit ddimension
        
        #log_pdf = -0.5*torch.sum( sigma*(samples - mu)**2, dim=-1) -0.5*D*math.log( 2 * math.pi ) - 0.5 * torch.sum( torch.log( sigma ), dim=-1)
        #log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
        log_pdf = -0.5 *(samples - mu)**2 / sigma**2  - 0.5 * torch.log( sigma**2 )
        log_pdf = torch.sum( log_pdf, dim=-1) - 0.5*D*math.log( 2 * math.pi ) # sum over dimensions. Log pi term is outside of sum over dimensions
        
    return log_pdf

# Distribution needed to sample the KL, maybe later put it into a separate file <-

class Gaussian_Reparametrization_Layer(nn.Module):
    """
    Layer which implements Gaussian Reparametrization
        and generates samples from a Gaussian distribution.
    """
    def __init__(self, in_dim, out_dim, p_num_samples=1, std_to_positive_transform='softplus', p_weights_init_type='xavier_normal'):
        super(Gaussian_Reparametrization_Layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        #import pdb; pdb.set_trace()
        
        self.mu = nn.Linear(in_dim, out_dim)
        self.sigma_lin = nn.Linear(in_dim, out_dim)
        self.num_samples = p_num_samples
        
        self.std_to_positive_transform = std_to_positive_transform
    
        if p_weights_init_type is not None:
            self.init_weights(p_weights_init_type)
            
    def init_weights(self, p_init_type='xavier_normal'):
        initialize_weights( (self.mu, self.sigma_lin) , p_init_type)
    
    def reparametrize(self, mu, sigma, p_num_samples=1):
        """
        Base stochastic layer that uses the
        reparametrization trick [Kingma 2013]
        to draw a sample from a distribution
        parametrised by mu and standard deviation.
   
    
        Gaussian based reparametrization trick
        """
        
        #import pdb; pdb.set_trace()
        
        epsilon = torch.randn( (mu.shape[0], p_num_samples, mu.shape[1] ), requires_grad=False ) # generate Gaussian random variable

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_sigma)
        #std = log_sigma.exp() # 

        # z = std * epsilon + mu
        z = mu.unsqueeze(1).addcmul(tensor1=sigma.unsqueeze(1), tensor2=epsilon) # posterior (of latent) is approximated as an uncorrelated multidimensional Gaussians.
                            # sigmas^2 - is the diagonal of covariance matrix.

        return z
    
    def forward(self, x):
        
        #import pdb; pdb.set_trace()
        
        mu = self.mu(x)
        
        if self.std_to_positive_transform == 'softplus':
            sigma = F.softplus(self.sigma_lin(x))
        elif self.std_to_positive_transform == 'exp':
            sigma = F.exp(self.sigma_lin(x))
        else:
            raise ValueError("Unknown positive transformation")
        
        return self.reparametrize(mu, sigma, self.num_samples), mu, sigma
# Layers (later to put into layers file) <-




# Layers initialization ->
def initialize_weights(layers, p_init_type='xavier_normal'):
        
        for ind, layer in enumerate(layers): 
            
            if p_init_type == 'xavier_uniform':
                init.xavier_uniform_(layer.weight.data)
                    
            elif p_init_type == 'xavier_normal':
                init.xavier_normal_(layer.weight.data)
            
            elif p_init_type == 'selu_paper':
                    fan_in = layer.weight.shape[0]; layer.weight.shape[1];
            
                    init.normal_(layer.weight.data, 0, math.sqrt( 1 / fan_in ))
                    
            layer.bias.data.zero_() # biases are initialized to zero
# Layers initialization <-
        
    
    
        
class Perceptron(nn.Module):
    """
    Multi-Layer Perceptron.
    """
    def __init__(self, dims, p_activation_fn=F.relu, p_output_activation=None, p_weights_init_type='xavier_normal', p_cuda=None):
        """
        Inputs:
        -------------
        dims: list
            [input_dim, output_dim_1, output_dim_2, ... ]
        """
        super(Perceptron, self).__init__()
        self.dims = dims
        self.activation_fn = p_activation_fn
        self.output_activation = p_output_activation

        self.layers = nn.ModuleList(  [ nn.Linear(dim_in, dim_out) for (dim_in, dim_out) in zip(dims, dims[1:])] )

        if p_weights_init_type is not None:
            self.init_weights(p_weights_init_type)
        
        if p_cuda:
            self.cuda()
            
    def init_weights(self, p_init_type='xavier_normal'):
        initialize_weights(self.layers, p_init_type=p_init_type)
                    
    def forward(self, x):
        """
        Output activation is either self.output_activation or None.
        """
        #import pdb; pdb.set_trace()
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers)-1: # last layer
                if self.output_activation is not None:
                    x = self.output_activation(x)
                else:
                    pass # no output nonlinearity
            else:
                x = self.activation_fn(x)

        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, dims, activation_function=F.relu, p_num_samples=1, p_kl_type='analytic', p_total_sample_num=None ):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        Inputs:
        ----------------
        
        dims: list [x_dim, h_dim, z_dim], where h_dim is also list
            x, z and hidden dimensions of the networks
        
        p_num_samples: int
            Amount of samples in estimating of expectations.
        kl_type: string
            'analytic' or 'sampling'
            
        p_total_sample_num: int
            Total number of training samples. Sometimes is need for proper normalization of ELBO.
        
        """
        super(VariationalAutoencoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims #  h_dim must be a list e.g. [784,[100,], 10 ]
        self.z_dim = z_dim
        self.num_samples = p_num_samples # number of samples to compute the estimation of elbo
        
        self.activation_function = activation_function
        self.kl_type = p_kl_type

        #import pdb; pdb.set_trace()
        self.encoder = Perceptron( [x_dim,] + h_dim,  p_activation_fn=self.activation_function, p_output_activation=self.activation_function )
        self.sample_layer = Gaussian_Reparametrization_Layer( h_dim[-1], z_dim, p_num_samples = self.num_samples ) #in_dim, out_dim, std_to_positive_transform='softplus', p_weights_init_type='xavier_normal')
        self.decoder = Perceptron( [z_dim,] + list(reversed(h_dim)) + [x_dim,],  p_activation_fn=self.activation_function, p_output_activation=None ) 
            # No output activation. It will be applied later, depending on the output data type.
   
        #import pdb; pdb.set_trace()
        
    def _kl_sampling(self, latent_posterior_samples, mu1, sigma1):
        """
        Computes the KL-divergence between latent posterior and latent prior.
        It is computed by the same sampling mathod, in the same way as likelihood calculation.
        
        KL-divergence between prior p(z) and posterior q(z|x).
    
        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]
        """

        #import pdb; pdb.set_trace()
        log_q = log_gaussian_density(latent_posterior_samples, mu=mu1, sigma=sigma1, unit_normal=False)
        log_p = log_gaussian_density(latent_posterior_samples, mu=None, sigma=None, unit_normal=True)
        
        return log_q - log_p # (batch_size, num_samples)


    def _kl_analytic(self, mu, sigma):
        """
        Computes the KL-divergence between latent posterior and latent prior.
        It is computed analytically, using the fact that prior and posteriors are Gaussians.
        
        """
        #import pdb; pdb.set_trace()
        # https://arxiv.org/abs/1312.6114 (Appendix B)
        # KL(q[mu, sigma] ||  p[0,I])  = -0.5 * sum_d(1 + log(sigma_d^2) - mu_d^2 - sigma_d^2)
        
        kl = -0.5 * torch.sum( (1 + torch.log(sigma**2) - mu**2 - sigma**2), dim=-1)

        return kl # (batch_size,)
    
    def forward(self, xx ):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        #z, z_mu, z_log_var = self.encoder(x)
        #import pdb; pdb.set_trace()
        
        xx = self.encoder(xx)
        z_samples, mu, sigma = self.sample_layer(xx) # z_samples.shape = (batch_size, num_samples, latent_dim); mu and sigma.shape = (batch_size, latent_dim)
        
        #import pdb; pdb.set_trace()
        xx_recon = self.decoder(z_samples) # shape: (batch_size, num_samples, input_dim)
        return xx_recon, z_samples, mu, sigma

    def likelihood(self, xx_recon, xx_true):
        """
        Computes p(x| z). This function assumes Bernulli likelihood.
        Note that xx_recon should come without sigmoid transformation.
        This is done in order to to have output activation and loss function in one place.
        Moreover, in Pytorch there is a function which unifies these two things.
        
        
        Inputs:
        -------------------
            xx_recon: tensor (batch_size, num_samples, input_dim)
        """
        
        #import pdb; pdb.set_trace()
        
        minus_likelihood = BCEWithLogitsLoss(reduce=False ) #reduction='none' ) # call to this function produce samples (batch_size, num_samples, output_dim)
        
        xx_true = xx_true.unsqueeze(1).expand_as(xx_recon) 
        
        likelihood_out = torch.sum( -minus_likelihood(xx_recon, xx_true), dim=-1) # sum over output dimentionality
        
        return likelihood_out #(batch_size, num_samples)
        
    def ELBO(self, xx_recon, xx_true, z_samples, mu, sigma):
        """
        ELBO = the loss function to optimize
        """
        #import pdb; pdb.set_trace()
        
        # Sum over output dimensions and average over samples, result is (batch_size,)
        likelihood = torch.mean( self.likelihood(xx_recon, xx_true), dim=-1) 
        
        if self.kl_type == 'analytic':
            kl = self._kl_analytic(mu, sigma) # (batch_size,)
        elif self.kl_type == 'sampling':
            kl = torch.mean( self._kl_sampling(z_samples, mu, sigma), dim=-1 ) #Average over samples, result is (batch_size,)
        else:
            raise ValueError("kl_type is wrong.")
        
        #import pdb; pdb.set_trace()
        elbo = torch.mean( likelihood - kl, dim=-1) # average over minibatch
        return elbo
    
        #likelihood = torch.mean( likelihood, dim=-1)
        #kl = torch.mean( kl, dim=-1)
        #return likelihood, kl
            
    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)
    
    def encode(self, xx):
        
        xx = self.encoder(xx)
        _, mu, sigma = self.sample_layer(xx) # z_samples.shape = (batch_size, num_samples, latent_dim); mu and sigma.shape = (batch_size, latent_dim)
        
        return mu, sigma
    
    @staticmethod
    def log_var_and_grad_summaries(logger, layers, global_step, p_prefix, log_histograms=False):
        """
        Logs variable and grad stats for layer. Transfers data from GPU to CPU automatically
        :param logger: TB logger
        :param layers: param list
        :param global_step: global step for TB
        :param prefix: name prefix
        :param log_histograms: (default: False) whether or not log histograms
        :return:
        """
        for ind, ll in enumerate(layers):
            # Weights
            weights = ll.weight.data.cpu().numpy()
            biases = ll.bias.data.cpu().numpy()

            if isinstance(p_prefix, collections.Iterable):
                prefix = p_prefix[ind]
            else:
                prefix = p_prefix
                
            logger.scalar_summary("Weights/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(weights), global_step)
            logger.scalar_summary("Biases/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(biases), global_step)
            
            if log_histograms:
                logger.histo_summary(tag="Weights/{}_{}".format(prefix, ind), values=weights,
                                     step=global_step)
                logger.histo_summary(tag="Biases/{}_{}".format(prefix, ind), values=biases,
                                     step=global_step)
    
            # Gradients
            if global_step > 0:
                weights_grad = ll.weight.grad.data.cpu().numpy()
                biases_grad = ll.bias.grad.data.cpu().numpy()
                
                logger.scalar_summary("Weights_Grads/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(weights_grad), global_step)
                logger.scalar_summary("Biases_Grads/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(biases_grad), global_step)
                
                if log_histograms:
                    logger.histo_summary(tag="Weights_Grads/{}_{}".format(prefix, ind), values=weights_grad, step=global_step)
                    logger.histo_summary(tag="Weights_Grads/{}_{}".format(prefix, ind), values=biases_grad, step=global_step)
            
    def train_model(self, train_data, p_num_epochs, p_lr, p_eval_freq=10, eval_data=None, p_logdir=None, p_log_weights_hists=False):
        """
        
        Inputs:
        ------------------
        p_minibatch_size: int
        
        """
        
        if p_logdir is not None:
            logger = tensorboard_logger.Logger(p_logdir)
            logger.clean_log_folder(True)
            
        else:
            logger = None
        #model_checkpoint_path = p_logdir + "/model"
        #path_to_model = Path(model_checkpoint)
    
        if eval_data is None:
            eval_data = train_data
        
        #import pdb; pdb.set_trace()
        
        optimizer = optim.Adam(self.parameters(),
                               lr=p_lr ) #weight_decay=p_weight_decay)
        
        # Evaluate before training
        self.evaluate(eval_data, p_epoch_no=0, p_logger=logger, p_log_weights_hists=p_log_weights_hists)
        
        global_step = 0 # amount of steps
        for epoch in range(p_num_epochs):
            if epoch == 0:
                print('Training epoch {} of {}'.format(epoch, p_num_epochs))
            e_start_time = time.time()
            
            self.train() # put current model in the training mode
            total_epoch_loss = 0.0 # loss per sample for one epoch
            denom = 0.0
            for i, minibatch in enumerate(train_data):
                
                if next(self.parameters()).is_cuda:
                    minibatch.cuda()
                    
                true_x = minibatch[0]; true_y = minibatch[1];
                #import pdb; pdb.set_trace()
                optimizer.zero_grad()
                xx_recon, z_samples, mu, sigma = self(true_x) # forwar call
                
                #import pdb; pdb.set_trace()
                
                #likelihood,kl = self.ELBO(xx_recon, true_x, z_samples, mu, sigma)
                #loss = -(likelihood - kl)
                #print(likelihood.data, kl.data)
                
                elbo = self.ELBO(xx_recon, true_x, z_samples, mu, sigma)
                loss = -elbo
                #print(loss.data)
                
                loss.backward()
                optimizer.step()
                
                total_epoch_loss += loss.data
                denom += 1
                global_step += 1
                
                # print('[%d, %5d] loss( %s ): %.7f' % (epoch, i, l_name, l_value ) )#running loss
                if p_logdir is not None:
                    logger.scalar_summary("Running loss:", loss.data, global_step)
                    
                    
            e_end_time = time.time()
            l_value = total_epoch_loss / denom
            
            print('Epoch {0}. time: {1:.3f} s.,     TRAINING loss: {2:.5f}'
                  .format(epoch, e_end_time - e_start_time, l_value))
            
            if p_logdir is not None:
                logger.scalar_summary("Training_loss_per_epoch", l_value, epoch)
            #logger.scalar_summary("Epoch_time", e_end_time - e_start_time, epoch)
            
            # Compute evaluation costs ->
            if ((epoch > 0) and (epoch % p_eval_freq == 0)) or (epoch == p_num_epochs - 1):
                #import pdb; pdb.set_trace()
                self.evaluate(eval_data, p_epoch_no=epoch, p_logger=logger, p_log_weights_hists=p_log_weights_hists)
                self.train() # get back into the training mode
                
            # Compute evaluation costs <-
            
        print("The end!")
        # Save model
        #print("Saving model to {}".format(model_checkpoint_path + ".last"))
        #torch.save(self.state_dict(), model_checkpoint_path + ".last")
            
            
    def evaluate(self, eval_data, p_epoch_no=None, p_logger=None, p_log_weights_hists=False ):
        """
        
        """

        #print('Doing evaluation on epoch no. {}'.format(p_epoch_no))
        e_start_time = time.time()
        
        self.eval() # put current model in the evaluate mode
        total_epoch_loss = 0.0 # loss per sample for one epoch
        denom = 0.0
        for i, minibatch in enumerate(eval_data):
            
            #import pdb; pdb.set_trace()
            
            true_x = minibatch[0]; true_y = minibatch[1];
            
            if next(self.parameters()).is_cuda:
                    minibatch.cuda()
                    
            xx_recon, z_samples, mu, sigma = self(true_x) # forwar call
            
            loss = -self.ELBO(xx_recon, true_x, z_samples, mu, sigma)
            
            total_epoch_loss += loss.data
            denom += 1
                
        e_end_time = time.time()
        l_value = total_epoch_loss / denom
        
        print('Evaluation epoch {0}. time: {1:.3f} s.,     EVAL mean loss: {2:.5f}'
              .format(p_epoch_no, e_end_time - e_start_time, l_value))
        
        if p_logger is not None:
            p_logger.scalar_summary("Eval. loss:", l_value, p_epoch_no)
    
            self.log_var_and_grad_summaries(p_logger, self.encoder.layers, p_epoch_no, "Encoder", log_histograms=p_log_weights_hists)
            self.log_var_and_grad_summaries(p_logger, [self.sample_layer.mu, self.sample_layer.sigma_lin], 
                                            p_epoch_no, ["Reparam.MU", "Reparam.SIGMA"], log_histograms=p_log_weights_hists)
            self.log_var_and_grad_summaries(p_logger, self.decoder.layers, p_epoch_no, "Decoder",log_histograms=p_log_weights_hists)
            
            
    def save(self,):
        pass
    
    def load(self,):
        pass

    
def train_vae_mnist_1(p_batch_size=64):
    """
    
    """
    
    import datautils
    
    
    _, unlabelled_data, validation_data = datautils.get_mnist(location="./downloaded_datasets", batch_size=p_batch_size, labels_per_class=0) # seupervised data

    vae = VariationalAutoencoder([784, [256, 128], 32], activation_function=F.relu, p_num_samples=3, p_kl_type='sampling', p_total_sample_num=len(unlabelled_data) )
    
    vae.train_model(unlabelled_data, p_num_epochs=20, p_lr=3e-4, p_eval_freq=2, eval_data=validation_data, p_logdir='./test_log', p_log_weights_hists=True)
    
    def make_full_val_data(p_vae, p_validation_data):
        z_loc = None; classes = None
        for i, minibatch in enumerate(p_validation_data):
            true_x = minibatch[0]; true_y = minibatch[1];
            z_loc_part, z_scale = vae.encode(true_x)
            z_loc_part = z_loc_part.detach().cpu().numpy();
            if i==0:
                z_loc = z_loc_part
            else:
                z_loc = np.vstack((z_loc,z_loc_part))
                
            if i==0:
                classes = true_y
            else:
                classes = np.vstack((classes,true_y))
        return z_loc, classes
        
        
    z_loc, classes = make_full_val_data(vae, validation_data)
    #import pdb; pdb.set_trace()
        
    visualizations.plot_tsne(z_loc, classes, 'VAE_alex_1')
    
if __name__ == '__main__':
    
    train_vae_mnist_1()
    
    #pr = Perceptron([100,40,20,10], p_weights_init_type='selu_paper')


