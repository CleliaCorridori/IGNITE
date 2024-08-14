import numpy as np
import matplotlib.pyplot as plt

import lib.fun_asynch as fun_asynch
import lib.fun_plotting as fun_plotting

class asynch_reconstruction():
    """
    Maximum likelihood reconstruction of an asymmetric dynamical Ising model
    evolving according to asynchronous Glauber updates.

    This class wraps the functions defined in lib/fun_asynch.py that use numba.
    """
    def __init__(self, x, delta_t, LAMBDA, MOM = None, gamma = 1, opt = 'NADAM', reg = 'L1', ax_names = []):
        assert (opt == 'NADAM') or (opt == 'MOMENTUM'), 'The optimizer can be either NADAM or MOMENTUM'
        if opt == 'MOMENTUM':
            assert MOM != None, 'Specify the momentum variable MOM'
        assert (reg == 'L1') or (reg == 'L2'), 'The regularizer can be either L1 or L2'
        
        self.ax_names = ax_names
        
        self.delta_t = delta_t
        self.gamma = gamma
        self.optimizer = opt
        self.L1 = reg == 'L1'
        
        self.LAMBDA = LAMBDA


        self.Nvar = x.shape[0]
        self.Nsteps = x.shape[1]

        self.J = np.zeros((x.shape[0], x.shape[0]))
        # np.random.seed(0)
        # self.J = np.random.uniform(low=-1, high=1, size=(24, 24))
        
        # self.h = np.arctanh(np.mean(x, axis = 1))
        self.h = np.zeros(len(np.mean(x, axis = 1)))
        
        self.covariance = fun_asynch.nb_cov(x)
        self.dot_covariance = fun_asynch.nb_dot_cov(x, delta_t, self.covariance)
        
        self.likelihood = self.find_likelihood(x)

        if self.optimizer == 'MOMENTUM':            
            self.momentum = MOM
                        

                
    def reconstruct(self, x, Nepochs, start_lr = 1, drop = 0.99, edrop = 20):
        print('Epoch', '\t', 'Max J gradient', '\t', 'Max h gradient', '\t', 'Max change in J', '\t', 'Max change in h',
              '\t', 'Likelihood')

        if self.optimizer == 'MOMENTUM':
            self.h, self.J = fun_asynch.momentum_reconstruct(x, self.h, self.J, self.covariance,
                                                             self.dot_covariance, self.delta_t, Nepochs,
                                                             LAMBDA = self.LAMBDA, MOM = self.momentum, L1 = self.L1,
                                                             start_lr = start_lr, drop = drop, edrop = edrop)
            
        if self.optimizer == 'NADAM':
            self.h, self.J = fun_asynch.NADAM_reconstruct(x, self.h, self.J, self.covariance,
                                                          self.dot_covariance, self.delta_t, Nepochs,
                                                          LAMBDA = self.LAMBDA, L1 = self.L1,
                                                          start_lr = start_lr, drop = drop, edrop = edrop)

        self.plot_fields_and_couplings()
        self.likelihood = self.find_likelihood(x)
    
    def find_likelihood(self, x):
        theta = fun_asynch.find_theta(self.h, self.J, x)
        return fun_asynch.L_asynch(self.h, self.J, x, self.covariance, self.dot_covariance,
                                   theta, self.delta_t, self.gamma)
    
    def load_parameters(self, h, J, plot = False):
        self.h = h
        self.J = J

        if plot:
            self.plot_fields_and_couplings()


    def plot_fields_and_couplings(self, h = None, J = None, ret = False):
        if h is None:
            hplot = np.repeat(self.h[:, np.newaxis], 10, axis = 1)
        else:
            hplot = np.repeat(h[:, np.newaxis], 10, axis = 1)
        if J is None:
            J = self.J

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
        plt.subplots_adjust(wspace=0.3, hspace=0)

        imgh = fun_plotting.plotmat(hplot, ax[0])
        imgJ = fun_plotting.plotmat(J, ax[1])

        cbarh = fig.colorbar(imgh, ax = ax[0])
        cbarJ = fig.colorbar(imgJ, ax = ax[1])

        cbarh.set_label(r'$h_i$', rotation = -90, labelpad = 25, fontsize = 20)
        cbarJ.set_label(r'$J_{ij}$', rotation = -90, labelpad = 25, fontsize = 20)
        cbarh.ax.tick_params(labelsize = 14)
        cbarJ.ax.tick_params(labelsize = 14)

        ax[0].set_xticks([])
        ax[0].set_yticks(np.arange(0,np.shape(np.array(J))[0]))
        ax[0].set_yticklabels(self.ax_names, fontsize=12)
        
        ax[1].set_xlabel('Genes label', fontsize = 14, labelpad = 10)
        ax[1].set_xticks(np.arange(0,np.shape(np.array(J))[0]))
        ax[1].set_xticklabels(self.ax_names, rotation='vertical', fontsize=12)
        ax[1].set_yticks(np.arange(0,np.shape(np.array(J))[0]))
        ax[1].set_yticklabels(self.ax_names, fontsize=12)
        

        for i in range(2):
            ax[i].tick_params(labelsize = 14)
            ax[i].set_ylabel('Genes label', fontsize = 14, labelpad = 10)

        if ret:
            return fig, ax
        else:
            plt.show()

    def generate_samples(self, t_size = None, seed=1):
        if t_size == None:
            t_size = self.Nsteps
        return fun_asynch.generate_samples_asynch(self.h, self.J, self.delta_t,
                                                  gamma = self.gamma, Nsteps = t_size, seed = seed)
    # def generate_samples_init(self, t_size = None, seed=1, spins = np.array([])):
    #     if t_size == None:
    #         t_size = self.Nsteps
    #     return fun_asynch.generate_samples_asynch_init(self.h, self.J, self.delta_t,
    #                                               gamma = self.gamma, Nsteps = t_size, seed = seed, spins = spins)
    
    # def generate_samples_SetData(self, matx=np.array([]), field=np.array([]), t_size = None, seed=0):
    #     if t_size == None:
    #         t_size = self.Nsteps
    #     return fun_asynch.generate_samples_asynch(field, matx, self.delta_t,
    #                                               gamma = self.gamma, Nsteps = t_size, seed=seed)
