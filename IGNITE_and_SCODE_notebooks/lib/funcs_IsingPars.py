import numpy as np
from concurrent.futures import ProcessPoolExecutor

from lib.ml_wrapper import asynch_reconstruction
import lib.funcs_general as funcs_general
import lib.funcs_ko as funcs_ko

def grid_search_trial(args):
    """Function to run a single trial of the grid search. 
    """
    spins, par_sel, genes_order, interaction_list, thr, Norm, N_sim, cm_original_lN, noise_dist = args
    print(f"Params: {par_sel}")
    
    # Initialize and reconstruct the model
    model = asynch_reconstruction(spins, delta_t=1, LAMBDA=par_sel["LAMBDA"], MOM=par_sel["MOM"],
                                  opt=par_sel["opt"], reg=par_sel["reg"], ax_names=genes_order) 
    model.reconstruct(spins, Nepochs=par_sel["Nepochs"], start_lr=par_sel["lr"], 
                      drop=par_sel["drop"], edrop=par_sel["edrop"])
    
    # Evaluate the model computing the FCI
    tp_val, info_int = funcs_general.TP_check(interaction_list, model.J, 
                                              genes_order, 
                                              inferred_int_thr = thr, Norm = Norm)
    
    # Generate new spins and compute correlation matrices
    cm_sim_lN = np.zeros((spins.shape[0], spins.shape[0], N_sim))
    for ll in range(N_sim):
        spins_new_lN = model.generate_samples(seed=ll*2, t_size=spins.shape[1])
        cm_sim_lN[:, :, ll] = np.corrcoef(spins_new_lN)

    # cm_sim_lN_mean = np.nanmean(cm_sim_lN, axis=2)
    dist_val = np.mean([funcs_ko.sum_squared_abs_diff(cm_original_lN, cm_sim_lN[:, :, i]) for i in range(cm_sim_lN.shape[2])]) / noise_dist

    # if the values in each rows are constant there could be a nan in the correlation matrix
    if np.isnan(cm_sim_lN).sum() > (spins.shape[0]*spins.shape[0]*N_sim)-spins.shape[0]*spins.shape[0]*0.5*N_sim:
        dist_val = np.nan

    return model.J, tp_val, info_int, dist_val


def grid_search(spins, params, interaction_list, genes_order, Ntrials=5, seedSet=20961, Norm=True, thr=0.0, cm_original_lN=[], noise_dist=1, max_workers=1):
    """Function to run a grid search for the best set of hyperparameters for the Ising model.
    It uses the grid_search_trial function to run each trial in parallel.
    Args:
        spins (np.array): matrix of binary values (Ngenes x Nsamples)
        params (dict): dictionary with the hyperparameters to be tested
        interaction_list (list): list of interactions to be evaluated
        genes_order (list): list of gene names
        Ntrials (int): number of trials
        seedSet (int): seed for the random number generator
        Norm (bool): whether to normalize the interaction matrix
        thr (float): threshold for the interaction matrix
        cm_original_lN (np.array): correlation matrix of the original data
        noise_dist (float): average distance between correlation matrices of the shuffled data and the original data
        max_workers (int): number of workers for the ProcessPoolExecutor
        
    Returns:
        matx_sel (np.array): matrix of selected interactions (Ntrials x Ngenes x Ngenes)
        tp_val (np.array): true positive values (Ntrials)
        dist_val (np.array): distance values (Ntrials)
        info_int (np.array): information values (4 x Ninteractions x Ntrials)
    """
    np.random.seed(seedSet)

    # Prepare arguments for each trial
    trial_args = []
    for _ in range(Ntrials):
        par_sel = {key: np.random.choice(value) for key, value in params.items()}
        trial_args.append((spins, par_sel, genes_order, interaction_list, thr, Norm, Ntrials, cm_original_lN, noise_dist))
    
    # Initialize arrays to store results
    matx_sel = np.zeros((Ntrials, len(genes_order), len(genes_order)))
    info_int = np.zeros((4, len(interaction_list), Ntrials))
    tp_val = np.zeros(Ntrials)
    dist_val = np.zeros(Ntrials)

    # Use ProcessPoolExecutor to parallelize the grid search
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(grid_search_trial, trial_args))

    # Extract results from futures
    for ii, (model_J, tp, info, dist) in enumerate(results):
        matx_sel[ii, :, :] = model_J
        tp_val[ii] = tp
        dist_val[ii] = dist
        info_int[:,:,ii] = info

    return matx_sel, tp_val, dist_val, info_int


