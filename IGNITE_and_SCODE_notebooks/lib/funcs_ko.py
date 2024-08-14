import numpy as np
import matplotlib.pyplot as plt
import lib.fun_asynch as fun_asynch

# # -------------------------------------------------------------------------------------------------
# # ------------------------------------- Distance between matrices ---------------------------------
# # -------------------------------------------------------------------------------------------------
def sum_squared_abs_diff(a, b):
    # create boolean masks for non-NaN values
    mask_a = ~np.isnan(a)
    mask_b = ~np.isnan(b)

    # use masks to select non-NaN values in both arrays
    array1 = a[mask_a & mask_b]
    array2 = b[mask_a & mask_b]
    """Calculate the sum of squared absolute differences between two matrices"""
    diff = (array1.flatten()-array2.flatten())**2
    return np.sqrt(np.sum(diff))

# -------------------------------------------------------------------------------------------------
# ------------------------------------- KO: knockout  functions -----------------------------------
# -------------------------------------------------------------------------------------------------

def KO_wrap(KO_gene, interactions_matx, Ising_Model, genes_order, wt_simulated_spins, df_KO_N24_Leeb_cl,  
            N_test_KO):
    """ Wrapper function to compute the difference between the average activity of the KO and the WT and compare it to experimental log2FC
    Args:
        KO_gene (string): name of the gene to remove
        interactions_matx (numpy array): interaction matrix
        Ising_Model: IGNITE model
        genes_order (list of strings): list of genes in the order of the interaction matrix
        wt_simulated_spins (numpy array): simulated spins in time for the wild type
        df_KO_N24_Leeb_cl (pandas dataframe): dataframe containing the logFC of the experimental data
        
    Output:
        - diff_R (array): difference between the average activity of the KO and the WT
        - exp_logFC (array): experimental logFC without the KO gene
        - KO_genes_order_R (list of strings): list of genes in the order of the interaction matrix without the KO gene"""
    # KO info
    KO_pN_rec_matx_R, KO_pN_rec_field_R, KO_gene_idk_R, KO_genes_order_R = info_KO(interactions_matx, Ising_Model, KO_gene,
                                                                                genes_order=genes_order)

    #Compute Experimental Log2FC (Leeb/single KO or Kalkan/Triple KO)
    exp_logFC = np.delete(np.array(df_KO_N24_Leeb_cl[KO_gene]), KO_gene_idk_R) 

    # wt from simulated spins in time without the KO gene
    wt_pN_mb_pst_spins_forKO_R = np.delete(np.array(wt_simulated_spins),KO_gene_idk_R, axis=0)

    # average activity 
    KO_pN_mb_pst_avg_R, KO_pN_mb_pst_std_R, wt_pN_mb_pst_avg_R, wt_pN_mb_pst_std_R, _ = KO_avg_weighted(KO_pN_rec_matx_R, KO_pN_rec_field_R, 
                                                                                                    wt_pN_mb_pst_spins_forKO_R, 
                                                                                                    N_test_KO=N_test_KO, N_time =9547)

    # Compute difference between KO and WT and its error, compare with Experimental data
    diff_R = KO_diff_sim(KO_pN_mb_pst_avg_R, KO_pN_mb_pst_std_R, wt_pN_mb_pst_avg_R, wt_pN_mb_pst_std_R)
    return(diff_R, exp_logFC, KO_genes_order_R)

def info_KO(matx, model, KO_gene="Rbpj", genes_order=[], multiple=False):
    """Remove the KO_gene from the interaction matrix and from the field
    Args:
        matx (numpy array): interaction matrix
        model: IGNITE model
        KO_gene (string): name of the gene to remove
        genes_order (list of strings): list of genes in the order of the interaction matrix
        multiple (bool): True if more than one gene is removed (For Triple KO in our case)
        
    Output:
        - KO_rec_matx (numpy array): interaction matrix without the KO gene
        - KO_rec_field (numpy array): field h without the KO gene
        - KO_gene_idk (int): index of the KO gene
        - KO_genes_order (list of strings): list of genes in the order of the interaction matrix without the KO gene"""
    if multiple:
        KO_gene_idk = [np.where(genes_order == KO_gene[i])[0][0]  for i in range(len(KO_gene))]
    else:
        KO_gene_idk = np.where(genes_order == KO_gene)[0] 
    # Adjacency matrix without KO gene
    KO_rec_matx = np.delete(matx, KO_gene_idk, axis=0)
    KO_rec_matx = np.delete(KO_rec_matx, KO_gene_idk, axis=1)
    # field without KO gene
    KO_rec_field = np.delete(model.h, KO_gene_idk, axis=0)
    # genes array without KO gene
    KO_genes_order = np.delete(genes_order, KO_gene_idk, axis=0)
    return(KO_rec_matx, KO_rec_field, KO_gene_idk, KO_genes_order)


def KO_avg_weighted(matx, field, wt_spins, N_test_KO=10, N_time =9547):
    """ Compute the average spins value for each gene in KO and in WT dataset
    Args:
        matx (numpy array): interaction matrix
        field (numpy array): field values
        wt_spins (numpy array): simulated spins in time for the wild type
        model: IGNITE model
        N_test_KO (int): number of times the simulation is performed
        N_time (int): number of time steps
        
    Output:
        - KO_avg (array): average activity of the KO
        - KO_std (array): standard deviation of the KO
        - wt_avg (array): average activity of the WT
        - wt_std (array): standard deviation of the WT
        - KO_spins (numpy array): activity of the GENERATED data, array of size (n_genes, n_timesteps, n_simulations)
    """
    # activity for each gene in KO
    KO_spins = np.zeros((matx.shape[0], N_time, N_test_KO))
    for i in range(N_test_KO):
        # generate KO data
        KO_spins[:,:,i] = fun_asynch.generate_samples_asynch(field, matx, 1,
                                                  gamma = 1, Nsteps = N_time, seed=i*5)
    KO_avg = KO_spins.mean(axis=(1,2))
    KO_std = KO_spins.std(axis=(1, 2), ddof=1) / np.sqrt(N_test_KO)

    # Compute WT averages and standard deviations
    wt_avg = wt_spins.mean(axis=(1, 2))
    wt_std = wt_spins.std(axis=(1, 2), ddof=1) / np.sqrt(wt_spins.shape[2])
    return(KO_avg, KO_std, wt_avg, wt_std, KO_spins)

def KO_diff_sim(KO_avg, KO_std, wt_avg, wt_std, thr_significance=0, additional_analysis=False):
    """ Compute the differences between the average activity of the KO and the WT
    Args:
        - KO_avg (array): average activity of the KO
        - KO_std (array): standard deviation of the KO
        - wt_avg (array): average activity of the WT
        - wt_std (array): standard deviation of the WT
        
    Output:
        - diff_sim (array): difference between the average activity of the KO and the WT
        - diff_sim_std (array): standard deviation of the difference
    """
    diff_sim = KO_avg - wt_avg
    diff_sim_std = np.sqrt(KO_std**2 + wt_std**2)

    if additional_analysis:
        # Vectorized significance check
        not_significant = np.where(np.abs(diff_sim) < thr_significance * diff_sim_std)[0]
        return diff_sim, diff_sim_std, not_significant
    else:
        return diff_sim
    
def KO_diff_sim_no_std(KO_avg, wt_avg):
    diff_sim = KO_avg - wt_avg
    return diff_sim
    
    
def KO_heat_comparison_T(diff, exp_data, title, KO_genes_order, fig, ax, Norm=True):
    # create a 23 x 2 matrix merging diff and exp_data
    if Norm == True:
        data = np.array([diff/np.max(np.abs(diff)), exp_data/np.max(np.abs(exp_data))])
    else:
        data = np.array([diff, exp_data])
    # print(data)
    # simulated data
    im0 = ax.imshow(data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            text = ax.text(i,j, data[j,i].round(1), fontsize=22, rotation=0,
                        ha="center", va="center", color="k")
    # add colorbar to ax[1] and remove the ticks 
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(['Generated', 'Experimental'], fontsize=30, rotation='horizontal')
    ax.set_xticks(np.arange(exp_data.shape[0]))
    ax.set_xticklabels(KO_genes_order, fontsize=30, rotation='vertical')
    ax.set_ylabel(title, fontsize=35)
    ax.tick_params(axis='y', pad=2)

    # # colorbar
    # cbar = fig.colorbar(im0, cax=ax, orientation='vertical',  pad=0.1, shrink=0.5)
    # cbar.ax.set_aspect(20)
    # cbar.set_label("Normalized \n KO and WT \ndifference", fontsize=35)
    # cbar.ax.tick_params(axis='y', labelsize=25)



### Fraction of differences in agremeent with experimental data
def fraction_agreement(diff_sim_norm, log2FC_exp_norm, genes_KOs, threshold):
    common_low = np.intersect1d(np.where(diff_sim_norm<=-threshold)[0], np.where(log2FC_exp_norm<=-threshold)[0])
    frac_low = len(common_low)/len(genes_KOs)
    common_high = np.intersect1d(np.where(diff_sim_norm>=threshold)[0], np.where(log2FC_exp_norm>=threshold)[0])
    frac_high = len(common_high)/len(genes_KOs)
    common_zero = np.intersect1d(np.where(np.abs(diff_sim_norm)<threshold)[0],  np.where(np.abs(log2FC_exp_norm)<threshold)[0])
    frac_zero = len(common_zero)/len(genes_KOs)
    return frac_low, frac_high, frac_zero, sum([frac_low, frac_high, frac_zero])


# # def KO_plots_oneSim(ko_spins, ko_avg, ko_std, wt_avg, wt_std, ko_genes_order, exp_data, raster=True, avg=True):
# #     """function to simulate the KO data (active/inactive genes in time) and to plot, depending on the decision of the user, 
# #     the raster plot and the average active time for each gene in wild type and in KO.
# #     Args:
# #         ko_spins (array): array of the KO data (active/inactive genes in time)
# #         ko_avg (array): array of the average active time for each gene in KO
# #         ko_std (array): array of the standard deviation of the active time for each gene in KO
# #         wt_avg (array): array of the average active time for each gene in wild type
# #         wt_std (array): array of the standard deviation of the active time for each gene in wild type
# #         ko_genes_order (array): array of the genes in the same order of the KO data
        
# #         raster (bool): True if the user wants to plot the raster plot
# #         avg (bool): True if the user wants to plot the average active time for each gene in wild type and in KO
# #     """
# #     # raster plot
# #     if raster:
# #         fun_plotting.raster_plot(ko_spins, 'Reconstruction', 1, ko_genes_order)
# #         plt.show()
        
# #     # average activity time per gene in wild type and in KO
# #     if avg:
# #         fig, ax = plt.subplots(3, 1, figsize=(20,10), gridspec_kw={'height_ratios': [10,0.5,0.2]})
# #         ax[0].errorbar(ko_genes_order, ko_avg, yerr=ko_std,
# #                      alpha=1, 
# #                      fmt="o", ms = 10,
# #                      elinewidth=3,
# #                      color="steelblue",
# #                      capsize=10,
# #                      label= "Simulated KO Data")

# #         ax[0].errorbar(ko_genes_order, wt_avg, yerr=wt_std,
# #                      alpha=1, 
# #                      fmt="o", ms = 10,
# #                      elinewidth=1,
# #                      color="indianred",
# #                      capsize=10,
# #                      label = "Simulated WT Data")
# #         ax[0].legend(loc="upper left", fontsize=16)
# #         ax[0].set_ylabel("Average spin", fontsize=16)
# #         # ax[0].set_xlabel("Genes", fontsize=16)
# #         ax[0].set_title("Average spin values for each genes", fontsize=20)
# #         ax[0].set_xticklabels(ko_genes_order, fontsize=12)
# #         ax[0].grid(True)

# #         # add colormap below the plot showing leeb data, exp_data
# #         im = ax[1].imshow(exp_data.reshape(1, exp_data.shape[0]), cmap='coolwarm', aspect='auto',
# #                           norm = MidpointNormalize(midpoint=0,
# #                                              vmin=-max(np.max(exp_data), np.abs(np.min(exp_data))),
# #                                              vmax=max(np.max(exp_data), np.abs(np.min(exp_data)))))
        
# #         # add colorbar to ax[1] and remove the ticks 
# #         ax[1].set_yticks([])
# #         ax[1].set_xticks([])
# #         # ax[1].set_xticks(np.arange(exp_data.shape[0]))
# #         ax[1].set_xlabel("Genes", fontsize=16)
        
# #         # change the horizontal size of the image
# #         ax[1].set_xlim(-1,exp_data.shape[0])
        
# #         # colorbar
# #         fig.colorbar(im, cax=ax[2], orientation='horizontal', fraction=0.1, pad=0.1)
# #         plt.show()


# # def KO_plots_oneSim_T(diff, ko_avg, ko_std, wt_avg, wt_std, ko_genes_order, exp_data):
# #     fig, ax = plt.subplots(5, 1, figsize=(20,10), gridspec_kw={'height_ratios': [10,0.5,0.2, 0.5,0.2]})
# #     ax[0].errorbar(ko_genes_order, ko_avg, yerr=ko_std,
# #                     alpha=1, 
# #                     fmt="o", ms = 10,
# #                     elinewidth=3,
# #                     color="steelblue",
# #                     capsize=10,
# #                     label= "Simulated KO Data")

# #     ax[0].errorbar(ko_genes_order, wt_avg, yerr=wt_std,
# #                     alpha=1, 
# #                     fmt="o", ms = 10,
# #                     elinewidth=1,
# #                     color="indianred",
# #                     capsize=10,
# #                     label = "Simulated WT Data")
# #     ax[0].legend(loc="upper left", fontsize=16)
# #     ax[0].set_ylabel("Average spin", fontsize=16)
# #     # ax[0].set_xlabel("Genes", fontsize=16)
# #     ax[0].set_title("Average spin values for each genes", fontsize=20)
# #     ax[0].set_xticklabels(ko_genes_order, fontsize=12)
# #     ax[0].grid(True)
    
# #     # plot showing simulated data
# #     im1 = ax[1].imshow(diff.reshape(1, diff.shape[0]), cmap='coolwarm', aspect='auto',
# #                         norm = MidpointNormalize(midpoint=0,
# #                                             vmin=-max(np.max(diff), np.abs(np.min(diff))),
# #                                             vmax=max(np.max(diff), np.abs(np.min(diff)))))
    
# #     # add colorbar to ax[1] and remove the ticks 
# #     ax[1].set_yticks([])
# #     ax[1].set_xticks([])
    
# #     # change the horizontal size of the image
# #     ax[1].set_xlim(-1,exp_data.shape[0])
# #     # colorbar
# #     fig.colorbar(im1, cax=ax[2], orientation='horizontal', fraction=0.1, pad=0.1)

# #     # add colormap below the plot showing leeb data, exp_data
# #     im3 = ax[3].imshow(exp_data.reshape(1, exp_data.shape[0]), cmap='coolwarm', aspect='auto',
# #                         norm = MidpointNormalize(midpoint=0,
# #                                             vmin=-max(np.max(exp_data), np.abs(np.min(exp_data))),
# #                                             vmax=max(np.max(exp_data), np.abs(np.min(exp_data)))))
    
# #     # add colorbar to ax[1] and remove the ticks 
# #     ax[3].set_yticks([])
# #     ax[3].set_xticks([])
# #     # ax[1].set_xticks(np.arange(exp_data.shape[0]))
# #     ax[3].set_xlabel("Genes", fontsize=16)
    
# #     # change the horizontal size of the image
# #     ax[3].set_xlim(-1,exp_data.shape[0])
# #     # colorbar
# #     fig.colorbar(im3, cax=ax[4], orientation='horizontal', fraction=0.1, pad=0.1)
# #     plt.show()


# def KO_plots_SimMultiple(ko_avg, ko_std, wt_avg, wt_std, exp_data, ko_genes_order):
#     """(For 3 KO genes) Plot the average activity of the simulated data and the original data"""
#     plt.figure(figsize=(18,5))
#     plt.errorbar(KO_genes_order, ko_avg, yerr=ko_std,  
#                     alpha=1, 
#                     fmt="o", ms = 10,
#                     elinewidth=3,
#                     color="steelblue",
#                     capsize=10,
#                     label= "KO Data")

#     plt.errorbar(KO_genes_order, wt_avg, yerr=wt_std,
#                     alpha=1, 
#                     fmt="o", ms = 10,
#                     elinewidth=1,
#                     color="indianred",
#                     capsize=10,
#                     label = "WT data")

#     fig, ax = plt.subplots(3, 1, figsize=(20,10), gridspec_kw={'height_ratios': [10,0.5,0.2]})
#     ax[0].errorbar(ko_genes_order, ko_avg, yerr=ko_std,
#                     alpha=1, 
#                     fmt="o", ms = 10,
#                     elinewidth=3,
#                     color="steelblue",
#                     capsize=10,
#                     label= "Simulated KO Data")

#     ax[0].errorbar(ko_genes_order, wt_avg, yerr=wt_std,
#                     alpha=1, 
#                     fmt="o", ms = 10,
#                     elinewidth=1,
#                     color="indianred",
#                     capsize=10,
#                     label = "Simulated WT Data")
#     ax[0].legend(loc="upper left", fontsize=16)
#     ax[0].set_ylabel("Average spin", fontsize=16)
#     # ax[0].set_xlabel("Genes", fontsize=16)
#     ax[0].set_title("Average spin values for each genes", fontsize=20)
#     ax[0].set_xticklabels(ko_genes_order, fontsize=12)
#     ax[0].grid(True)

#     # add colormap below the plot showing leeb data, exp_data
#     im = ax[1].imshow(exp_data.reshape(1, exp_data.shape[0]), cmap='coolwarm', aspect='auto')
#     # add colorbar to ax[1] and remove the ticks 
#     ax[1].set_yticks([])
#     ax[1].set_xticks([])
#     ax[1].set_xlabel("Genes", fontsize=16)
#     # change the horizontal size of the image
#     ax[1].set_xlim(-1,exp_data.shape[0])
    
#     # colorbar
#     fig.colorbar(im, cax=ax[2], orientation='horizontal', fraction=0.1, pad=0.1)
#     plt.show()
    
# # --------------------------------------------------------------------------------------------
# # ------------------------------- KO plots - SCODE -------------------------------------------
# # --------------------------------------------------------------------------------------------
    
# # def KO_plots_SimMultiple_SCODE(KO_spins, KO_genes_order, wt_avg, wt_std):
# #     """(For 3 KO genes)
# #     compute the average and std of the activity of the simulated data"""
# #     # mean active time
# #     std_temp = KO_spins.reshape((KO_spins.shape[0],KO_spins.shape[1]*KO_spins.shape[2]))
# #     # KO_std_spin = np.array(KO_spins_std.mean(axis=1))
# #     KO_std_spin = std_temp.std(axis=1)
# #     KO_avg_spin = np.array(KO_spins.mean(axis=1))
# #     KO_avg_spin = np.array(KO_avg_spin.mean(axis=1))

# #     plt.figure(figsize=(18,5))
# #     plt.errorbar(KO_genes_order, KO_avg_spin, yerr=KO_std_spin/np.sqrt(len(wt_std)),  
# #                     alpha=1, 
# #                     fmt="o", ms = 10,
# #                     elinewidth=3,
# #                     color="steelblue",
# #                     capsize=10,
# #                     label= "simulated Data")

# #     plt.errorbar(KO_genes_order, wt_avg, yerr=wt_std/np.sqrt(len(wt_std)), 
# #                     alpha=1, 
# #                     fmt="o", ms = 10,
# #                     elinewidth=1,
# #                     color="indianred",
# #                     capsize=10,
# #                     label = "original data")
# #     plt.legend(loc="upper left", fontsize=16)
# #     plt.xticks(fontsize=12)
# #     plt.ylabel("Average spin", fontsize=16)
# #     plt.xlabel("Genes", fontsize=16)
# #     plt.title("Average spin values for each genes", fontsize=20)
# #     plt.grid(True)
# #     plt.show()
    
# def WT_avg_w( wt_spins):
#     """ Compute the weighted average of the activity of the WT genes
#     Args:
#         - wt_spins: array of shape (N_genes, N_time, N_test)
#     """
#     # activity for each gene in WT
#     wt_avg_spin = np.array(wt_spins.mean(axis=1))
#     wt_std_spin = np.array(wt_spins.std(axis=1, ddof=1))/np.sqrt(wt_spins.shape[1])
#     wt_weighted_avg = np.zeros(wt_avg_spin.shape[0])
#     wt_weighted_std = np.zeros(wt_avg_spin.shape[0])    
#     for j in range(wt_avg_spin.shape[0]):
#         wt_weighted_avg[j] = DescrStatsW(wt_avg_spin[j,:], weights=1/(wt_std_spin[j,:])**2, ddof=1).mean
#         wt_weighted_std[j] = DescrStatsW(wt_avg_spin[j,:], weights=1/(wt_std_spin[j,:])**2, ddof=1).std
        
#     return(wt_weighted_avg, wt_weighted_std)

# def KO_plots_avgAct_SCODE(KO_avg, KO_std, wt_avg, wt_std, KO_genes_order, N_sigma=1):
#     """ For SCODE
#     Args: 
#         - KO_avg: array of shape (N_genes)
#         - KO_std: array of shape (N_genes)
#         - wt_avg: array of shape (N_genes)
#         - wt_std: array of shape (N_genes)
#         - KO_genes_order: array of shape (N_genes) containing the genes names
#     """
#     plt.figure(figsize=(18,5))
#     plt.errorbar(KO_genes_order, KO_avg, yerr=N_sigma*KO_std,  
#                     alpha=1, 
#                     fmt="o", ms = 10,
#                     elinewidth=3,
#                     color="steelblue",
#                     capsize=10,
#                     label= "Simulated KO")

#     plt.errorbar(KO_genes_order, wt_avg, yerr=N_sigma*wt_std, 
#                     alpha=1, 
#                     fmt="o", ms = 10,
#                     elinewidth=1,
#                     color="indianred",
#                     capsize=10,
#                     label = "Simulated WT")
#     plt.legend(loc="upper left", fontsize=16)
#     plt.xticks(fontsize=12)
#     plt.ylabel("Average GE", fontsize=16)
#     plt.xlabel("Genes", fontsize=16)
#     plt.title("Average GE values for each genes", fontsize=20)
#     plt.grid(True)
#     plt.show()


# def KO_heat_comparison_vert(diff, exp_data, title, KO_genes_order, Norm=True):
#     # create a 23 x 2 matrix merging diff and exp_data
#     if Norm:
#         data = np.array([diff / np.max(np.abs(diff)), exp_data / np.max(np.abs(exp_data))])
#     else:
#         data = np.array([diff, exp_data / np.max(np.abs(exp_data))])
    
#     # Transpose the data to swap x and y axes
#     data = data.T  # Transpose the matrix
    
#     fig, ax = plt.subplots(1, 2, figsize=(3,24), gridspec_kw={'height_ratios': [1], 'width_ratios': [1.7, 0.1]})
    
#     # simulated data
#     im0 = ax[0].imshow(data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
#     for i in range(data.shape[1]):
#         for j in range(data.shape[0]):
#             text = ax[0].text(i, j, data[j, i].round(1), fontsize=25, rotation=0,
#                         ha="center", va="center", color="k")
    
#     # Swap x and y labels
#     ax[0].set_xticks(np.arange(2))
#     ax[0].set_xticklabels(['Generated', 'Experimental'], fontsize=35, rotation='vertical')
#     ax[0].set_yticks(np.arange(exp_data.shape[0]))
#     ax[0].set_yticklabels(KO_genes_order, fontsize=35, rotation='horizontal')
    
#     # colorbar
#     cbar = fig.colorbar(im0, cax=ax[1], orientation='vertical', pad=0.1, shrink=0.5)
#     cbar.ax.set_aspect(20)
#     cbar.set_label("Normalized \n KO and WT \ndifference", fontsize=35)
#     cbar.ax.tick_params(axis='y', labelsize=25)

#     # save image as .tiff
#     plt.savefig(title + '.svg', bbox_inches='tight', dpi=300)
#     # plt.title(title)
#     plt.show()

# def KO_diff_ExpVsSim(logFC_Exp, diff_Sim, diff_Sim_std, genes_order = genes_order, thr_significance=3):
#     """ Compute the agreement between the logFC of the experiment and the KO-WT difference for simulated data.
#     Args:
#         - logFC_Exp (array): logFC of the experiment
#         - diff_Sim (array): difference between the average activity of the KO and the WT
#         - diff_Sim_std (array): standard deviation of the difference
#         - genes_order (list of strings): list of genes in the order of the interaction matrix (remember to remove the KO genes)
#         - thr_significance (float): number of standard deviation to consider the difference significant
        
#     Output:
#         - in_agreement: fractions of considered genes (only significant data) in agreement between the experiment and the simulation
#         - data_considered: number of considered genes (only significant LogFC and KO-WT difference)
#         - idx_Acc: indexes of the genes that are significant for the LofFC of the experimental data and for the KO-WT difference    
#     """
#     comparison= np.array(np.sign(logFC_Exp)*np.sign(diff_Sim))
#     index_logFC_Exp = np.where(logFC_Exp==0)[0]
#     index_diffSim = np.where((np.abs(diff_Sim))<thr_significance*diff_Sim_std)[0]
#     # print(index_diffSim)
#     print("KO_std-wt_std not significant for gene ", genes_order[index_diffSim]) #, index_diffSim)

#     # union of the two indexes
#     idx_notAcc = np.union1d(index_logFC_Exp, index_diffSim)
#     # find the indexes of the genes that are not in idx_notAcc
#     idx_Acc = np.setdiff1d(np.arange(len(logFC_Exp)), idx_notAcc)
#     data_considered = len(idx_Acc)
    
#     # consider the comparison elements that are not in idx_notAcc
#     comparison_sel = comparison[idx_Acc]
    
#     if data_considered == 0:
#         in_agreement = 0
#         no_agreement = 1
#     else:
#         in_agreement = len(np.where(comparison_sel==1)[0])/data_considered
#         no_agreement = len(np.where(comparison_sel==-1)[0])/data_considered
    
#     # Check
#     check_sum = in_agreement+no_agreement-1
#     check = np.where(check_sum>0.001)[0]
#     if check.size > 0:
#         print("Error in comparison Exp and Sim")
        
#     return(in_agreement, data_considered, genes_order[idx_Acc])

# def KO_plof_Diff_LogFC(logFC, diff, diff_std, KO_genes_order, idx_notS, title, n_sigma=1):
#     """ Plot the difference between the average activity of the KO and the WT and the logFC of the experiment
#     Args:
#         logFC (array): logFC for experimental data
#         diff (array): difference between WT and KO for simulated data
#         diff_std (array): standard deviation of the difference between WT and KO for simulated data
#         KO_genes_order (array): genes order with the KO gene removed
#         idx_notS (array): array of indexes of genes that are not significant
#         title (str): title of the plot
#     """
#     x_range = np.arange(0, len(diff))
#     plt.figure(figsize=(10,5))
    
#     # simulated data
#     plt.errorbar(x_range, diff, yerr=n_sigma*diff_std, fmt='o', color='slateblue', ecolor='slateblue', elinewidth=2, capsize=0, label='Simulated')
#     # plot with a different color the genes that are not significant (idx_notS)
#     if len(idx_notS)>0:
#         plt.errorbar(x_range[idx_notS], diff[idx_notS], yerr=n_sigma*diff_std[idx_notS], fmt='o', color='red', ecolor='red', elinewidth=3, capsize=0, label='Not significant difference')

#     # experimental data
#     plt.plot(x_range, logFC, 'o', color='Lightseagreen', label='Experimental')
#     # line at 0
#     plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
#     # plot settings
#     plt.xticks(x_range, KO_genes_order, rotation='vertical')
#     plt.grid(which='both', axis='both')
#     plt.title(title)
#     plt.legend(fontsize=18)
#     plt.show()
    
# def KO_plof_Diff_LogFC_heat(logFC, diff, diff_std, KO_genes_order, title, n_sigma=1):
#     x_range = np.arange(0, len(diff))
#     fig, ax = plt.subplots(3, 1, figsize=(20,10), gridspec_kw={'height_ratios': [10,0.5,0.2]})

#     # simulated data
#     ax[0].errorbar(x_range, diff, yerr=n_sigma*diff_std, fmt='o', 
#                  color='slateblue', ecolor='slateblue', elinewidth=2, 
#                  capsize=0, label='Simulated')
#     # experimental data
#     ax[0].plot(x_range, logFC, 'o', color='Lightseagreen', label='Experimental')
#     # line at 0
#     ax[0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
#     # plot settings
#     ax[0].set_xticks(x_range)
#     ax[0].set_xticklabels(KO_genes_order, fontsize=12, rotation='vertical')
#     ax[0].grid(which='both', axis='both')
#     ax[0].set_title(title)
#     ax[0].legend(fontsize=18)

#     # add colormap below the plot showing exp data
#     im = ax[1].imshow(np.array(logFC).T, cmap='coolwarm', aspect='auto', 
#                       norm = MidpointNormalize(midpoint=0,
#                                              vmin=-max(np.max(logFC), np.abs(np.min(logFC))),
#                                              vmax=max(np.max(logFC), np.abs(np.min(logFC)))))
#     # add colorbar to ax[1] and remove the ticks 
#     ax[1].set_yticks([])
#     ax[1].set_xticks([])
#     ax[1].set_xlabel("Genes", fontsize=16)
#     ax[1].set_xlim(-1,logFC.shape[0])

#     # colorbar
#     fig.colorbar(im, cax=ax[2], orientation='horizontal', fraction=0.1, pad=0.1)
#     plt.show()
    
# class MidpointNormalize(pltcolors.Normalize):
#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#         self.midpoint = midpoint
#         pltcolors.Normalize.__init__(self, vmin, vmax, clip)

#     def __call__(self, value, clip=None):
#         x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#         return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# def KO_heat_comparison(diff, exp_data, title, KO_genes_order):
#     fig, ax = plt.subplots(2, 1, figsize=(15,3), gridspec_kw={'height_ratios': [1,1]})
#     # simulated data
#     im0 = ax[0].imshow(diff.reshape(1, exp_data.shape[0]), cmap='coolwarm', aspect='auto')
#     # add colorbar to ax[1] and remove the ticks 
#     ax[0].set_yticks([])
#     ax[0].set_xticks([])
#     # ax[0].set_xticklabels(KO_genes_order, fontsize=12, rotation='horizontal')
#     # ax[0].set_xlabel("Genes", fontsize=16)
#     # colorbar
#     # fig.colorbar(im0, cax=ax[1], orientation='horizontal', fraction=0.1, pad=0.1)

#     # add colormap below the plot showing leeb data, exp_data
#     im1 = ax[1].imshow(exp_data.reshape(1, exp_data.shape[0]), cmap='coolwarm', aspect='auto')
#     # add colorbar to ax[1] and remove the ticks 
#     ax[1].set_yticks([])
#     # ax[2].set_xticks([])
#     ax[1].set_xticks(np.arange(exp_data.shape[0]))
#     ax[1].set_xticklabels(KO_genes_order, fontsize=12, rotation='horizontal')
#     ax[1].set_xlabel("Genes", fontsize=16)
#     # colorbar
#     # fig.colorbar(im1, cax=ax[3], orientation='horizontal', fraction=0.1, pad=0.1)
#     plt.title(title)
#     plt.show()


# # -------------------------------------------------------------------------------------------------
# # ------------------------------ Plot average activity time of genes ------------------------------
# # -------------------------------------------------------------------------------------------------

# def plot_activity_simulated(spins_df_sim, genes_order, title, color, ax, reshape=True):
#     """ Compute the average activity of the genes in the simulated dataset and their standard deviation.
#     Plot the average activity of the genes in the simulated dataset.
#     Input:
#         - spins_df_sim (array): simulated dataset
#         - genes_order (list): list of genes names
#         - title (str): title of the plot
#         - color (str): color of the data
#         - ax (axis): axis of the plot
#     Output:
#         - avg_activity (array): average activity of the genes in the simulated dataset
#         - avg_activity_std (array): standard deviation of the average activity of the genes in the simulated dataset
        
#     NOTES: The standard deviation is computed condidering all the data (i.e. all the time steps, all the tests)
#     and dividing by the square root of the number of tests.
#         """
#     if reshape:
#         df_reshaped = spins_df_sim.reshape((spins_df_sim.shape[0], spins_df_sim.shape[1]*spins_df_sim.shape[2]))
#     else:
#         df_reshaped = spins_df_sim

#     # spins_df_sim is (n_genes, n_time, n_test)
#     # avg_activity_each = spins_df_sim.mean(axis=1)
#     # # print(avg_activity_each.shape, avg_activity_std_each.shape)
#     # avg_activity = np.zeros(spins_df_sim.shape[0])
#     # avg_activity_std = np.zeros(spins_df_sim.shape[0])
#     # for j in range(spins_df_sim.shape[0]):
#     #     # avg_activity_std[j] = DescrStatsW(avg_activity_each[j,:], weights=1/(avg_activity_std_each[j,:])**2, ddof=1).std
#     #     avg_activity[j] = np.mean(avg_activity_each[j,:])
#     #     avg_activity_std[j] = np.std(spins_df_sim[j,:,:], ddof=1)/np.sqrt(spins_df_sim.shape[2])
#     # comupute first and third quartile
#     quartiles = np.quantile(df_reshaped, [0.25, 0.75], axis=1)
#     yerr = []
#     for ii in range(len(genes_order)):
#         yerr.append((quartiles[1,ii],quartiles[0,ii]))
#     yerr = np.array(yerr).T
#     ax.errorbar(genes_order, df_reshaped.mean(axis=1), yerr=[df_reshaped.mean(axis=1)-quartiles[0,:], quartiles[1,:]-df_reshaped.mean(axis=1)], #avg_activity_std, 
#                  alpha=1, 
#                  fmt="o", ms = 10,
#                  elinewidth=1,
#                  color=color,
#                  capsize=10,
#                  label = title)
#     # ax.plot(genes_order, quartiles[0,:], 'o', color="red")
#     # ax.plot(genes_order, quartiles[1,:], 'o', color="red")
#     ax.legend(loc="upper left", fontsize=20)
#     ax.set_ylabel("Average spin", fontsize=20)
#     ax.set_xlabel("Genes", fontsize=20)
#     # ax.set_title(title, fontsize=20)
#     ax.grid(True)
#     return(df_reshaped.mean(axis=1), df_reshaped.std(axis=1)) #(avg_activity, avg_activity_std)

# def plot_activity(spins_df, genes_order, title, color, ax):
#     """ Compute the average activity of the genes in the EXPERIMENTAL dataset and their standard deviation.
#     Input:
#         - spins_df (array): experimental dataset
#         - genes_order (list): list of genes names
#         - title (str): title of the plot
#         - color (str): color of the data
#         -  ax (axis): axis of the plot
#     Output:
#         - avg_activity (array): average activity of the genes in the experimental dataset
#         - avg_activity_std (array): standard deviation of the average activity of the genes in the experimental dataset.
#     """
#     # spins_df is (n_genes, n_time)
#     avg_activity     = spins_df.mean(axis=1)
#     avg_activity_std = spins_df.std(axis=1)/np.sqrt(spins_df.shape[1])

#     quartiles = np.quantile(spins_df, [0.25, 0.75], axis=1)

#     ax.errorbar(genes_order, avg_activity, yerr=[avg_activity-quartiles[0,:], quartiles[1,:]-avg_activity],  # avg_activity_std, 
#                  alpha=1, 
#                  fmt="o", ms = 10,
#                  elinewidth=1,
#                  color=color,
#                  capsize=10,
#                  label = title)

#     ax.legend(loc="upper left", fontsize=20)
#     # ax.set_xticks(fontsize=12)
#     ax.set_ylabel("Average spin", fontsize=20)
#     ax.set_xlabel("Genes", fontsize=20)
#     # ax.set_title(title, fontsize=20)
#     ax.grid(True)
#     return(avg_activity, avg_activity_std)