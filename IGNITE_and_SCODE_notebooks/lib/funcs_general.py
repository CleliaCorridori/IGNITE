import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
# Custom modules
import lib.figs_funcs as figfunc

# Setting for the plots
matplotlib.rc('text', usetex=True)
sns.set(font='Avenir')
sns.set(style="white")

# ------------------------------ check index dataframes -------------------------------
def check_index(array1, array2):
    """function to check that the two numpy array of indices are the same with the same order"""
    if np.array_equal(np.array(array1), np.array(array2)):
        print("True")
    else:
        print("ERROR: The two arrays are different")
        
# ------------------------------ binnarization ----------------------------------------
def binnarization(df, thr=0.5):
    """ Binarize the values of a DataFrame based on a threshold
    input: 
        df: DataFrame to binarize
        thr: threshold to binarize the values
        
    output:
        spins_df: binarized DataFrame
    """
    # Calculate the maximum value for each row of the DataFrame
    df_max = df.max(axis=1)

    # Create a copy of the DataFrame to store the binarized values
    spins_df = df.copy()
    # Loop through each row of the DataFrame: genes_order contains the genes of the dataframe
    for ii in range(df.shape[0]):
        # Binarize the values in the row
        spins_df.iloc[ii,:] = (df.iloc[ii,:] > (df_max[ii] * thr)).astype(float)
    # Replace all 0 values with -1
    spins_df[spins_df==0] = -1

    return spins_df


# ------------------------------ randomization
def check_shuffle(dataset, Ntest):
    """Function to check if the randomization is working properly.
    input:
    dataset: dataset to check, dimension: (Ntest, Ngenes, Ncells)
    Ntest: number of tests performed
    output:
    percentage of equal elements
    """
    # Check: for each test the result should be different
    check_eq = 0
    for ii in range(Ntest):
        for jj in range(ii+1, Ntest):
            # number of equal elements
            temp = len(np.where(np.abs(dataset[ii,:,:]-dataset[jj,:,:])<0.001)[0]) # number of equal elements
            if temp > 0.9*dataset.shape[1]*dataset.shape[2]: # if more than 70% of the elements are equal
                check_eq += 1 # count the number of equal elements
    return(check_eq/(Ntest*(Ntest-1))*2) # percentage of equal elements


# ------------------------------ check correct interactions given the adjacency matrix
def TP_check(interaction_list, interaction_matrix, genes_list, inferred_int_thr = 0, Norm = True):
    """
    NOTE: for not symmetric interaction_matrix:
    - rows: who undergoes the action;
    -columns: who acts.
    NOTE: to read the input interaction_list:
    - couple[0] : who acts;
    - couple[1] : who undergoes the action.
    
    NOTE: inferred_int_thr is computed as fraction of max(interaction_matrix) 
    
    inputs: 
    interaction_list: list of strings, each string is a couple of genes and the interaction value (+1, -1 )
    interaction_matrix: matrix of interactions
    genes_list: list of genes names
    inferred_int_thr: threshold to consider an interaction as inferred
    Norm: if True, normalize the known interaction divding by the maximum value of the interaction_matrix
    
    output:
    out_matx: matrix of interactions, dimension: (4, len(interaction_list)). The rows are:
            - row 0: who acts;
            - row 1: who undergoes the action;
            - row 2: interaction value;
            - row 3: 1 if the interaction is correctly inferred, 0 otherwise.
    """
    if Norm:
        m_max = np.nanmax(np.abs(interaction_matrix))
    else:
        m_max = 1

    out_matx = np.zeros((4, len(interaction_list)))
    
    for ii in range(len(interaction_list)): # split the list of strings
        couple = interaction_list[ii].split(" ")
        gene1_idx = np.where(genes_list == couple[1])[0] #idx of gene 1
        gene0_idx = np.where(genes_list == couple[0])[0] #idx of gene 0  
        
        # check if the interaction's genes already exist:
        if (len(np.where(genes_list == couple[0])[0])==0):
            print("Gene "+ couple[0]+" not found")
            continue
        if (len(np.where(genes_list == couple[1])[0])==0):
            print("Gene "+ couple[1]+" not found")
            continue
            
        # the subjects of the interaction
        out_matx[0,ii] = gene0_idx # who acts
        out_matx[1,ii] = gene1_idx # who undergoes the action
  
        # the interaction value (and the sign of the interaction)
        out_matx[2,ii] = interaction_matrix[gene1_idx[0], gene0_idx[0]]
        interaction = np.sign(out_matx[2,ii])

        if (interaction==int(couple[2])) and (np.abs(out_matx[2,ii])/m_max >= inferred_int_thr):
            out_matx[3,ii] = 1
        elif  (interaction==int(couple[2])) and (int(couple[2])==0):
            out_matx[3,ii] = 1
        else:
            out_matx[3,ii] = 0

    return(np.sum(out_matx[3,:])/len(out_matx[3,:]), out_matx)
        
        
def TP_plot(interaction_list, interaction_matrix, genes_order, inferred_int_thr=0, Norm_Matx = False,
            data_type="scRNA-seq PST MB", 
            figplot=True, nbin=30, 
            verbose=False, Norm=True):
    """Wrap function to visualize all the results of the comparison with the known interactions (TP)
    input:
    interaction_list: list of known interactions,
    interaction_matrix: matrix of inferred interactions,
    genes_order: list of genes in the same order as the rows of the interaction_matrix,
    inferred_int_thr: threshold to consider an interaction as correctly inferred (otherwise it is 0),
    Norm_Matx: if True, normalize the interaction_matrix to the maximum value,
    data_type: string only to print the type of data,
    verbose: if True, print the fraction of true positives and the TP and all interaction values,
    Norm: if True, normalize the KNOWN interactions to the maximum value of the interaction_matrix
    
    output:
    TP_fraction: fraction of true positives,
    TP_info: matrix of interactions, dimension: (4, len(interaction_list)) -> see TP_check function "out_matx" output
    interaction_matrix: matrix of inferred interactions, useful if it is normalized.
    """
    
    if Norm_Matx:
        interaction_matrix = interaction_matrix/np.nanmax(np.abs(interaction_matrix))

    # Check the list of known interactions correctly inferred
    TP_fraction, TP_info = TP_check(interaction_list, interaction_matrix, genes_order, inferred_int_thr, Norm=Norm)
    
    # Print the fraction of true positives and the TP and all interaction values:
    if verbose==True:
        print("\nRESULTS for " + data_type)
        print("\nTP fraction:", np.round(TP_fraction, 2))
        print("\nInteraction values:\n", np.round(TP_info[2,:],3))
        print("\nTP ints values:\n", np.round(TP_info[2,:]*TP_info[3,:],3))
    
    # If the figplot flag is set to True, plot the matrix and the distribution of the INTERACTION MATRIX
    if figplot==True:
        bins = np.linspace(np.ndarray.flatten(interaction_matrix).min(), np.ndarray.flatten(interaction_matrix).max(), nbin)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        figfunc.plotmat(interaction_matrix, fig, ax[0], genes_order, data_type+"")
        sns.histplot(np.ndarray.flatten(interaction_matrix), ax=ax[1], stat="density", bins=bins)
        plt.show()
        
    return(TP_fraction, TP_info, interaction_matrix)

# ----------------------- # BEST MODEL SELECTION using known interactions
def select_best_inferred_matrix(precision_scores, distance_scores, matrices, stop=False, idx_sel=0, use_prior=False):
    """
    Selects the best inferred matrix based on precision or distance scores.

    Args:
        precision_scores (numpy.ndarray): Array of precision scores.
        distance_scores (numpy.ndarray): Array of distance scores.
        matrices (numpy.ndarray): Array of matrices to select from.
        stop (bool, optional): If True, return the best matrix and 0. Default is False.
        idx_sel (int, optional): Index of the selected matrix when stop is True. Default is 0.
        use_prior (bool, optional): If True, use precision scores for selection; else, use distance scores. Default is False.

    Returns:
        numpy.ndarray or tuple: If stop is True, returns the best matrix and 0. Otherwise, returns the mean matrix and indices of selected matrices.
    """
    best_indices = []
    if use_prior == True:
        # print("Max precision score: ", precision_scores.max())
        best_indices = np.where(precision_scores == precision_scores.max())[0]
    else:
        # print("Min distance score: ", np.nanmin(distance_scores))
        best_indices = np.where(distance_scores == np.nanmin(distance_scores))[0]

    if stop or len(best_indices) == 1:
        print("Computing the best matrix index")
        high_mean_matrix = matrices[best_indices[idx_sel], :, :]
        return high_mean_matrix, best_indices
    else:
        print("Computing the indices of alle the best matrices and the mean matrix")
        selected_matrices = matrices[best_indices, :, :]
        high_mean_matrix = np.mean(selected_matrices, axis=0)

        return high_mean_matrix, best_indices



# # # ----------------------- # LogFC INFO
# # def InteractionList(df, perc=0):
# #     """function to extract the list of interactions from a dataframe of logFC values in the format:
# #     list of strings
# #     each string: "gene1 gene2 sign"

# #     Args:
# #         df (dataFrame): dataframe of logFC values
# #         perc (float, optional): threshold to consider an interaction. Defaults to 0.
        
# #     output: list of interactions 
# #     """
# #     thr = np.abs(df.max().max()*perc)
# #     output = []
# #     for row in df.index:
# #         for col in df.columns:
# #             element = df.loc[row, col]
# #             if element > thr:
# #                 sign = "1"
# #             elif element < -thr:
# #                 sign = "-1"
# #             else:
# #                 df.loc[row, col] = 0
# #                 sign = "0"
# #             if (sign == "-1") or (sign == "1"):
# #                 output.append(f"{col} {row} {sign}")
# #     return(output)


# # -------------------------------- For KO implementation --------------------------------
# #  Average Gene Expression
# def avg_heat_comparison(diff, exp_data, title, KO_genes_order, Norm=True):
#     # create a 23 x 2 matrix merging diff and exp_data
#     if Norm == True:
#         data = np.array([diff/np.max(np.abs(diff)), exp_data/np.max(np.abs(exp_data))])
#     else: 
#         data = np.array([diff, exp_data])
#     # print(data)
#     fig, ax = plt.subplots(1, 2, figsize=(19,2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1,0.1]})
#     # simulated data
#     im0 = ax[0].imshow(data, cmap='coolwarm', aspect='auto', vmin=0, vmax=max(np.max(diff), np.max(exp_data)))
#     for i in range(data.shape[1]):
#         for j in range(data.shape[0]):
#             text = ax[0].text(i,j, data[j,i].round(2), fontsize=12,
#                         ha="center", va="center", color="k")
#     # add colorbar to ax[1] and remove the ticks 
#     ax[0].set_yticks(np.arange(2))
#     ax[0].set_yticklabels(['Simulated', 'Experimental'], fontsize=16, rotation='horizontal')
#     ax[0].set_xticks(np.arange(exp_data.shape[0]))
#     ax[0].set_xticklabels(KO_genes_order, fontsize=20, rotation='vertical')
#     # ax[0].set_xlabel("Genes", fontsize=20)
#     # colorbar
#     fig.colorbar(im0, cax=ax[1], orientation='horizontal', fraction=0.1, pad=0.1, label="Avg Gene Expression")
#     plt.title(title)
#     plt.show()
    
def avg_heat_comparison_IGNITE(diff, exp_data, title, KO_genes_order, Norm=True):
    # create a N_genes x 2 matrix merging diff and exp_data
    if Norm == True:
        data = np.array([diff/np.max(np.abs(diff)), exp_data/np.max(np.abs(exp_data))])
    else: 
        data = np.array([diff, exp_data])
    # print(data)
    fig, ax = plt.subplots(1, 2, figsize=(19,2), gridspec_kw={'height_ratios': [1], 'width_ratios': [1,0.1]})
    # simulated data
    im0 = ax[0].imshow(data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            text = ax[0].text(i,j, data[j,i].round(2), fontsize=12,
                        ha="center", va="center", color="k")
    # add colorbar to ax[1] and remove the ticks 
    ax[0].set_yticks(np.arange(2))
    ax[0].set_yticklabels(['Simulated', 'Experimental'], fontsize=16, rotation='horizontal')
    ax[0].set_xticks(np.arange(exp_data.shape[0]))
    ax[0].set_xticklabels(KO_genes_order, fontsize=20, rotation='vertical')
    # ax[0].set_xlabel("Genes", fontsize=20)
    # colorbar
    fig.colorbar(im0, cax=ax[1], orientation='horizontal', fraction=0.1, pad=0.1, label="Avg Gene Expression")
    plt.title(title)
    plt.show()
    
# def avg_fraction_agreement_SCODE(diff_sim_norm, log2FC_exp_norm, genes_KOs, threshold=0.05, norm=True):
#     threshold = threshold*(max(np.max(diff_sim_norm), np.max(log2FC_exp_norm)))
#     common_high = np.intersect1d(np.where(diff_sim_norm>=threshold)[0], np.where(log2FC_exp_norm>=threshold)[0])
#     frac_high = len(common_high)/len(genes_KOs)
#     common_zero = np.intersect1d(np.where(np.abs(diff_sim_norm)<threshold)[0],  np.where(np.abs(log2FC_exp_norm)<threshold)[0])
#     frac_zero = len(common_zero)/len(genes_KOs)
#     return frac_high, frac_zero, sum([frac_high, frac_zero])

def avg_fraction_agreement_IGNITE(diff_sim_norm, log2FC_exp_norm, genes_KOs, threshold=0.05, norm=True):
    threshold = threshold*(max(np.max(np.abs(diff_sim_norm)), np.max(np.abs(log2FC_exp_norm))))
    # print(threshold)
    common_low = np.intersect1d(np.where(diff_sim_norm<=-threshold)[0], np.where(log2FC_exp_norm<=-threshold)[0])
    frac_low = len(common_low)/len(genes_KOs)
    common_high = np.intersect1d(np.where(diff_sim_norm>=threshold)[0], np.where(log2FC_exp_norm>=threshold)[0])
    frac_high = len(common_high)/len(genes_KOs)
    common_zero = np.intersect1d(np.where(np.abs(diff_sim_norm)<threshold)[0],  np.where(np.abs(log2FC_exp_norm)<threshold)[0])
    frac_zero = len(common_zero)/len(genes_KOs)
    return frac_low, frac_high, frac_zero, sum([frac_low, frac_high, frac_zero])