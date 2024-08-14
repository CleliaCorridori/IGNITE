import pandas as pd
import numpy as np
import lib.funcs_general as funcs_general
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import r2_score 
from scipy.stats import norm


def shuffle_dataframe(df, genes_names, interactions, N_test=500, Nbin=30, inferred_int_thr=0.03, method="Pearson"):
    """ Shuffle the dataframe and compute the Pearson correlation matrix.

    Args:
        - df (dataframe): dataframe with the gene expression values
        - genes_names (array): list of genes names
        - interactions (array): list of interactions
        - N_test (int): number of random shuffles
        - Nbin (int): number of bins for the histogram
        - MaxThr (float): threshold for the Pearson correlation values
       
    """
    N_rows = df.shape[0]
    N_cols = df.shape[1]

    # Melt the dataframe
    trial = df.melt(ignore_index=False)
    val_rnd = np.array(trial.iloc[:,1]) #gene expression values
    # Create an array of zeros to save the correlation matrices
    corr_matrices  = np.zeros((N_test,len(genes_names), len(genes_names)))
    corr_matrices_no_diag  = np.zeros((N_test,len(genes_names), len(genes_names)))
    TP_frac_rnd = np.zeros(N_test)
    info_int_rnd= np.zeros((N_test, 4, len(interactions)))

    # Loop over the number of random shuffles
    for ii in range(N_test):
        np.random.seed(1234+ii)
        # Random reshuffle of the GE data
        np.random.shuffle(val_rnd)
        # Reshape as the original dataframe
        val_rnd = val_rnd.reshape(N_rows,N_cols)
        trial_long = pd.DataFrame(val_rnd, index= genes_names, columns= df.columns).set_index(genes_names) # reshaped dataframe setting the genes as index
        # Pearson matrix
        if method == "Pearson":
            corr_matr = np.corrcoef(trial_long)
            corr_matr_no_diag = corr_matr.copy()
            np.fill_diagonal(corr_matr_no_diag, float("Nan")) # fill the diagonal with NaN
        elif method == "MaxEnt":
            # corr_matr = -np.linalg.pinv(np.corrcoef(trial_long))
            corr_matr = - compute_interaction_matrix(trial_long)
            corr_matr_no_diag = corr_matr.copy()
            # np.fill_diagonal(corr_matr_no_diag, float("Nan")) # fill the diagonal with NaN
            # print("Post-max:", np.nanmax(np.abs(corr_matr)))
            
            
        # save all the correlation values
        corr_matrices[ii] = corr_matr
        corr_matrices_no_diag[ii] = corr_matr_no_diag

        TP_frac_rnd[ii], info_int_rnd[ii,:,:], _ = funcs_general.TP_plot(interactions, corr_matr, genes_names, 
                                                       inferred_int_thr=inferred_int_thr, Norm_Matx = False,
                                                       data_type=" Best model for lN PST MB data",
                                                       figplot=False, verbose=False, nbin=Nbin, Norm = False)
    return(TP_frac_rnd, info_int_rnd, corr_matrices_no_diag)





def to_thr_matrix(matrix, thr=0.02):
    """Function to set the interactions values below thr to 0
    """
    thr_matrix = np.copy(matrix)
    #Set to zero too small absolute values
    thr_matrix[np.abs(thr_matrix) <  (thr*np.nanmax(np.abs(matrix)))] = 0
    return thr_matrix

def fit_normal_distribution(data, noise_thr=3, text="", Nbins=19):
    """ Fit a normal distribution to the data and plot the histogram and the pdf.
    Args:
        - data (array): array containing the data
        - noise_thr (float): threshold for the noise
        - text (str): text to add to the title of the plot
    
    Output:
        - mu (float): mean of the fitted distribution
        - std (float): standard deviation of the fitted distribution
        - Nnoise (int): number of values above the noise threshold
        - Ntot (int): total number of values
        """
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)
    data = np.array([x for x in data.flatten() if ~np.isnan(x)])
    mu, std = norm.fit(data)

    # Plot the histogram
    plt.figure(figsize=(7,5))
    n, _, _ = plt.hist(data, bins=Nbins, density=True, alpha=0.6, color='darkblue')

    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, Nbins)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=1.5, label = 'fit')
    plt.xlabel('Correlation', fontsize=16)
    plt.ylabel('Pdf', fontsize=16 )
    # title = ", mu = %.2f,  2std = %.2f" % (np.abs(mu), noise_thr*std)
    title = ", mu = %.2f,  %d$\sigma$ = %.2f" % (np.abs(mu), noise_thr, noise_thr*std)
    plt.title(text+title, fontsize=20)

    #evaluation of the fit
    centroids = (n[1:]+n[:-1])/2
    R_square = r2_score(n, p) 
    print('regression score', R_square) 
    print('The noise-threshold is ', np.round(std*noise_thr,3)) 

    plt.axvline(x = -noise_thr*std, color = 'r')
    plt.axvline(x = noise_thr*std, color = 'r', label = str(noise_thr)+' $\sigma$')
    plt.legend()
    plt.text(-std*noise_thr, np.max(n)-0.1*np.max(n), '$R^2$ = '+str(np.round(R_square,2)),
            bbox={'facecolor': 'b', 'alpha': 0.5, 'pad': 5})
    plt.show()
    return(std*noise_thr)

# # ----------------------------------------------------------------------------------------------
# # ------------------------------  single known interaction Check  ------------------------------
# # ----------------------------------------------------------------------------------------------
def TP_distribution(interaction_list, interaction_matrices, genes_list):
    """ Compute the values of the interactions in the interaction list for all the matrices in the interaction_matrices array."""
    int_val = np.zeros((len(interaction_list), interaction_matrices.shape[0]))
    for ii in range(len(interaction_list)):
        couple = interaction_list[ii].split(" ")
        gene1_idx = np.where(genes_list == couple[1])[0] #idx of gene 1 (target)
        gene0_idx = np.where(genes_list == couple[0])[0] #idx of gene 0 (source)
            
        # Get the interaction value and the sign of the interaction
        int_val[ii,:] = interaction_matrices[:,gene1_idx[0], gene0_idx[0]]
    return(int_val)

def single_int_check(TPtrial_list, corr_matrices_df_spec, info_int_spec, genes_order, text=""):
    # Compute the distribution of the shuffled interactions
    interactions_shuffled = TP_distribution(TPtrial_list, corr_matrices_df_spec, genes_order)
    
    # Set the figure size to be 1.5 cm narrower than A4 size
    fig_width_cm = 19.5  # A4 width - 1.5 cm
    fig_height_cm = 25.7  # A4 height / 2 roughly for a balanced aspect
    n_rows = 6
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width_cm / 2.54, fig_height_cm / 2.54))

    # Create a list to store the histograms
    histograms = []
    
    # Iterate over the rows and columns of the subplot to compute the histograms
    for ii in range(info_int_spec.shape[1]):
        quant = np.quantile(interactions_shuffled[ii,:],[0.01, 0.5, 0.99])
        # Compute the histogram
        bins_thr = max(np.abs(quant[0]), quant[2] )
        bins = np.linspace(-bins_thr-0.55*bins_thr, bins_thr+0.55*bins_thr, 15)
        nn, bins_J, _ = plt.hist(interactions_shuffled[ii,:], bins=bins, density=True, alpha=0.6, color='white')
        centroids_J = (bins_J[1:] + bins_J[:-1]) / 2
        
        # Store the histogram in the list
        histograms.append((centroids_J, nn))
        
    # Iterate over the histograms to plot them in the main loop
    check = len(TPtrial_list)
    for ii, (centroids_J, nn) in enumerate(histograms):
        perc_plot = np.quantile(interactions_shuffled[ii,:],[0.01, 0.5, 0.99])
        # Compute the histogram
        bins_thr = max(np.abs(perc_plot[0]), perc_plot[2] )
        # Get the axis object for the current subplot
        ax = axs[ii // n_cols, ii % n_cols]
        # comute the 5-th and 95-th percentile
        quant = np.quantile(interactions_shuffled[ii,:],[0.05, 0.5, 0.95])
        # Plot the histogram
        ax.axvline(x=info_int_spec[2,ii], color="red", lw=2, label="Inferred Interactions")
        ax.set_title(TPtrial_list[ii], fontsize=10)
        ax.axvline(x=quant[0], color="orange", lw=1, label="5-th and 95-th percentile")
        ax.axvline(x=quant[2], color="orange", lw=1)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.text(-0.98*bins_thr, np.max(nn)*0.9, np.round(info_int_spec[2,ii],2), fontsize=8, ha="center",weight='bold')
        ax.set_xlim([-bins_thr-0.55*bins_thr, bins_thr+0.55*bins_thr])
        # ax.set_ylabel('Density', fontsize=8)
        # ax.set_xlabel('Interaction value', fontsize=8)
        # Plot the histogram
        ax.plot(centroids_J, nn, color="navy", lw=2)
        
        # 5-th and 95-th percentile
        # print("5-th and 95-th percentile: ", quant[0], quant[2])
        if (info_int_spec[2,ii]>=quant[0]) and (info_int_spec[2,ii]<=quant[2]):
            print("IN random: ", TPtrial_list[ii], info_int_spec[2,ii])
            check -= 1
    plt.tight_layout()
    legend = plt.legend(fontsize=10, loc='upper right')

    # Rotate each text in the legend by 90 degrees
    for text in legend.get_texts():
        text.set_rotation(0)
    # general title using an input text "text"
    # fig.suptitle(text, fontsize=27)   


