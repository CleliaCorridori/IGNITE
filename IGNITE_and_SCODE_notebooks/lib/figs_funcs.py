import pandas as pd
import numpy as np
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rc('text', usetex=True)
sns.set(font='Avenir')
sns.set(style="white")

# RASTER PLOT _________________________________________________________________

def raster_plot(mat, ax, title = None, delta_t = 1, ax_names=[]):
    """ Plot the raster plot of the gene expression"""
    N, T = mat.shape
    
    
    ax.imshow(mat[:,:], aspect='auto', cmap = 'gray_r', interpolation = 'None')
             # extent = [0,T*delta_t, 1, N])
    
    if title != None:
        ax.set_title(title, fontsize = 25, y = 1.02)
        
    ax.set_ylabel('Gene label', fontsize = 25, labelpad = 10)
    ax.set_xlabel('Cells', fontsize = 25, labelpad = 10)
    ax.set_xticks([])
    #ax.tick_params(labelsize=17)
    ax.set_yticks(np.linspace(0,N-1,N))
    ax.set_yticklabels(ax_names, fontsize=10)
    
    return ax

# ______________________________________________________________________________
# FOR DATA PROCESSING _________________________________________________________
def plot_ge_in_time(df_bulk, df_std, genes_type, title):
    """Plot the gene expression for each gene in each group in time with theis std

    Args:
        df_bulk (dataframe): average gene expression for each gene in each group in time
        df_std (dataframe): std of the gene expression for each gene in each group in time
        genes_type (list of lists): list of lists of genes for each group
        title (list of strings): list of strings with the title for each group
    
    """
    for ii in range(len(genes_type)):
        plt.figure(figsize=(12,8))
        for jj in range(len(genes_type[ii])):
            plt.errorbar(df_bulk.columns,df_bulk.loc[genes_type[ii][jj]].T, yerr=df_std.loc[genes_type[ii][jj]].T, label=genes_type[ii][jj], marker='o')
        plt.legend(fontsize=16)
        plt.title(title[ii], fontsize=20)
        plt.grid()
        plt.show()
# ______________________________________________________________________________
# ______________________________________________________________________________


# ________________________________ IGNITE ______________________________________
def scatter_modelSelection(lN_prec_sel_MCIf, lN_dist_MCIf, color='navy', label=""):
    """ FCI vs CMD scatter plot"""
    plt.scatter(lN_prec_sel_MCIf, lN_dist_MCIf, color=color, s=50, alpha=0.5, label=label)
    plt.xlabel("FCI", fontsize=16)
    plt.ylabel("CMD", fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)
    # plt.title("LogNorm data", fontsize=20)
    
def plot_histogram(data, bins, xlabel, title, xlim=(0, 1)):
    """ Plot the pdf of the data in Ising_InferenceMethods 
    Args:
        data (list): list of data to plot
        bins (int): number of bins
        xlabel (string): label of the x axis
        title (string): title of the plot
        xlim (tuple): limits of the x axis
    
    Used for CMD and FCI plot    
    """
    plt.figure(figsize=(9, 6))
    sns.histplot(data, bins=bins, stat="density", color='darkred', alpha=0.4)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xlim(xlim)
    plt.grid(True)
    plt.tight_layout()  
    
def plot_interaction_distribution(data, bin_edges, quantiles=(0.05, 0.5, 0.95), 
                                  title="Interaction Distribution", xlabel="Interaction value", 
                                  figsize=(9, 6), fontsize=14, legend_fontsize=14, title_fontsize=16):
    """
    Plots a histogram pdf of the interaction distribution of a matrix (GRN) with quantile lines.
    Colors are chosen to be colorblind-friendly and suitable for paper publication.
    """

    # Set the style of the plot to be suitable for a paper
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # Define specific colors for the quantiles and median
    quantile_colors = ['orange', 'red', 'orange']  # Colors for the 5th percentile, median, and 95th percentile
    
    plt.figure(figsize=figsize)
    sns.histplot(data, bins=bin_edges, stat="density", color='blue', alpha=0.4)  # A colorblind-friendly blue for the histogram
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    plt.title(title, fontsize=title_fontsize)

    # Plot the quantile lines
    for q, color in zip(quantiles, quantile_colors):
        value = np.quantile(data, q)
        plt.axvline(x=value, color=color, linestyle='--', linewidth=2,
                    label=f'{int(q*100)}th percentile = {value:.2f}')

    plt.legend(title='Quantiles', title_fontsize=legend_fontsize, fontsize=legend_fontsize, frameon=True, fancybox=True)
    plt.tight_layout()

    plt.show()
# ______________________________________________________________________________
# ______________________________________________________________________________


class MidpointNormalize(pltcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        pltcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


from matplotlib.font_manager import FontProperties
    
def plotmat(m, fig, ax, ax_names, text, fix = False, cmap = 'RdBu_r'):
    plt.rcParams['font.family'] = 'Avenir'
    plt.rcParams['text.usetex'] = False
    # Crea un oggetto FontProperties per Avenir
    avenir_font = FontProperties(family='Avenir', size=16)

    # Imposta i font degli xticks
    for label in ax.get_xticklabels():
        label.set_fontproperties(avenir_font)

    # Imposta i font degli yticks
    for label in ax.get_yticklabels():
        label.set_fontproperties(avenir_font)
    matplotlib.rc('text', usetex=True)
    sns.set(font='Avenir')
    sns.set(style="white")
    if fix == True:
        img = ax.imshow(m, cmap = cmap, clim=(-1, 1),
                    norm = MidpointNormalize(midpoint=0,
                                             vmin=-1,
                                             vmax=1)
                   )
    else:
        lim_val = max(np.abs(np.nanmin(m)), np.nanmax(m))
        
        img = ax.imshow(m, cmap = cmap, clim=(-lim_val, lim_val),
                        norm = MidpointNormalize(midpoint=0,
                                                vmin = -lim_val,
                                                vmax =  lim_val)
                    )
    
    # cbarM = fig.colorbar(img, ax = ax)
    cbarM = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
    cbarM.set_label(r'$J_{ij}$', rotation = -90, labelpad = 20, fontsize = 22)

    # tick_min = np.floor(np.nanmin(np.ndarray.flatten(m)))
    # tick_max = np.ceil(np.nanmax(np.ndarray.flatten(m)))
    # tick = np.nanmax([np.abs(tick_min), np.abs(tick_max)])
    # print(tick)
    # if tick<=1:
        # ticks = np.arange(-0.8, .9, 0.2) # for IGNITE
        # ticks = np.arange(-1.3, 1.5, 0.2) # for CM
        # ticks = np.arange(-.6, .7, 0.2) # for CM
        
        
    # else:
    # ticks = np.arange(-tick - (-tick % 10), tick + (tick % 10), 10)
        # ticks = np.arange(-1.3, 1.5, 0.2) # for CM
 
    # cbarM.set_ticks(ticks[1:-1])
    cbarM.ax.tick_params(labelsize = 20)
    cbarM.ax.set_yticklabels(cbarM.ax.get_yticklabels(), fontname='Avenir')

    ax.set_xticks(np.arange(0,np.shape(np.array(m))[0]))
    ax.set_xticklabels(ax_names, rotation='vertical', fontsize=18)
    ax.set_yticks(np.arange(0,np.shape(np.array(m))[0]))
    ax.set_yticklabels(ax_names, fontsize=18)
    fig.suptitle(text, fontsize=24)


