import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import matplotlib.colors as pltcolors
import seaborn as sns
import matplotlib

from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize
sys.path.append('../')


plt.rcParams['text.usetex'] = False


# -------------------------------------------------------------------------------------------------
# ------------------------------------- KO: knockout  functions -----------------------------------
# -------------------------------------------------------------------------------------------------

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

def KO_diff_sim(KO_avg, wt_avg):
    """
    Computes the difference in average activity between knockout (KO) and wild type (WT).

    Args:
        KO_avg (numpy array): Average activity of KO.
        wt_avg (numpy array): Average activity of WT.

    Returns:
        numpy array: Differences in average activity.
    """
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




### Fraction of differences in agremeent with experimental data
def fraction_agreement(diff_sim_norm, log2FC_exp_norm, genes_KOs, threshold = 0.05):
    common_low = np.intersect1d(np.where(diff_sim_norm<=-threshold)[0], np.where(log2FC_exp_norm<=-threshold)[0])
    frac_low = len(common_low)/len(genes_KOs)
    common_high = np.intersect1d(np.where(diff_sim_norm>=threshold)[0], np.where(log2FC_exp_norm>=threshold)[0])
    frac_high = len(common_high)/len(genes_KOs)
    common_zero = np.intersect1d(np.where(np.abs(diff_sim_norm)<threshold)[0],  np.where(np.abs(log2FC_exp_norm)<threshold)[0])
    frac_zero = len(common_zero)/len(genes_KOs)
    return frac_low, frac_high, frac_zero, sum([frac_low, frac_high, frac_zero])



def KO_heat_comparison_and_agreement(diff, exp_data, title, KO_genes_order, Norm=True, threshold=0.05):
    """
    Creates a heatmap comparing simulated and experimental data and calculates the fraction of agreement.

    Args:
        diff (array): Simulated data.
        exp_data (array): Experimental data.
        title (str): Title of the heatmap.
        KO_genes_order (list): Order of genes.
        Norm (bool, optional): Normalize data or not. Defaults to True.
        threshold (float, optional): Threshold for agreement calculation. Defaults to 0.05.

    Returns:
        tuple: Fraction of agreement and the heatmap data.
    """
    # Create a matrix merging diff and exp_data
    if Norm:
        data = np.array([diff/np.max(np.abs(diff)), exp_data/np.max(np.abs(exp_data))])
    else:
        data = np.array([diff, exp_data])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(24,3), gridspec_kw={'height_ratios': [1], 'width_ratios': [1.7,0.1]})
    im0 = ax[0].imshow(data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

    # Adding text to the heatmap
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            ax[0].text(i, j, round(data[j, i], 1), fontsize=25, ha="center", va="center", color="k")

    # Setting axis labels
    ax[0].set_yticks(np.arange(2))
    ax[0].set_yticklabels(['Generated', 'Experimental'], fontsize=35)
    ax[0].set_xticks(np.arange(exp_data.shape[0]))
    ax[0].set_xticklabels(KO_genes_order, fontsize=35, rotation='vertical')

    # Colorbar settings
    cbar = fig.colorbar(im0, cax=ax[1], orientation='vertical', pad=0.1, shrink=0.5)
    cbar.ax.set_aspect(20)
    cbar.set_label("Normalized \n KO and WT \ndifference", fontsize=35)
    cbar.ax.tick_params(axis='y', labelsize=25)

    plt.title(title)
    plt.show()

    # Fraction of agreement calculation
    diff_sim_norm, log2FC_exp_norm = data
    common_low = np.intersect1d(np.where(diff_sim_norm <= -threshold)[0], np.where(log2FC_exp_norm <= -threshold)[0])
    frac_low = len(common_low)/len(KO_genes_order)
    common_high = np.intersect1d(np.where(diff_sim_norm >= threshold)[0], np.where(log2FC_exp_norm >= threshold)[0])
    frac_high = len(common_high)/len(KO_genes_order)
    common_zero = np.intersect1d(np.where(np.abs(diff_sim_norm) < threshold)[0], np.where(np.abs(log2FC_exp_norm) < threshold)[0])
    frac_zero = len(common_zero)/len(KO_genes_order)

    frac_agreement = sum([frac_low, frac_high, frac_zero])
    print("Fraction of genes with agreement: ", frac_agreement)

    return (frac_low, frac_high, frac_zero, frac_agreement), data



def KO_heat_comparison_and_agreement_vertical(diff, exp_data, title, KO_genes_order, max_sim_diff, max_exp_lf, Norm=True, threshold=0.05, gene_KO=None):
    """
    Creates a vertical heatmap comparing simulated and experimental data and calculates the fraction of agreement.

    Args:
        diff (array): Simulated data.
        exp_data (array): Experimental data.
        title (str): Title of the heatmap.
        KO_genes_order (list): Order of genes.
        Norm (bool, optional): Normalize data or not. Defaults to True.
        threshold (float, optional): Threshold for agreement calculation. Defaults to 0.05.

    Returns:
        tuple: Fraction of agreement and the heatmap data.
    """
    KO_genes = KO_genes_order.copy()

    data_o = np.vstack((diff, exp_data)).T
    # remove from data the column with the index of the KO_genes==gene_KO
#     data_o = np.delete(data_o, np.where(np.array(KO_genes)==gene_KO)[0][0], axis=0)
    print(data_o.shape)
    if Norm:
        diff_normalized = data_o[:,0] / max_sim_diff
#         print(diff_normalized)
        exp_data_normalized = data_o[:,1] / max_exp_lf

    else:
        diff_normalized = diff
        exp_data_normalized = exp_data
    data = np.vstack((diff_normalized, exp_data_normalized)).T
        
    # remove it also from KO_genes
#     KO_genes.remove(gene_KO)
    KO_genes = [gene for gene in KO_genes if gene not in gene_KO]


    # Plotting
    plt.figure(figsize=(3, 8))
    sns.heatmap(data, annot=True, fmt=".3f", yticklabels=KO_genes, xticklabels=['Generated', 'Experimental'], cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

    # Fraction of agreement calculation
    common_low = np.intersect1d(np.where(diff_normalized <= -threshold)[0], np.where(exp_data_normalized <= -threshold)[0])
    frac_low = len(common_low)/len(KO_genes)
    common_high = np.intersect1d(np.where(diff_normalized >= threshold)[0], np.where(exp_data_normalized >= threshold)[0])
    frac_high = len(common_high)/len(KO_genes)
    common_zero = np.intersect1d(np.where(np.abs(diff_normalized) < threshold)[0], np.where(np.abs(exp_data_normalized) < threshold)[0])
    frac_zero = len(common_zero)/len(KO_genes)

    frac_agreement = sum([frac_low, frac_high, frac_zero])
    print("Fraction of genes with agreement: ", frac_agreement)

    return (frac_low, frac_high, frac_zero, frac_agreement), data


# ---------------------------------------------------------------------
# ------------------------ Plot matrices ------------------------------
# ---------------------------------------------------------------------
#class MidpointNormalize(pltcolors.Normalize):
#    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#        self.midpoint = midpoint
#        pltcolors.Normalize.__init__(self, vmin, vmax, clip)
#
#    def __call__(self, value, clip=None):
#        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plotmat(m, fig, ax, ax_names, text, fix = False, cmap = 'RdBu_r'):
    avenir_font = FontProperties(family='sans-serif', size=16) 

    for label in ax.get_xticklabels():
        label.set_fontproperties(avenir_font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(avenir_font)
        
    # matplotlib.rc('text', usetex=True)
    sns.set(style="white")

    if fix == True:
        img = ax.imshow(m, cmap = cmap, clim=(-1, 1),
                        norm = MidpointNormalize(midpoint=0,
                                                 vmin=-1,
                                                 vmax=1)
                       )
    else:
        lim_val = 0.056 #max(np.abs(np.nanmin(m)), np.nanmax(m))
        
        img = ax.imshow(m, cmap = cmap, clim=(-lim_val, lim_val),
                        norm = MidpointNormalize(midpoint=0,
                                                 vmin = -lim_val,
                                                 vmax =  lim_val)
                    )
    
    cbarM = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
    cbarM.set_label('$C_{ij}$', rotation = -90, labelpad = 20, fontsize = 20)

    tick_min = np.floor(np.nanmin(np.ndarray.flatten(m)))
    tick_max = np.ceil(np.nanmax(np.ndarray.flatten(m)))
    tick = np.nanmax([np.abs(tick_min), np.abs(tick_max)])
    if tick<=1:
        #lim_val = max(np.abs(np.nanmin(m)), np.nanmax(m))
        ticks = np.arange(-lim_val - (-lim_val % (0.1*lim_val)), lim_val + (0.1*lim_val), lim_val*0.5)
    else:
        ticks = np.arange(-tick - (-tick % 10), tick + 10, 10)
    cbarM.set_ticks(ticks)
    cbarM.ax.tick_params(labelsize = 16)

    ax.set_xticks(np.arange(0,np.shape(np.array(m))[0]))
    ax.set_xticklabels(ax_names, rotation='vertical', fontsize=18)
    ax.set_yticks(np.arange(0,np.shape(np.array(m))[0]))
    ax.set_yticklabels(ax_names, fontsize=18)
    fig.suptitle(text, fontsize=24)


from matplotlib.patches import FancyArrowPatch
import networkx as nx

def visualize_graphSel(adj_matrix, node_names, naive_nodes, formative_nodes, committed_nodes, interactions=[], title=""):
    """ Funzione per visualizzare la rete delle interazioni correttamente inferite da una lista """
    # Crea un grafo diretto dalla matrice di adiacenza
    G = nx.DiGraph(adj_matrix)
    # Rinomina i nodi con i nuovi nomi
    G = nx.relabel_nodes(G, {i: node_name for i, node_name in enumerate(node_names)})

    # Crea un set di nodi unici coinvolti nelle interazioni
    involved_nodes = set()
    for interaction in interactions:
        node1, node2, _ = interaction.split(" ")
        involved_nodes.add(node1)
        involved_nodes.add(node2)

    # Filtra solo i nodi coinvolti nelle interazioni
    G = G.subgraph(involved_nodes)

    # Disegna il grafo usando il layout circolare
    plt.figure(figsize=(10,10))
    pos = nx.circular_layout(G)

    # Crea un dizionario di mappatura nodo-colore
    color_map = {node: "lightskyblue" if node in naive_nodes else "palegoldenrod" if node in formative_nodes else "salmon" if node in committed_nodes else "silver" for node in G.nodes() if node in involved_nodes}

    # Disegna i nodi con colori diversi
    nx.draw_networkx_nodes(G, pos, node_color=[color_map.get(node) for node in G.nodes()], node_size=2000)
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()}, font_size=18)

    # Disegna solo le interazioni date se esistono
    total_edges_drawn = 0
    for interaction in interactions:
        node1, node2, direction = interaction.split(" ")
        if node1 in G and node2 in G:  # Controlla se entrambi i nodi sono nel grafo
            edges_to_draw = None
            if direction == "1" and adj_matrix[list(node_names).index(node1), list(node_names).index(node2)] > 0:
                edge_color = 'r'
                edges_to_draw = (node1, node2)
                total_edges_drawn += 1
                
            elif direction == "-1" and adj_matrix[list(node_names).index(node1), list(node_names).index(node2)] < 0:
                edge_color = 'b'
                edges_to_draw = (node1, node2)
                total_edges_drawn += 1
            if edges_to_draw: 
                # Disegno personalizzato della freccia con FancyArrowPatch
                arrow = FancyArrowPatch(posA=pos[edges_to_draw[0]], posB=pos[edges_to_draw[1]],
                                        arrowstyle='simple,head_length=15,head_width=10,tail_width=3',
                                        color=edge_color,
                                        connectionstyle='arc3,rad=-0.05',
                                        mutation_scale=3,
                                        lw=1.0,
                                        alpha=0.7,
                                        shrinkA=0.5, shrinkB=0.5)
                plt.gca().add_patch(arrow)
    
    print(f"Total edges drawn: {total_edges_drawn}")


    
def to_adj_matrix(matrix, thr=0):
    """Converts an interaction matrix to an adjacency matrix.
    
    Values with absolute value below the threshold are set to 0.
    Negative values are set to -1, and positive values to +1.
    
    Args:
        matrix (np.ndarray): The interaction matrix.
        thr (float): The threshold for determining significance of values.
    
    Returns:
        np.ndarray: The adjacency matrix.
    """
    threshold = thr * np.max(np.abs(matrix))
    adj_matrix = np.zeros_like(matrix)

    adj_matrix[matrix < -threshold] = -1
    adj_matrix[matrix > threshold] = 1
    return(adj_matrix)


# -----------------------------------------------------------------------
# ------------------ check signs of known interactions ------------------
# -----------------------------------------------------------------------

def check_signs(adjacency_matrix, TPtrial_list):
    """
    Checks the number of correct signs in the interaction matrix based on a list of known interactions.

    Parameters:
    adjacency_matrix (pd.DataFrame): A pandas DataFrame representing the adjacency matrix.
    TPtrial_list (list): A list of known interactions in the format "Gene1 Gene2 Sign".

    Returns:
    float: The fraction of correct signs.
    """
    # Parse the TPtrial_list
    parsed_list = [item.split() for item in TPtrial_list]

    # Initialize a counter for correct signs
    correct_signs = 0

    for gene_from, gene_to, sign in parsed_list:
        if gene_from in adjacency_matrix.index and gene_to in adjacency_matrix.columns:
            # Compare sign from the list with the sign in the matrix
            if np.sign(adjacency_matrix.loc[gene_from, gene_to]) == int(sign):
                correct_signs += 1

    # Calculate and return the fraction of correct signs
    return correct_signs / len(TPtrial_list)

# -----------------------------------------------------------------------
# ----------------------- adj matrix from GRN CO output -----------------
# -----------------------------------------------------------------------

def create_adjacency_matrix(links_info, genes_list, adj_i=[]):
    """
    Creates an adjacency matrix from a DataFrame of links between genes and optionally plots it.

    Parameters:
    links_info (pd.DataFrame): DataFrame containing the link information between genes.
    genes_list (list): List of genes to be included as indices and columns in the adjacency matrix.
    adj_i (pd.DataFrame): DataFrame containing the initial adjacency matrix from scATAC-seq data.
    adj_fig (bool): If True, display a heatmap of the adjacency matrix.

    Returns:
    pd.DataFrame: Adjacency matrix created from the provided data.
    """
    # Create an empty DataFrame with genes as indices and columns
    adjacency_matrix = pd.DataFrame(np.nan, index=genes_list, columns=genes_list)

    # Populate the adjacency matrix
    for idx, row in links_info.iterrows():
        if row['source'] in genes_list and row['target'] in genes_list:
            adjacency_matrix.loc[row['source'], row['target']] = row['coef_mean']

    # Replace NaN values with 0
    adjacency_matrix.fillna(0, inplace=True)

    return adjacency_matrix




