import pandas as pd
import numpy as np

# # ________________________ FOR DATA PROCESSING ________________________________

def bulk_scRNA(df, time_sep):
    """ Separate the scRNA data in time steps as it would be a BULK data
    Args:
        df (dataframe): dataframe with the gene expression
        time_sep (list of ints): list with the index of the time steps"""
    df_00 = df.iloc[:,:time_sep[0]]
    df_06 = df.iloc[:, time_sep[0]: np.sum(time_sep[:2])]
    df_12 = df.iloc[:, np.sum(time_sep[:2]): np.sum(time_sep[:3])]
    df_24 = df.iloc[:, np.sum(time_sep[:3]): np.sum(time_sep[:4])]
    df_48 = df.iloc[:, np.sum(time_sep[:4]): np.sum(time_sep[:5])]                        
    return([df_00, df_06, df_12, df_24, df_48])

# ______________________________________________________________________________
# ______________________________________________________________________________