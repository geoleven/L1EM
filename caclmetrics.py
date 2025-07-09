from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, median_absolute_error, root_mean_squared_error, explained_variance_score, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import argparse
import sys
from typing import List
import os
import matplotlib.pyplot as plt
import math
import re

# Extract lengths
def extract_length(locus, truth:bool=False):
    if not truth:
#        parts = locus.strip().split(':')
#        coords = parts[1].split('-')
#        start = int(coords[0])
#        end = int(coords[1].strip("__"))
#        return end - start + 1
        rs = re.search("([0-9]+)-([0-9]+)", locus)
#        if rs[1] != 3:
#          print("Regex problem")
#          print("Locus: " + locus)
#          exit
        if len(rs.groups()) != 2:
            print("Groups found are not of correct number!")
            print("Locus: "+ str(locus))
            exit
        res = int(rs.group(2)) - int(rs.group(1))
        return res
    else:
#        parts = locus.strip().strip("__").split('_')
#        start = int(parts[4])
#        end = int(parts[-1])
#        return end - start + 1
        rs = re.search("([0-9]+)-([0-9]+)", locus)
        if len(rs.groups()) != 2:
            print("Groups found are not of correct number!")
            print("Locus: "+ str(locus))
            exit
        res = int(rs.group(2)) - int(rs.group(1))
        return res +1

def calcTPM_L1EM(L1EM_res:pd.DataFrame): 
    df = pd.read_csv(L1EM_res, sep='\t', engine='python')
    df.columns = [col.strip() for col in df.columns]
    df['length'] = df['family.category.locus.strand'].apply(extract_length)
    count_cols = df.columns[1:6]
    df['total_counts'] = df[count_cols].sum(axis=1)
    df['rpk'] = df['total_counts'] / (df['length'] / 1000)
    rpk_sum = df['rpk'].sum()
    df['tpm'] = df['rpk'] / rpk_sum * 1e6
    df['family.category.locus.strand'] = df['family.category.locus.strand'].str.replace(".", "_") 
    df['family.category.locus.strand'] = df['family.category.locus.strand'].str.replace(":", "_") 
    df['family.category.locus.strand'] = df['family.category.locus.strand'].str.replace("-", "_") 
    df['family.category.locus.strand'] = df['family.category.locus.strand'].str.replace("+", "_") 
    df['family.category.locus.strand'] = df['family.category.locus.strand'].str.split("__").str[0] 
    tpm_df = df[['family.category.locus.strand', 'tpm']]
    return tpm_df

def calcTPM_truth(cmpr_res:pd.DataFrame):
    df = pd.read_csv(cmpr_res, sep='\t')
    df['length'] = df['#transcript_id'].apply(lambda x: extract_length(x, truth=True))
    df['rpk'] = df['cnt'] / (df['length'] / 1000) 
    rpk_sum = df['rpk'].sum()
    df['true_tpm'] = df['rpk'] / rpk_sum * 1e6
    df['#transcript_id'] = df['#transcript_id'].str.replace('-','_') 
    df['#transcript_id'] = df['#transcript_id'].str.replace('+','_') 
    df['#transcript_id'] = df['#transcript_id'].str.replace('.','_') 
    df['#transcript_id'] = df['#transcript_id'].str.split('__').str[0] 
    tpm_df = df[['#transcript_id', 'true_tpm']]
    return tpm_df

def org_values(truth:pd.DataFrame, pd_pred:pd.DataFrame, outpath:str=None, pos_only=False):
    pd_true = truth.rename(columns={'#transcript_id':'locus'})
    pd_pred = pd_pred.rename(columns={'family.category.locus.strand':'locus'})
    merged = pd.merge(pd_true, pd_pred, on='locus', how='outer')
    print(merged[merged['true_tpm'].isna()])
    merged = merged.fillna(0.0)
    #merged = merged.fillna(0)
    if pos_only:
        merged = merged[~merged['locus'].str.contains('_0_')]
    array1 = merged['true_tpm']
    array2 = merged['tpm']
    if outpath:
        merged.to_csv(outpath, sep='\t', index=False)
    return array1, array2

def calc_metrics(array_true:List, array_pred:List):
    y_true = np.array(array_true)
    y_pred = np.array(array_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mdae = median_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    return r2, mae, mdae, rmse, mape, evs

def plot_tpm(df:pd.DataFrame, plotpath:str, truth:bool=True):
    if truth:
        plt.hist(df.iloc[:,1], bins = 2000, color='skyblue', edgecolor='black')
    else:
        plt.hist(df.iloc[:,1], bins =2000, color='skyblue', edgecolor='black')
    plt.xlabel("TPM values")
    plt.ylabel("frequancy")
    plt.xlim(0,1000)
    plt.title("TPM distribution")
    plt.savefig(plotpath)

def main(method:str="EM", metrics:List[str]=["r2", "mae", "mdae", "rmse", "mape", "evs"]):
    parser = argparse.ArgumentParser(description='Process L1EM output and Camparee and calculate metrics (TPM)')
    parser.add_argument('-ic', help='Camparee input file path')
    parser.add_argument('-il', help='L1EM input file path')
    parser.add_argument('-o', help='Output file path with filename')
    parser.add_argument('-otemp', required=False, help='output merged df')
    parser.add_argument('-plot_tpm', action='store_true', required=False, help='plot TPM distribution')
    parser.add_argument('-truth', action='store_true', required=False, help='argument for plotting either true values or expected')
    args = parser.parse_args()
    # Get metrics after caclulating TPM
    pd_pred_tpm = calcTPM_L1EM(args.il)
    pd_true_tpm = calcTPM_truth(args.ic)
    array_true, array_pred = org_values(pd_true_tpm, pd_pred_tpm, args.otemp, pos_only=False)
    scores ={}
    scores[method]={}
    for score, name in zip(calc_metrics(array_true, array_pred), metrics):
        scores[method][name] = score
    score_df = pd.DataFrame([scores]).T
    csv = args.o
    if not os.path.exists(csv):
        score_df.to_csv(csv, sep='\t', index_label="method")
    else:
        score_df.to_csv(csv, mode='a', header=False, index_label="method")
    if args.plot_tpm and args.truth:
        plot_tpm(pd_true_tpm, args.o.replace("metrics.txt", "tpm_dist_true.png", True))
    if args.plot_tpm and not args.truth:
        plot_tpm(pd_pred_tpm, args.o.replace("metrics.txt", "tpm_dist_pred.png", False))
    if not args.plot_tpm:
        pass

if __name__=="__main__":
    main()
