import pandas as pd
import numpy as np
import re
import scipy.linalg
import torch
import random
import logging
import argparse

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import make_scorer
from scipy.sparse import load_npz
from numpy.linalg import LinAlgError



def get_pops_args(argv=None):
    parser = argparse.ArgumentParser(description='Compute the Kernel-Polygenic Priority Score (K-PoPS).')
    parser.add_argument("--gene_annot_path", help="Path to tab-separated gene annotation file. Must contain ENSGID, CHR, and TSS columns")
    parser.add_argument("--kernel_mat_prefix", help="Prefix to the kernel file. There must be a kernel.bin file and kernel.genes file")
    parser.add_argument("--magma_prefix", help="Prefix to the gene-level association statistics outputted by MAGMA. There must be a .genes.out file and a .genes.raw file")
    parser.add_argument('--use_magma_covariates', dest='use_magma_covariates', action='store_true', help="(Default) Set this flag to project out MAGMA covariates before fitting")
    parser.add_argument('--ignore_magma_covariates', dest='use_magma_covariates', action='store_false', help="Set this flag to ignore MAGMA covariates")
    parser.set_defaults(use_magma_covariates=True)
    parser.add_argument("--y_path", help="Path to a custom target score. Use this if you want to fit something other than MAGMA. Must contain ENSGID and Score columns. Note that if --magma_prefix is set, then y_path will be ignored")
    parser.add_argument("--y_covariates_path", help="Optional path to covariates for custom target score provided in --y_path. Must contain ENSGID column followed by columns for each covariate")
    parser.add_argument("--y_error_cov_path", help="Optional path to error covariance for custom target score provided in --y_path. Must be provided SciPy .npz format or NumPy .npy format, and the rows/columns must directly correspond to the ordering provided in --y_path")
    parser.add_argument("--project_out_covariates_chromosomes", nargs="*", help="List chromosomes to consider when projecting out covariates. If not set, will use all chromosomes in --gene_annot_path by default")
    parser.add_argument('--project_out_covariates_remove_hla', dest='project_out_covariates_remove_hla', action='store_true', help="(Default) Set this flag to remove HLA genes before projecting out covariates")
    parser.add_argument('--project_out_covariates_keep_hla', dest='project_out_covariates_remove_hla', action='store_false', help="Set this flag to keep HLA genes when projecting out covariates")
    parser.set_defaults(project_out_covariates_remove_hla=True)
    parser.add_argument("--subset_features_path", help="Optional path to list of features (one per line) to subset to")
    parser.add_argument("--control_features_path", help="Optional path to list of features (one per line) to always include")
    parser.add_argument("--training_chromosomes", nargs="*", help="List chromosomes to consider when computing model coefficients. If not set, will use all chromosomes in --gene_annot_path by default")
    parser.add_argument('--training_remove_hla', dest='training_remove_hla', action='store_true', help="(Default) Set this flag to remove HLA genes when computing model coefficients")
    parser.add_argument('--training_keep_hla', dest='training_remove_hla', action='store_false', help="Set this flag to keep HLA genes when computing model coefficients")
    parser.set_defaults(training_remove_hla=True)
    parser.add_argument("--method", default="ridge", help="Regularization used when computing model coefficients, ridge (L2 penalty) by default. Also accepts lasso (L1 penalty) and linreg (no penalty)")
    parser.add_argument("--out_prefix", help="Prefix that results will be saved with. Will write out a .preds, .coefs, and .marginals file")
    parser.add_argument('--save_matrix_files', dest='save_matrix_files', action='store_true', help="Set this flag to also save the matrices used to compute the model coefficients (.traindata) and compute the PoP score (.matdata)")
    parser.add_argument('--no_save_matrix_files', dest='save_matrix_files', action='store_false', help="(Default) Set this flag to not save matrices used to compute model coefficients and PoP score")
    parser.set_defaults(save_matrix_files=False)
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility. 42 by default")
    parser.add_argument('--verbose', dest='verbose', action='store_true', help="Set this flag to get verbose output")
    parser.add_argument('--no_verbose', dest='verbose', action='store_false', help="(Default) Set this flag to silence output")
    parser.add_argument("--device", type = str, default="cpu", help="Use either cpu or cuda; for MacOS, use mps")
    parser.add_argument('--use_explain_mode', dest='use_explain_mode', action='store_true', help="Output the contribution scores for each gene")
    parser.add_argument('--suppress_explain_mode', dest='use_explain_mode', action='store_false', help="Suppress contribution scores")
    parser.set_defaults(use_explain_mode=True)
    parser.add_argument('--explain_n_gene', dest='explain_n_gene', type=int, default=10, help="Produce the network for top n genes (default: 10); Only effective when --use_explain_mode")
    parser.add_argument('--explain_n_contributor', dest='explain_n_contributor', type=int, default=10, help="Produce the network for top n contributors (default: 5); Only effective when --use_explain_mode")
    parser.add_argument('--explain_remove_hla', dest='explain_remove_hla', action='store_true', help="Ignore scores for HLA for both source and target; Only effective when --use_explain_mode")
    parser.add_argument('--explain_keep_hla', dest='explain_remove_hla', action='store_false', help="Ignore scores for HLA for both source and target; Only effective when --use_explain_mode")
    parser.set_defaults(explain_remove_hla=True)
    return parser.parse_args(argv)



### --------------------------------- GENERAL --------------------------------- ###

def natural_key(string_):
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def get_hla_genes(gene_annot_df):
    sub_gene_annot_df = gene_annot_df[gene_annot_df.CHR == "6"]
    sub_gene_annot_df = sub_gene_annot_df[sub_gene_annot_df.TSS >= 20 * (10 ** 6)]
    sub_gene_annot_df = sub_gene_annot_df[sub_gene_annot_df.TSS <= 40 * (10 ** 6)]
    return sub_gene_annot_df.index.values


### Returns as vector of booleans of length len(Y_ids)
def get_gene_indices_to_use(Y_ids, gene_annot_df, use_chrs, remove_hla):
    all_chr_genes_set = set(gene_annot_df[gene_annot_df.CHR.isin(use_chrs)].index.values)
    if remove_hla == True:
        hla_genes_set = set(get_hla_genes(gene_annot_df))
        use_genes = [True if (g in all_chr_genes_set) and (g not in hla_genes_set) else False for g in Y_ids]
    else:
        use_genes = [True if g in all_chr_genes_set else False for g in Y_ids]
    return np.array(use_genes)


def project_out_covariates(Y, covariates, Y_ids, gene_annot_df, project_out_covariates_Y_gene_inds):
    ### If covariates doesn't contain intercept, add intercept
    if not np.isclose(covariates.var(axis=0), 0).any():
        covariates = np.hstack((covariates, np.ones((covariates.shape[0], 1))))
    X_train, y_train = covariates[project_out_covariates_Y_gene_inds], Y[project_out_covariates_Y_gene_inds]
    reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    Y_proj = Y - reg.predict(covariates)
    return Y_proj

def get_indices_in_target_order(ref_list, target_names):
    ref_to_ind_mapper = {}
    for i, e in enumerate(ref_list):
        ref_to_ind_mapper[e] = i
    return np.array([ref_to_ind_mapper[t] for t in target_names])




def krr_fit_loocv_from_K(K, y, lam, jitter=1e-6):
    """
    Exact Kernel Ridge Regression LOOCV using hat matrix formulation.

    Args:
        K: (n, n) PSD kernel matrix
        y: (n,) or (n, m)
        lam: regularization parameter
        jitter: small value added to diagonal for numerical stability

    Returns:
        LOOCV mean squared error
    """
    n = K.shape[0]
    I_n = torch.eye(n, device=K.device, dtype=K.dtype)
    K_reg = K + (lam + jitter) * I_n
    H = torch.linalg.solve(K_reg.T, K.T).T
    denom = 1 - torch.diag(H)
    denom = torch.clamp(denom, min=1e-8)

    # Support multi-output y
    errors = (y - H @ y) / denom.unsqueeze(-1) if y.ndim == 2 else (y - H @ y) / denom

    return torch.mean(errors ** 2)


def krr_loocv_gridsearch(K, y, lambdas, jitter=0.0):
    scores = []
    for lam in lambdas:
        mse = krr_fit_loocv_from_K(K, y, lam, jitter=jitter)
        scores.append(mse)
    
    scores = torch.stack(scores)  # shape: (len(lambdas),)
    best_idx = torch.argmin(scores)
    best_lam = lambdas[best_idx].item()
    best_score = torch.min(scores)
    
    n = K.shape[0]
    I_n = torch.eye(n, device=K.device, dtype=K.dtype)
    K_reg = K + (best_lam + jitter) * I_n
    alpha = torch.linalg.solve(K_reg, y)
    yhat = K @ alpha
    
    return(yhat, alpha, best_lam, best_score)


### --------------------------------- READING DATA --------------------------------- ###

def read_gene_annot_df(gene_annot_path):
    gene_annot_df = pd.read_csv(gene_annot_path, sep = "\s+").set_index("ENSGID")
    gene_annot_df["CHR"] = gene_annot_df["CHR"].astype(str)
    return gene_annot_df


def munge_magma_covariance_metadata(magma_raw_path):
    gene_metadata = []
    with open(magma_raw_path) as f:
        ### Get all lines
        lines = list(f)[2:]
        lines = [np.asarray(line.strip('\n').split(' ')) for line in lines]
        ### Check that chromosomes are sequentially ordered
        all_chroms = np.array([l[1] for l in lines])
        all_seq_breaks = np.where(all_chroms[:-1] != all_chroms[1:])[0]
        assert len(all_seq_breaks) == len(set(all_chroms)) - 1, "Chromosomes are not sequentially ordered."
        ### Get starting chromosome and set up temporary variables
        curr_chrom = lines[0][1]
        curr_ind = 0
        num_genes_in_chr = sum([1 for line in lines if line[1] == curr_chrom])
        curr_gene_metadata = []
        for line in lines:
            ### If we move to a new chromosome, we reset everything
            if line[1] != curr_chrom:
                ### Symmetrize and save
                gene_metadata.append(curr_gene_metadata)
                ### Reset
                curr_chrom = line[1]
                curr_ind = 0
                num_genes_in_chr = sum([1 for line in lines if line[1] == curr_chrom])
                curr_gene_metadata = []
            ### Add metadata; GENE, NSNPS, NPARAM, MAC
            curr_gene_metadata.append([line[0], float(line[4]), float(line[5]), float(line[7])])
            if len(line) > 9:
                ### Add covariance
                gene_corrs = np.array([float(c) for c in line[9:]])
            curr_ind += 1
        ### Save last piece
        gene_metadata.append(curr_gene_metadata)
    gene_metadata = pd.DataFrame(np.vstack(gene_metadata), columns=["GENE", "NSNPS", "NPARAM", "MAC"])
    gene_metadata.NSNPS = gene_metadata.NSNPS.astype(np.float64)
    gene_metadata.NPARAM = gene_metadata.NPARAM.astype(np.float64)
    gene_metadata.MAC = gene_metadata.MAC.astype(np.float64)
    return gene_metadata



def get_hla_gene_indices(Y_ids, gene_annot_df):
    hla_genes_set = set(get_hla_genes(gene_annot_df))
    hla_genes = [True if g in hla_genes_set else False for g in Y_ids]
    return np.array(hla_genes)


### Returns as vector of booleans of length len(Y_ids)
def get_gene_indices_to_use(Y_ids, gene_annot_df, use_chrs, remove_hla):
    all_chr_genes_set = set(gene_annot_df[gene_annot_df.CHR.isin(use_chrs)].index.values)
    if remove_hla == True:
        hla_genes_set = set(get_hla_genes(gene_annot_df))
        use_genes = [True if (g in all_chr_genes_set) and (g not in hla_genes_set) else False for g in Y_ids]
    else:
        use_genes = [True if g in all_chr_genes_set else False for g in Y_ids]
    return np.array(use_genes)

    

def build_control_covariates(metadata):
    genesize = metadata.NPARAM.values
    genedensity = metadata.NPARAM.values/metadata.NSNPS.values
    inverse_mac = 1.0/metadata.MAC.values
    cov = np.stack((genesize, np.log(genesize), genedensity, np.log(genedensity), inverse_mac, np.log(inverse_mac)), axis=1)
    cov_df = pd.DataFrame(cov, columns=["gene_size", "log_gene_size", "gene_density", "log_gene_density", "inverse_mac", "log_inverse_mac"])
    cov_df["GENE"] = metadata.GENE.values
    cov_df = cov_df.loc[:,["GENE", "gene_size", "log_gene_size", "gene_density", "log_gene_density", "inverse_mac", "log_inverse_mac"]]
    cov_df = cov_df.set_index("GENE")
    return cov_df

def read_magma(magma_prefix, use_magma_covariates):
    ### Get Y and Y_ids
    magma_df = pd.read_csv(magma_prefix + ".genes.out", sep = "\s+")
    Y = magma_df.ZSTAT.values
    Y_ids = magma_df.GENE.values

    if use_magma_covariates is not None:
        ### Get covariates and error_cov
        gene_metadata = munge_magma_covariance_metadata(magma_prefix + ".genes.raw")
        cov_df = build_control_covariates(gene_metadata)
        ### Process
        assert (cov_df.index.values == Y_ids).all(), "Covariate ids and Y ids don't match."
        covariates = cov_df.values
    
    if use_magma_covariates == False:
        covariates = None
    return Y, covariates, Y_ids


def load_kernel_matrix(kernel_mat_prefix):
    K_genes = np.loadtxt(kernel_mat_prefix + ".genes", dtype=str).flatten()
    K = np.memmap(kernel_mat_prefix + ".bin", dtype=np.float32, mode="r", 
                  shape=(len(K_genes), len(K_genes)), order="C").astype(np.float32)
    return (K, K_genes)

def build_training(K, K_genes, Y, Y_ids, training_Y_gene_inds):
    ### Get training Y
    training_genes = Y_ids[training_Y_gene_inds]
    sub_Y = Y[training_Y_gene_inds]
    
    ### Get training X
    K_train_inds = get_indices_in_target_order(K_genes, training_genes)
    K_train = K[K_train_inds][:, K_train_inds]
    assert (K_genes[K_train_inds] == training_genes).all(), "Something went wrong. This shouldn't happen."
    sub_Y = sub_Y - sub_Y.mean()

    return K_train.astype(np.float32), sub_Y.astype(np.float32), K_train_inds, training_genes
    

def compute_coefficients(K_train, Y_train, training_genes, device):
    K_train = torch.tensor(K_train, device = device)
    Y_train = torch.tensor(Y_train, device = device)
    
    lams = torch.logspace(0, 10, 21, device=device, dtype=torch.float32)
    
    _, coef_, lam_, best_score = krr_loocv_gridsearch(K_train, Y_train, lams, jitter=1e-10)

    
    coefs_df = pd.DataFrame([["METHOD", "KernelCV"],
                             ["SELECTED_CV_ALPHA", lam_],
                             ["BEST_CV_SCORE", -best_score.cpu().numpy()]])
    
    coefs_df = pd.concat([coefs_df, pd.DataFrame([training_genes, coef_.cpu().numpy()]).T])
    coefs_df.columns = ["parameter", "beta"]
    coefs_df = coefs_df.set_index("parameter")
    coef_ = coef_.cpu().numpy()
    
    return (coefs_df, coef_)




def pops_predict(K, coef_, K_genes, K_train_inds, explaining_Y_gene_inds, 
                 use_explain_mode, explain_n_gene, explain_n_contributor):
    # prediction is for every genes, including HLA
    attr = K[:, K_train_inds] * coef_
    pred = np.sum(attr, axis=1)
    preds_df = pd.DataFrame([K_genes, pred]).T
    preds_df.columns = ["ENSGID", "PoPS_Score"]
    
    if use_explain_mode:
        training_genes = K_genes[K_train_inds]
        explaining_genes = K_genes[explaining_Y_gene_inds]
        attr_sub = attr[explaining_Y_gene_inds] # subset to genes of interest
        pred_sub = np.sum(attr_sub, axis=1)
        lead_genes_idx = np.argsort(pred_sub)[-explain_n_gene:][::-1]
        lead_contribs_idx = [np.argsort(attr_sub[lead_gene])[-explain_n_contributor:][::-1] for lead_gene in lead_genes_idx]
        netwk = []
        for i in range(explain_n_gene):
            for j in range(explain_n_contributor):
                lead_gene_idx = lead_genes_idx[i]
                lead_contrib_idx = lead_contribs_idx[i][j]
                netwk.append((explaining_genes[lead_genes_idx[i]], 
                              training_genes[lead_contribs_idx[i][j]], 
                              attr_sub[lead_gene_idx, lead_contrib_idx]))
        netwk = pd.DataFrame(netwk, columns = ["gene", "contributor", "score"])
    else: 
        netwk = None

    return preds_df, netwk
        
    


def main(config_dict):
    ### --------------------------------- Basic settings --------------------------------- ###
    ### Set logging settings
    if config_dict["verbose"]:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
        logging.info("Verbose output enabled.")
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")
    ### Set random seeds
    np.random.seed(config_dict["random_seed"])
    random.seed(config_dict["random_seed"])

    ### Display configs
    logging.info("Config dict = {}".format(str(config_dict)))

        ### --------------------------------- Reading/processing data --------------------------------- ###
    gene_annot_df = read_gene_annot_df(config_dict["gene_annot_path"])
    ### If chromosome arguments are None, replace their values in config_dict with all chromosomes
    all_chromosomes = sorted(gene_annot_df.CHR.unique(), key=natural_key)
    all_chromosome_training = False
    
    if config_dict["project_out_covariates_chromosomes"] is None:
        config_dict["project_out_covariates_chromosomes"] = all_chromosomes
        logging.info("--project_out_covariates_chromosomes is None, defaulting to all chromosomes")

    if config_dict["training_chromosomes"] is None:
        config_dict["training_chromosomes"] = all_chromosomes
        config_dict["testing_chromosomes"]  = all_chromosomes
        logging.info("--training_chromosomes is None, defaulting to all chromosomes")
    else:
        config_dict["testing_chromosomes"] = list(set(all_chromosomes) - set(config_dict["training_chromosomes"]))
        if config_dict["testing_chromosomes"] is None:
            config_dict["testing_chromosomes"]  = all_chromosomes
    
    
    ### Make sure all chromosome arguments are fully contained in gene_annot_df's chromosome list
    assert set(config_dict["project_out_covariates_chromosomes"]).issubset(all_chromosomes), "Invalid --project_out_covariates_chromosomes argument."
    assert set(config_dict["training_chromosomes"]).issubset(all_chromosomes), "Invalid --training_chromosomes argument."

    
    if config_dict["magma_prefix"] is not None:
        logging.info("MAGMA scores provided, loading MAGMA.")
        Y, covariates,Y_ids = read_magma(config_dict["magma_prefix"],
                                                     config_dict["use_magma_covariates"])
        if config_dict["use_magma_covariates"] == True:
            logging.info("Using MAGMA covariates.")
        else:
            logging.info("Ignoring MAGMA covariates.")
    elif config_dict["y_path"] is not None:
        # logging.info("Reading scores from {}.".format(config_dict["y_path"]))
        # if config_dict["y_covariates_path"] is not None:
        #     logging.info("Reading covariates from {}.".format(config_dict["y_covariates_path"]))
        # if config_dict["y_error_cov_path"] is not None:
        #     logging.info("Reading error covariance from {}.".format(config_dict["y_error_cov_path"]))
        # ### Note that we do not regularize covariance matrix provided in y_error_cov_path. It will be used as is.
        # Y, covariates, error_cov, Y_ids = read_from_y(config_dict["y_path"],
        #                                               config_dict["y_covariates_path"],
        #                                               config_dict["y_error_cov_path"])
        pass
    else:
        raise ValueError("At least one of --magma_prefix or --y_path must be provided (--magma_prefix overrides --y_path).")


    ## HLA index, if ever used
    # if config_dict["project_out_covariates_remove_hla"] or config_dict["training_remove_hla"] or config_dict["explain_remove_hla"]:
    #     hla_inds = get_hla_gene_indices(Y_ids, gene_annot_df)
    # else: 
    #     hla_inds = None

    
    ### Get projection, feature selection, and training genes
    project_out_covariates_Y_gene_inds = get_gene_indices_to_use(Y_ids,
                                                                 gene_annot_df,
                                                                 config_dict["project_out_covariates_chromosomes"],
                                                                 config_dict["project_out_covariates_remove_hla"])


    ### Project out covariates if using
    if covariates is not None:
        logging.info("Projecting {} covariates out of target scores using genes on chromosome {}. HLA region {}."
                     .format(covariates.shape[1],
                             ", ".join(sorted(gene_annot_df.loc[Y_ids[project_out_covariates_Y_gene_inds]].CHR.unique(), key=natural_key)),
                             "removed" if config_dict["project_out_covariates_remove_hla"] else "included"))
        Y_proj = project_out_covariates(Y,
                                        covariates,
                                        Y_ids,
                                        gene_annot_df,
                                        project_out_covariates_Y_gene_inds)
    else:
        Y_proj = Y
    

    ### --------------------------------- Training --------------------------------- ###
    training_Y_gene_inds = get_gene_indices_to_use(Y_ids,
                                                   gene_annot_df,
                                                   config_dict["training_chromosomes"],
                                                   config_dict["training_remove_hla"])
    

    K, K_genes = load_kernel_matrix(config_dict["kernel_mat_prefix"])
    K_train, Y_train, K_train_inds, training_genes = build_training(K, K_genes, Y_proj, Y_ids, training_Y_gene_inds)


    
    logging.info("K dimensions = {}. Y dimensions = {}".format(K_train.shape, Y_train.shape))
    coefs_df, coef_ = compute_coefficients(K_train, Y_train, training_genes, config_dict["device"])
    
    
    ### --------------------------------- Prediction --------------------------------- ###
    logging.info("Computing PoPS scores.")
  

    explaining_Y_gene_inds = get_gene_indices_to_use(Y_ids,
                                               gene_annot_df,
                                               config_dict["testing_chromosomes"],
                                               config_dict["explain_remove_hla"])

    
    preds_df, netwk = pops_predict(K, coef_, K_genes, K_train_inds, explaining_Y_gene_inds, 
                 config_dict["use_explain_mode"], 
                  config_dict["explain_n_gene"], 
                  config_dict["explain_n_contributor"])

    
    
    preds_df = preds_df.merge(pd.DataFrame(np.array([Y_ids, Y]).T, columns=["ENSGID", "Y"]),
                              how="left",
                              on="ENSGID")
    
    if covariates is not None:
        preds_df = preds_df.merge(pd.DataFrame(np.array([Y_ids, Y_proj]).T, columns=["ENSGID", "Y_proj"]),
                                  how="left",
                                  on="ENSGID")
        preds_df["project_out_covariates_gene"] = preds_df.ENSGID.isin(Y_ids[project_out_covariates_Y_gene_inds])

    preds_df["training_gene"] = preds_df.ENSGID.isin(Y_ids[training_Y_gene_inds])

    ### --------------------------------- Save --------------------------------- ###
    logging.info("Writing output files.")
    preds_df.to_csv(config_dict["out_prefix"] + ".preds", sep="\t", index=False)
    coefs_df.to_csv(config_dict["out_prefix"] + ".coefs", sep="\t")
    if config_dict["use_explain_mode"]:
        netwk.to_csv(config_dict["out_prefix"] + ".netwk", sep="\t", index=False)
    

    if config_dict["save_matrix_files"] == True:
        logging.info("Saving matrix files as well.")
        pd.DataFrame(np.hstack((Y_train.reshape(-1,1), K_train)),
                     index=Y_ids[training_Y_gene_inds],
                     columns=["Y_train"] + list(training_genes)).to_csv(config_dict["out_prefix"] + ".traindata", sep="\t")

### Main
if __name__ == '__main__':
    args = get_pops_args()
    config_dict = vars(args)
    main(config_dict)
    
    




    