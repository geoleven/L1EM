# -*- coding: utf-8 -*-
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import sys
import datetime
from scipy import sparse
from multiprocessing import Pool
import argparse
import torch
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.experimental.sparse import BCOO, bcoo_multiply_sparse, bcoo_dot_general
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal, init_to_uniform, init_to_median
from numpyro.optim import Adam, ClippedAdam
import os
from scipy.sparse import vstack

# Main calculation for the E step (unchanged)
def calculate_expcounts(G_of_R_pkl, X):
    with open(G_of_R_pkl, 'rb') as f:
        G_of_R = pickle.load(f)
    if G_of_R is None:
        return 0.0, 0.0
    L_of_R_mat = G_of_R.multiply(X)
    L_of_R = np.array(L_of_R_mat.sum(1))
    L_of_R_mat = L_of_R_mat[L_of_R[:, 0] >= 10**-200, :]
    L_of_R = L_of_R[L_of_R >= 10**-200]
    L_of_R_inv = sparse.csr_matrix(1.0 / L_of_R).transpose()
    exp_counts = L_of_R_mat.multiply(L_of_R_inv).sum(0)
    loglik = np.sum(np.log(L_of_R))
    if np.isfinite(loglik):
        return exp_counts, loglik
    else:
        return np.zeros(G_of_R.shape[1]), 0.0

def calculate_expcounts_chunk(input):
    G_of_R_pkl_list, X_len = input
    exp_counts = np.zeros(X_len.shape, dtype=np.float64)
    loglik = 0.0
    for G_of_R_pkl in G_of_R_pkl_list:
        this_exp_counts, this_loglik = calculate_expcounts(G_of_R_pkl, X_len)
        exp_counts += this_exp_counts
        loglik += this_loglik
    return exp_counts, loglik

def softmax(z):
    z_max = jnp.max(z)
    exp_z = jnp.exp(z - z_max)
    return exp_z / jnp.sum(exp_z)

def improved_schedule(step, n_steps, base_lr=1e-4):
    """
    Improved learning rate schedule for transcript abundance estimation
    Args:
        step: current step
        n_steps: total steps (10-15k)
        base_lr: base learning rate
    """
    # Longer warmup for complex initialization
    warmup_steps = max(2000, n_steps // 2)  # 20% of training for warmup
    if step < warmup_steps:
        # Smoother warmup with sqrt schedule
        warmup_progress = step / warmup_steps
        return base_lr * jnp.sqrt(warmup_progress)
    else:
        # Slower cosine decay with higher minimum
        progress = (step - warmup_steps) / (n_steps - warmup_steps)
        min_lr_ratio = 0.0001  # Don't decay below 0.1% of base_lr
        cosine_factor = 0.5 * (1 + jnp.cos(jnp.pi * progress))
        return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_factor)

def plateau_schedule(step, n_steps, base_lr=1e-4):
    """
    Plateau-based schedule
    """
    warmup_steps = n_steps // 10  # 30% warmup
    plateau_steps = n_steps // 2  # 50% plateau
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    elif step < warmup_steps + plateau_steps:
        # Plateau at full learning rate
        return base_lr
    else:
        # Gentle exponential decay
        decay_steps = step - warmup_steps - plateau_steps
        remaining_steps = n_steps - warmup_steps - plateau_steps
        decay_rate = 0.1  # Decay to 10% of original
        return base_lr * (decay_rate ** (decay_steps / remaining_steps))

def run_variational_bayes(G_of_R_list, TE_list, prefix,
                          lr=1e-3, n_steps=500,
                          prior_type="dirichlet", schedule="constant", sparsity_strength=0.1):
    if schedule == "constant":
        schedule_fn = lr
    elif schedule == "warmup":
        schedule_fn = lambda step: improved_schedule(step, n_steps, base_lr=1e-2)
    elif schedule == "plateau":
        schedule_fn = lambda step: plateau_schedule(step, n_steps, base_lr=1e-3)
    else:
        raise ValueError(f"Unknown scheduler: {schedule}")
    # Load transcript names
    TE_names = [line.strip().split('\t')[0] for line in open(TE_list)]
    L = len(TE_names)
    print("Computing expected read counts for multinomial VB...")
    # Initialize uniform transcript abundances (like EM does)
    X = np.ones(L, dtype=np.float64) / L
    # Compute expected counts using the same logic as EM algorithm
    total_expected_counts = np.zeros(L, dtype=np.float64)
    total_reads = 0
    for pkl_file in open(G_of_R_list):
        pkl_file = pkl_file.strip()
        with open(pkl_file, 'rb') as f:
            G_of_R = pickle.load(f)
        if G_of_R is not None:
            # Ensure G_of_R is in CSR format for efficient operations
            if not isinstance(G_of_R, sparse.csr_matrix):
                G_of_R = G_of_R.tocsr()
            # This is the same calculation as in calculate_expcounts()
            # G_of_R is (reads x transcripts) matrix of mapping probabilities
            # X is current transcript abundance estimate
            # L_of_R_mat[i,j] = G_of_R[i,j] * X[j] (probability read i came from transcript j)
            L_of_R_mat = G_of_R.multiply(X)
            # Convert to CSR for efficient row operations
            L_of_R_mat = L_of_R_mat.tocsr()
            # L_of_R[i] = sum_j L_of_R_mat[i,j] (total probability of read i)
            L_of_R = np.array(L_of_R_mat.sum(1)).flatten()
            # Filter out reads with very low probability
            valid_reads = L_of_R >= 1e-200
            if np.any(valid_reads):
                L_of_R_mat = L_of_R_mat[valid_reads, :]
                L_of_R = L_of_R[valid_reads]
            else:
                # Skip this file if no valid reads
                continue
            if len(L_of_R) > 0:
                # Normalize to get posterior probabilities
                L_of_R_inv = sparse.csr_matrix(1.0 / L_of_R).transpose()
                # Expected counts: sum over reads of P(transcript|read)
                expected_counts = np.array(L_of_R_mat.multiply(L_of_R_inv).sum(0)).flatten()
                total_expected_counts += expected_counts
                total_reads += len(L_of_R)
    # Convert expected counts to integer counts for multinomial
    # Scale up to preserve precision
    scale_factor = max(1000, int(total_reads * 0.1))  # Use at least 1000 pseudo-counts
    scaled_counts = np.round(total_expected_counts * scale_factor).astype(int)
    # Ensure at least some counts
    if np.sum(scaled_counts) == 0:
        print("Warning: No expected counts found, using uniform pseudo-counts")
        scaled_counts = np.ones(L, dtype=int)
    total_pseudo_reads = np.sum(scaled_counts)
    print(f"Total reads processed: {total_reads}")
    print(f"Total expected counts: {np.sum(total_expected_counts):.2f}")
    print(f"Scaled pseudo-counts: {total_pseudo_reads}")
    print(f"Non-zero transcripts: {np.sum(scaled_counts > 0)}")
    # Convert to JAX arrays
    counts = jnp.array(scaled_counts)
    # Define multinomial model
    if prior_type == "dirichlet":
        def model():
            # Dirichlet prior with sparsity bias
            alpha_prior = jnp.ones(L) * sparsity_strength
            x = numpyro.sample("x", dist.Dirichlet(alpha_prior))
            numpyro.sample("obs", dist.Multinomial(total_count=total_pseudo_reads, probs=x), obs=counts)
        def guide():
            # Variational Dirichlet posterior
            alpha_q = numpyro.param("alpha_q", 
                                   jnp.array(scaled_counts + 1.0),  # Initialize with data + prior
                                   constraint=dist.constraints.positive)
            numpyro.sample("x", dist.Dirichlet(alpha_q))
    elif prior_type == "logistic_normal":
        def model():
            z = numpyro.sample("z", dist.Normal(jnp.zeros(L), jnp.ones(L)))
            x = numpyro.deterministic("x", softmax(z))
            numpyro.sample("obs", dist.Multinomial(total_count=total_pseudo_reads, probs=x), obs=counts)
        guide = AutoNormal(model)
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")
    # Set up optimizer and SVI
    optimizer = ClippedAdam(step_size=schedule_fn, clip_norm=1.0)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    # Initialize
    rng_key = jax.random.PRNGKey(0)
    svi_state = svi.init(rng_key)
    # Training loop
    print("Starting multinomial VB training...")
    for step in range(n_steps):
        svi_state, loss = svi.update(svi_state)
        if step % 10 == 0:
            print(f"[VB:{prior_type}] Step {step} | Loss: {loss:.4f}")
    # Extract results
    params = svi.get_params(svi_state)
    if prior_type == "dirichlet":
        # For Dirichlet guide, the mean is alpha_q / sum(alpha_q)
        alpha_q = params["alpha_q"]
        x_est = np.array(alpha_q / jnp.sum(alpha_q))
        print(f"Learned Dirichlet concentrations - min: {jnp.min(alpha_q):.4f}, max: {jnp.max(alpha_q):.4f}")
    elif prior_type == "logistic_normal":
        from numpyro.infer import Predictive
        predictive = Predictive(guide, params=params, num_samples=1000)
        posterior_samples = predictive(jax.random.PRNGKey(1))
        z_samples = posterior_samples["z"]
        z_mean = jnp.mean(z_samples, axis=0)
        x_est = np.array(softmax(z_mean))
    # Final normalization
    x_est = np.maximum(x_est, 1e-12)
    x_est = x_est / np.sum(x_est)
    print(f"VB Results: {np.sum(x_est > 1e-10)} non-zero transcripts")
    print(f"Max abundance: {np.max(x_est):.6f}")
    print(f"Top 5 abundances: {np.sort(x_est)[-5:]}")
    # Save results
    nonzero = x_est > 1e-10
    with open(prefix + "X_final.pkl", "wb") as f:
        pickle.dump(x_est[nonzero], f)
    with open(prefix + "names_final.pkl", "wb") as f:
        pickle.dump(np.array(TE_names)[nonzero], f)

def run_variational_bayes_continuous(G_of_R_list, TE_list, prefix,
                                     lr=1e-3, n_steps=500,
                                     prior_type="dirichlet", schedule="constant", 
                                     sparsity_strength=0.1):
    """
    VB that more closely matches EM by using continuous expected counts
    instead of discrete multinomial counts
    """
    # Same setup as before
    if schedule == "constant":
        schedule_fn = lr
    elif schedule == "warmup":
        schedule_fn = lambda step: improved_schedule(step, n_steps, base_lr=1e-4)
    elif schedule == "plateau":
        schedule_fn = lambda step: plateau_schedule(step, n_steps, base_lr=1e-3)
    else:
        raise ValueError(f"Unknown scheduler: {schedule}")
    TE_names = [line.strip().split('\t')[0] for line in open(TE_list)]
    L = len(TE_names)
    print("Computing expected read counts for continuous VB...")
    # Use the SAME initialization as EM
    X = np.ones(L, dtype=np.float64) / L
    # Compute expected counts exactly like EM does
    total_expected_counts = np.zeros(L, dtype=np.float64)
    total_reads = 0
    for pkl_file in open(G_of_R_list):
        pkl_file = pkl_file.strip()
        with open(pkl_file, 'rb') as f:
            G_of_R = pickle.load(f)
        if G_of_R is not None:
            if not isinstance(G_of_R, sparse.csr_matrix):
                G_of_R = G_of_R.tocsr()
            # Exact same calculation as EM
            L_of_R_mat = G_of_R.multiply(X)
            L_of_R_mat = L_of_R_mat.tocsr()
            L_of_R = np.array(L_of_R_mat.sum(1)).flatten()
            valid_reads = L_of_R >= 1e-200
            if np.any(valid_reads):
                L_of_R_mat = L_of_R_mat[valid_reads, :]
                L_of_R = L_of_R[valid_reads]
            else:
                continue
            if len(L_of_R) > 0:
                L_of_R_inv = sparse.csr_matrix(1.0 / L_of_R).transpose()
                expected_counts = np.array(L_of_R_mat.multiply(L_of_R_inv).sum(0)).flatten()
                total_expected_counts += expected_counts
                total_reads += len(L_of_R)
    # NO SCALING - use continuous expected counts directly
    expected_counts = jnp.array(total_expected_counts)
    total_expected = jnp.sum(expected_counts)
    print(f"Total reads processed: {total_reads}")
    print(f"Total expected counts: {total_expected:.2f}")
    print(f"Non-zero transcripts: {jnp.sum(expected_counts > 0)}")
    # Define continuous model that matches EM objective
    if prior_type == "dirichlet":
        def model():
            # Dirichlet prior
            alpha_prior = jnp.ones(L) * sparsity_strength
            x = numpyro.sample("x", dist.Dirichlet(alpha_prior))
            # Use Normal likelihood with the expected counts
            # This approximates the continuous optimization that EM does
            # Variance scaled by abundance to match EM behavior
            sigma = jnp.sqrt(x * total_expected + 1e-8)  # Variance ~ abundance
            numpyro.sample("obs", dist.Normal(x * total_expected, sigma), obs=expected_counts)
        def guide():
            alpha_q = numpyro.param("alpha_q", 
                                   expected_counts + sparsity_strength,  # Initialize with data
                                   constraint=dist.constraints.positive)
            numpyro.sample("x", dist.Dirichlet(alpha_q))
    elif prior_type == "logistic_normal":
        def model():
            z = numpyro.sample("z", dist.Normal(jnp.zeros(L), jnp.ones(L)))
            x = numpyro.deterministic("x", softmax(z))
            # Same continuous likelihood
            sigma = jnp.sqrt(x * total_expected + 1e-8)
            numpyro.sample("obs", dist.Normal(x * total_expected, sigma), obs=expected_counts)
        guide = AutoNormal(model)
    # Rest is the same...
    optimizer = ClippedAdam(step_size=schedule_fn, clip_norm=1.0)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    rng_key = jax.random.PRNGKey(0)
    svi_state = svi.init(rng_key)
    print("Starting continuous VB training...")
    for step in range(n_steps):
        svi_state, loss = svi.update(svi_state)
        if step % 2 == 0:
            print(f"[VB-Continuous:{prior_type}] Step {step} | Loss: {loss:.4f}")
    # Extract results
    params = svi.get_params(svi_state)
    if prior_type == "dirichlet":
        alpha_q = params["alpha_q"]
        x_est = np.array(alpha_q / jnp.sum(alpha_q))
    elif prior_type == "logistic_normal":
        from numpyro.infer import Predictive
        predictive = Predictive(guide, params=params, num_samples=1000)
        posterior_samples = predictive(jax.random.PRNGKey(1))
        z_samples = posterior_samples["z"]
        z_mean = jnp.mean(z_samples, axis=0)
        x_est = np.array(softmax(z_mean))
    # Final normalization (same as EM)
    x_est = np.maximum(x_est, 1e-12)
    x_est = x_est / np.sum(x_est)
    print(f"VB-Continuous Results: {np.sum(x_est > 1e-10)} non-zero transcripts")
    print(f"Max abundance: {np.max(x_est):.6f}")
    # Save results
    nonzero = x_est > 1e-10
    with open(prefix + "X_final.pkl", "wb") as f:
        pickle.dump(x_est[nonzero], f)
    with open(prefix + "names_final.pkl", "wb") as f:
        pickle.dump(np.array(TE_names)[nonzero], f)

# Parse commandline arguments

def GetArgs():
    class Parser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = Parser()
    parser.add_argument('-g', '--G_of_R_list', required=True, type=str)
    parser.add_argument('-l', '--TE_list', required=True, type=str)
    parser.add_argument('-s', '--stop_thresh', default=1e-7, type=float)
    parser.add_argument('-r', '--report_every', default=100, type=int)
    parser.add_argument('-m', '--max_nEMsteps', default=10000, type=int)
    parser.add_argument('-t', '--nThreads', default=16, type=int)
    parser.add_argument('-p', '--prefix', default='', type=str)
    parser.add_argument('--method', choices=['em','vb'], default='em')
    parser.add_argument('--vb_lr', default=0.01, type=float)
    parser.add_argument('--vb_steps', default=10, type=int)
    parser.add_argument('--prior_type', type=str, default="dirichlet", choices=["dirichlet", "logistic_normal"])
    parser.add_argument('--schedule', type=str, default="constant", choices=['constant', 'warmup', 'plateau'])
    parser.add_argument('--vb_likelihood', type=str, default="multinomial", choices=['multinomial', 'continuous'])

    return parser.parse_args()

def main():
    args = GetArgs()

    if args.method == 'vb':
        if args.vb_likelihood == 'multinomial':
            run_variational_bayes(args.G_of_R_list, args.TE_list, args.prefix, args.vb_lr, args.vb_steps, args.prior_type, args.schedule)
        else:
            run_variational_bayes_continuous(args.G_of_R_list, args.TE_list, args.prefix, args.vb_lr, args.vb_steps, args.prior_type, args.schedule)
        return

    TE_names = [line.strip().split('\t')[0] for line in open(args.TE_list)]
    X = sparse.csr_matrix(np.ones((1, len(TE_names)), dtype=np.float64) / len(TE_names))

    G_of_R_pkl_fulllist = [line.strip() for line in open(args.G_of_R_list)]
    G_of_R_pkl_lists = []
    listsize = len(G_of_R_pkl_fulllist) // args.nThreads
    nlistsp1 = len(G_of_R_pkl_fulllist) % args.nThreads
    k = 0
    for i in range(nlistsp1):
        G_of_R_pkl_lists.append(G_of_R_pkl_fulllist[k:k + listsize + 1])
        k += listsize + 1
    for i in range(nlistsp1, args.nThreads):
        G_of_R_pkl_lists.append(G_of_R_pkl_fulllist[k:k + listsize])
        k += listsize

    masterPool = Pool(processes=args.nThreads)

    for step in range(args.max_nEMsteps):
        starttime = datetime.datetime.now()
        exp_counts = np.zeros((1, len(TE_names)), dtype=np.float64)
        loglik = 0.0
        outputs = masterPool.map(calculate_expcounts_chunk, zip(G_of_R_pkl_lists, [X] * args.nThreads))
        for output in outputs:
            this_exp_counts, this_loglik = output
            exp_counts += this_exp_counts
            loglik += this_loglik

        last_X = X.copy()
        X = sparse.csr_matrix(exp_counts / np.sum(exp_counts))
        print(f"{step} {np.max(np.abs(X.toarray() - last_X.toarray()))} {loglik} {datetime.datetime.now() - starttime}")

        if (step + 1) % args.report_every == 0:
            pickle.dump(X.toarray()[X.toarray() > 1e-10], open(args.prefix + 'X_step_' + str(step + 1) + '.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(np.array(TE_names)[X.toarray()[0, :] > 1e-10], open(args.prefix + 'names_step_' + str(step + 1) + '.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        if np.max(np.abs(X.toarray() - last_X.toarray())) < args.stop_thresh:
            break

    pickle.dump(X.toarray()[X.toarray() > 1e-10], open(args.prefix + 'X_final.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(np.array(TE_names)[X.toarray()[0, :] > 1e-10], open(args.prefix + 'names_final.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
