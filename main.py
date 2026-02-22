# main.py - Main execution script for analyzing the effect of label corruption on neural network loss landscapes.


import time
import torch
import logging
from model import MLP
from plotter import Plotter
from torch.utils.data import DataLoader
from landscape import make_direction, get_base_params, compute_1d_curve
from utils import setup_logger, load_mnist, apply_corruption, train_model, eval_loss, save_csv, log_json


setup_logger()
log = __import__("logging").getLogger("main")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")


def save_plot_values_csv(p_values, all_alphas, all_curves):
    """
    Save plot values (alphas and losses) for each p value in separate CSV files.
    
    This function iterates through provided p values and their corresponding alpha and loss curve data,
    writing each dataset to a separate CSV file in the results/plots/1d_csvs/ directory.
    Each CSV file contains two columns: 'alpha' and 'loss'.
    
    Args:
        p_values (list): List of p parameter values used as identifiers for output files.
        all_alphas (list): List of alpha arrays, one for each p value.
        all_curves (list): List of loss curve arrays, one for each p value.
    
    Returns:
        None
    
    Side Effects:
        - Creates CSV files at paths: results/plots/1d_csvs/plot_data_p{p}.csv
        - Logs info messages for each saved CSV file
    
    Raises:
        IOError: If the output directory does not exist or file cannot be written.
    """
    import csv
    for p, alphas, curve in zip(p_values, all_alphas, all_curves):
        path = f"results/plots/1d_csvs/plot_data_p{p}.csv"
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['alpha', 'loss'])
            for alpha, loss in zip(alphas, curve):
                writer.writerow([float(alpha), float(loss)])
        log.info(f"Saved plot values CSV for p={p}: {path}")

def run():
    """
    Execute the main analysis pipeline for studying the effect of label corruption on neural network loss landscapes.
    This function:
    1. Loads the MNIST dataset and splits it into train, test, and evaluation sets
    2. Iterates over corruption probability values (p = 0.0 to 1.0)
    3. For each corruption level:
       - Applies label corruption to the training set
       - Trains an MLP model for 10 epochs
       - Evaluates the model on clean evaluation data
       - Computes a 1D loss landscape curve along a random direction
       - Calculates basin width metrics (r_max and area)
       - Logs results and saves training curves
    4. Generates summary CSV with results across all corruption levels
    5. Creates visualization plots:
       - Overlaid parabola fits of loss landscapes
       - Parabolas with alpha_0 markers
       - Individual parabolas with consistent axes
       - Loss vs corruption probability plots
       - Basin width vs corruption probability plots
    6. Logs total execution time
    Returns:
        None
    Side effects:
        - Trains and evaluates multiple neural network models
        - Generates CSV files with results in results/ directory
        - Creates visualization plots
        - Writes detailed logs to configured logging output
    """

    log.info("Starting run")
    start_ts = time.time()
    train_set, test_set, eval_clean, corrupt_idx = load_mnist()
    try:
        train_len = len(train_set)
        test_len = len(test_set)
    except Exception:
        train_len = getattr(train_set, "data", None)
        test_len = getattr(test_set, "data", None)
    log.info(f"Loaded MNIST datasets: train_len={train_len}, test_len={test_len}")

    test_loader=DataLoader(test_set,batch_size=256,shuffle=False)
    original_targets=train_set.targets.clone()

    pvals=[0.0,0.2,0.4,0.6,0.8,1.0]
    clean_losses=[]; test_accs=[]; areas=[]; rows=[]
    all_alphas = []
    all_curves = []

    plotter = Plotter(csv_pattern="results/1d_csvs/plot_data_p*.csv", log_level=logging.INFO)
    for p in pvals:
        p_start = time.time()                      # <-- per-p timer start
        log.info(f"===== p={p} =====")
        apply_corruption(train_set, original_targets, corrupt_idx, p)
        # count how many labels differ from the original
        try:
            corrupted_count = int((train_set.targets != original_targets).sum().item())
        except Exception:
            corrupted_count = None
        log.debug(f"Applied corruption for p={p}; corrupted_count={corrupted_count}")
        train_loader=DataLoader(train_set,batch_size=128,shuffle=True)
        model=MLP().to(device)
        log.info(f"Initialized model for p={p}")

        losses,accs=train_model(model,train_loader,test_loader,epochs=10,lr=1e-3,device=device)
        log.info(f"Finished training p={p}: final_train_loss={float(losses[-1]) if len(losses)>0 else 'n/a'} final_test_acc={float(accs[-1]) if len(accs)>0 else 'n/a'}")
        clean_loss=eval_loss(model,eval_clean,device); final_acc=accs[-1]

        clean_losses.append(clean_loss); test_accs.append(final_acc)

        d=make_direction(model); base=get_base_params(model)
        alphas,curve,alpha_star=compute_1d_curve(model,base,d,lambda m,_:eval_loss(m,eval_clean,device),eval_clean,scale=0.5)
        log.info(f"Computed 1D curve for p={p}; alpha_star={float(alpha_star)}")

        Lmin=curve.min(); span=curve.max()-Lmin; thresh=Lmin+0.2*span
        mask=curve<=thresh
        r_max=float(max(abs(alphas[mask]))) if mask.any() else 0.0
        area=r_max*r_max; areas.append(area)

        log.info(f"p={p} summary: clean_loss={clean_loss:.6f} final_acc={float(final_acc):.4f} r_max={r_max:.6f} area={area:.6f}")

        log_json("result",p=p,clean_loss=clean_loss,test_acc=float(final_acc),r_max=r_max,area=area)
        rows.append([p,clean_loss,float(final_acc),r_max,area])
        # Plot Training Curves
        plotter.save_train_curves(p, losses, accs)
        all_alphas.append(alphas)
        all_curves.append(curve)

        # per-p elapsed and log
        elapsed_p = time.time() - p_start
        hrs_p, rem_p = divmod(int(elapsed_p), 3600)
        mins_p, secs_p = divmod(rem_p, 60)
        human_p = f"{hrs_p:d}h{mins_p:02d}m{secs_p:02d}s"
        log.info(f"p={p} completed (elapsed={human_p}, {elapsed_p:.2f}s)")
        log_json("p_complete", p=p, duration_sec=elapsed_p, duration_human=human_p)

    csv_path = save_csv(rows)
    log.info(f"Saved CSV results: {csv_path if csv_path is not None else 'results/summary.csv (default)'}")
    save_plot_values_csv(pvals, all_alphas, all_curves)

    # Plot overlaid parabolas
    plotter.plot_aligned_parabolas_overlay()
    # Plot with alpha0 marker
    plotter.plot_parabolas_with_alpha0_line()
    # Plot individual parabolas with same axes
    plotter.plot_individual_parabolas_same_axes()
    # Plot L vs p
    plotter.plot_L_vs_p()
    # Plot basin widths
    plotter.plot_width_vs_p()
    
    log.info("All plots generated successfully")
    elapsed = time.time() - start_ts
    # human-readable H:M:S and seconds
    hrs, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)
    human = f"{hrs:d}h{mins:02d}m{secs:02d}s"
    log.info(f"Run complete (elapsed={human}, {elapsed:.2f}s)")
    log_json("run_complete", duration_sec=elapsed, duration_human=human)

if __name__=="__main__": run()
