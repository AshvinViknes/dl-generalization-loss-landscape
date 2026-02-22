# plotter.py - Class-based plotting utilities for parabolic basin analysis.

import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger("plotter")


class Plotter:
    """Handles loading and plotting of parabolic basin data from CSV files."""
    
    def __init__(self, csv_pattern="plot_data_p*.csv", alpha0=None, alpha0_ratio=0.7, log_level=logging.INFO, output_dir="results/plots"):
        """Initialize plotter by loading CSV files and fitting parabolas.
        
        Args:
            csv_pattern: Glob pattern for CSV files (tries /content/ first, then local)
            alpha0: Fixed alpha0 value. If None, computed as alpha0_ratio * R_common
            alpha0_ratio: Ratio for computing alpha0 as fraction of R_common (default 0.7)
            log_level: Logging level (default logging.INFO)
            output_dir: Directory to save plots (default "results/plots")
        """
        logger.setLevel(log_level)
        self.csv_pattern = csv_pattern
        self.alpha0_ratio = alpha0_ratio
        self.output_dir = output_dir
        self.fits = []
        self.ranges = []
        self.R_common = None
        self.alpha0 = alpha0
        self.x_centered = None
        self.global_ylim = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
        logger.info(f"Initializing Plotter with csv_pattern='{csv_pattern}'")
        self._load_and_fit_parabolas()
        self._compute_common_parameters()
    
    def _get_output_path(self, filename):
        """Get full output path for a file."""
        return os.path.join(self.output_dir, filename)
    
    def _load_and_fit_parabolas(self):
        """Load CSV files and fit quadratic parabolas to the data."""
        logger.debug(f"Loading CSV files with pattern: {self.csv_pattern}")
        files = sorted(glob.glob(f"/content/{self.csv_pattern}"))
        if not files:
            files = sorted(glob.glob(self.csv_pattern))
        if not files:
            logger.error(f"No files found matching pattern: {self.csv_pattern}")
            raise FileNotFoundError(f"No {self.csv_pattern} found.")
        
        for f in files:
            logger.debug(f"Processing file: {f}")
            df = pd.read_csv(f)
            if not {"alpha", "loss"}.issubset(df.columns):
                logger.error(f"{f} missing required columns: alpha, loss")
                raise ValueError(f"{f} must have columns: alpha, loss")
            
            alpha = df["alpha"].to_numpy(float)
            loss = df["loss"].to_numpy(float)
            
            # Fit quadratic: L(a) = A a^2 + B a + C
            A, B, C = np.polyfit(alpha, loss, 2)
            
            # Vertex (minimum)
            alpha_star = -B / (2 * A)
            loss_star = A * alpha_star**2 + B * alpha_star + C
            
            # Centered range available for this dataset
            alpha_centered = alpha - alpha_star
            self.ranges.append(np.max(np.abs(alpha_centered)))
            
            p_str = f.split("plot_data_p")[-1].replace(".csv", "")
            p_val = float(p_str)
            
            logger.debug(f"Fitted parabola for p={p_val}: A={A:.6e}, alpha_star={alpha_star:.6f}, loss_star={loss_star:.6e}")
            
            self.fits.append({
                "p": p_val,
                "p_str": p_str,
                "A": A, "B": B, "C": C,
                "alpha_star": alpha_star,
                "loss_star": loss_star
            })
        
    
    def _compute_common_parameters(self):
        """Compute common x-window, alpha0, and global y-limits."""
        self.R_common = float(min(self.ranges))
        
        if self.alpha0 is None:
            self.alpha0 = self.alpha0_ratio * self.R_common
        
        self.x_centered = np.linspace(-self.R_common, self.R_common, 800)
        
        # Compute global y-limits
        global_ymin = +np.inf
        global_ymax = -np.inf
        
        for d in self.fits:
            A, B, C = d["A"], d["B"], d["C"]
            alpha_star, loss_star = d["alpha_star"], d["loss_star"]
            
            x = self.x_centered + alpha_star
            y = A*x**2 + B*x + C
            y_centered = y - loss_star
            
            global_ymin = min(global_ymin, float(y_centered.min()))
            global_ymax = max(global_ymax, float(y_centered.max()))
        
        ypad = 0.05 * (global_ymax - global_ymin + 1e-12)
        self.global_ylim = (global_ymin - ypad, global_ymax + ypad)
        
        logger.info(f"R_common = {self.R_common:.6f}")
        logger.info(f"Using alpha0 = {self.alpha0:.6f}")
        logger.info(f"Common x window: [{-self.R_common:.4f}, {self.R_common:.4f}]")
        logger.info(f"Common y window: [{self.global_ylim[0]:.4e}, {self.global_ylim[1]:.4e}]")
    
    def plot_aligned_parabolas_overlay(self, figsize=(10, 6), save_path=None):
        """Plot all parabolas overlaid on the same window."""
        if save_path is None:
            save_path = self._get_output_path("parabola_basin_comparison_visible.png")
        logger.info("Generating overlaid parabolas plot")
        plt.figure(figsize=figsize)
        
        for d in sorted(self.fits, key=lambda z: z["p"]):
            A, B, C = d["A"], d["B"], d["C"]
            alpha_star, loss_star = d["alpha_star"], d["loss_star"]
            
            x = self.x_centered + alpha_star
            y = A*x**2 + B*x + C
            y_centered = y - loss_star
            
            plt.plot(self.x_centered, y_centered, linewidth=3, label=f"p={d['p_str']}")
        
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.title(f"Aligned Parabolic Basins (common window: [-{self.R_common:.2f}, {self.R_common:.2f}])")
        plt.xlabel("α (shifted so minimum is at 0)")
        plt.ylabel("Loss (shifted so minimum is 0)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
        else:
            logger.debug("Plot not saved (save_path=None)")
    
    def plot_parabolas_with_alpha0_line(self, figsize=(10, 6), save_path=None):
        """Plot centered parabolas with vertical line at alpha0."""
        if save_path is None:
            save_path = self._get_output_path("parabola_with_alpha0_line.png")
        logger.info(f"Generating parabolas plot with alpha0={self.alpha0:.6f} line")
        plt.figure(figsize=figsize)
        
        for d in sorted(self.fits, key=lambda z: z["p"]):
            A, B, C = d["A"], d["B"], d["C"]
            alpha_star, loss_star = d["alpha_star"], d["loss_star"]
            
            x = self.x_centered + alpha_star
            y = A*x**2 + B*x + C
            y_centered = y - loss_star
            
            plt.plot(self.x_centered, y_centered, linewidth=2.5, label=f"p={d['p_str']}")
        
        plt.axvline(self.alpha0, linestyle="--", linewidth=2)
        plt.title(f"Aligned Parabolic Basins + α0 line (α0={self.alpha0:.4f})")
        plt.xlabel("α (shifted so minimum is at 0)")
        plt.ylabel("Loss increase from minimum")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
        else:
            logger.debug("Plot not saved (save_path=None)")
    
    def plot_individual_parabolas_same_axes(self, save_dir=None):
        """Plot each parabola separately with identical axes."""
        if save_dir is None:
            save_dir = self.output_dir
        logger.info("Generating individual parabola plots with same axes")
        for d in sorted(self.fits, key=lambda z: z["p"]):
            A, B, C = d["A"], d["B"], d["C"]
            alpha_star, loss_star = d["alpha_star"], d["loss_star"]
            
            x = self.x_centered + alpha_star
            y = A*x**2 + B*x + C
            y_centered = y - loss_star
            
            plt.figure(figsize=(7, 5))
            plt.plot(self.x_centered, y_centered, linewidth=3)
            
            plt.title(f"Aligned Parabolic Basin (p={d['p_str']})")
            plt.xlabel("α (shifted so minimum is at 0)")
            plt.ylabel("Loss (shifted so minimum is 0)")
            
            plt.xlim(-self.R_common, self.R_common)
            plt.ylim(*self.global_ylim)
            
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            
            out = os.path.join(save_dir, f"loss_line_parabola_p{d['p_str']}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {out}")
    
    def compute_intersection_heights(self):
        """Compute intersection heights L(p) at alpha0.
        
        Returns:
            tuple: (p_vals, L_vals) arrays of p values and corresponding losses
        """
        Lp = []
        for d in self.fits:
            A, B, C = d["A"], d["B"], d["C"]
            alpha_star, loss_star = d["alpha_star"], d["loss_star"]
            
            a = self.alpha0 + alpha_star
            y = A*a**2 + B*a + C
            L = y - loss_star
            
            Lp.append((d["p"], float(L)))
        
        Lp.sort(key=lambda t: t[0])
        p_vals = np.array([t[0] for t in Lp], dtype=float)
        L_vals = np.array([t[1] for t in Lp], dtype=float)
        
        logger.info("Intersection heights L(p) at alpha0:")
        for p, L in Lp:
            logger.info(f"p={p:.1f} -> L={L:.6e}")
        
        return p_vals, L_vals
    
    def plot_L_vs_p(self, save_path=None):
        """Plot intersection heights L vs p."""
        if save_path is None:
            save_path = self._get_output_path("L_vs_p.png")
        logger.info("Generating L vs p plot")
        p_vals, L_vals = self.compute_intersection_heights()
        
        plt.figure(figsize=(8, 5))
        plt.plot(p_vals, L_vals, marker="o", linewidth=2.5)
        plt.title(f"L(p) at fixed α0={self.alpha0:.4f} (intersection height)")
        plt.xlabel("p (label corruption)")
        plt.ylabel("L = loss increase at α0")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
        else:
            logger.debug("Plot not saved (save_path=None)")
        
        return p_vals, L_vals
    
    def compute_basin_widths(self, loss_level=1e-7):
        """Compute basin widths at a fixed loss level.
        
        Args:
            loss_level: Fixed loss threshold (default 1e-7)
        
        Returns:
            tuple: (p_vals, widths) arrays of p values and basin widths
        """
        p_vals = []
        widths = []
        
        for d in self.fits:
            A = d["A"]
            p = d["p"]
            
            r = np.sqrt(loss_level / A)
            W = 2 * r
            
            p_vals.append(p)
            widths.append(W)
        
        p_vals = np.array(p_vals)
        widths = np.array(widths)
        idx = np.argsort(p_vals)
        
        p_vals = p_vals[idx]
        widths = widths[idx]
        
        return p_vals, widths
    
    def plot_width_vs_p(self, loss_level=1e-7, save_path=None):
        """Plot basin width vs p at fixed loss level."""
        if save_path is None:
            save_path = self._get_output_path(f"width_vs_p_L{loss_level:.0e}.png")
        logger.info(f"Generating width vs p plot (loss_level={loss_level:.0e})")
        p_vals, widths = self.compute_basin_widths(loss_level)
        
        plt.figure(figsize=(7, 5))
        plt.plot(p_vals, widths, marker="o", linewidth=2.5)
        plt.xlabel("p (label corruption)")
        plt.ylabel(rf"Width at $L={loss_level:.0e}$")
        plt.title("Basin width at fixed loss level")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved plot to {save_path}")
        else:
            logger.debug("Plot not saved (save_path=None)")
        
        return p_vals, widths
    
    def plot_test_accuracy_vs_p(self, p_vals, test_acc_vals, save_path=None):
        """Plot test accuracy vs p.
        
        Args:
            p_vals: Array of p values
            test_acc_vals: Array of test accuracy values
            save_path: Output file path (default: results/plots/test_acc_vs_p.png)
        """
        if save_path is None:
            save_path = self._get_output_path("test_acc_vs_p.png")
        logger.info("Generating test accuracy vs p plot")
        plt.figure()
        plt.plot(p_vals, test_acc_vals, marker="o")
        plt.xlabel("p (label corruption)")
        plt.ylabel("Test accuracy (%)")
        plt.title("Test Accuracy vs Label Corruption")
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved plot to {save_path}")
        else:
            logger.debug("Plot not saved (save_path=None)")
    
    def plot_basin_width_vs_accuracy(self, basin_widths, test_acc_vals, p_vals=None, save_path=None):
        """Plot basin width vs test accuracy.
        
        Args:
            basin_widths: Array of basin widths
            test_acc_vals: Array of test accuracy values
            p_vals: Optional array of p values for labels
            save_path: Output file path (default: results/plots/width_vs_acc.png)
        """
        if save_path is None:
            save_path = self._get_output_path("width_vs_acc.png")
        logger.info("Generating basin width vs test accuracy plot")
        plt.figure()
        plt.scatter(test_acc_vals, basin_widths)
        plt.xlabel("Test accuracy (%)")
        plt.ylabel(r"Basin width at $L = 10^{-7}$")
        plt.title("Basin Width vs Test Accuracy")
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved plot to {save_path}")
        else:
            logger.debug("Plot not saved (save_path=None)")

    def save_train_curves(self, p, train_losses, test_accs):
        """
        Generate and save training curves for a given parameter p.
        
        Creates a figure with two subplots: one for training loss and one for test accuracy.
        The figure is saved as a PNG file in the results/plots directory.
        
        Args:
            p: Parameter value used in the experiment (included in plot titles and filename).
            train_losses: List or array of training loss values across epochs.
            test_accs: List or array of test accuracy values (as decimals) across epochs.
        """
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(train_losses, "o-")
        plt.title(f"Train Loss (p={p})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.subplot(1,2,2)
        plt.plot([a*100 for a in test_accs], "o-")
        plt.title(f"Test Acc (p={p})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.tight_layout()
        path = f"results/plots/train_curves_p{p}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved train curves for p={p}: {path}")

  