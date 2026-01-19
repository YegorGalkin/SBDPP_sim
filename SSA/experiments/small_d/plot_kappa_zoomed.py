"""
Create zoomed-in kappa extraction plot for reduced dispersal case.

This script loads pre-calculated data from kappa_results.npz and creates
a kappa extraction plot zoomed into the interval d ∈ [0, 0.01].
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_kappa_zoomed(data_file: str, zoom_range: tuple = (0, 0.01), 
                      output_file: str = "kappa_extraction_zoomed.png"):
    """
    Create zoomed kappa extraction plot.
    
    Args:
        data_file: Path to kappa_results.npz
        zoom_range: (d_min, d_max) for zooming
        output_file: Output filename for the plot
    """
    # Load data
    data = np.load(data_file)
    d_values = data['d_values']
    density_means = data['density_means']
    density_stds = data['density_stds']
    n_expected = data['n_expected']
    d_prime = float(data['d_prime'])
    sigma = float(data['sigma'])
    n_batches = int(data['n_batches'])
    
    print(f"Loaded data from: {data_file}")
    print(f"  Total d values: {len(d_values)}")
    print(f"  d range: [{d_values.min():.6f}, {d_values.max():.6f}]")
    print(f"  σ = {sigma}, d' = {d_prime}")
    print(f"  n_batches = {n_batches}")
    
    # Filter for zoom range (exclude d=0)
    d_min, d_max = zoom_range
    zoom_mask = (d_values > d_min) & (d_values <= d_max)
    
    d_zoom = d_values[zoom_mask]
    n_exp_zoom = n_expected[zoom_mask]
    n_meas_zoom = density_means[zoom_mask]
    n_err_zoom = density_stds[zoom_mask]
    
    print(f"\nZoom range: d ∈ [{d_min}, {d_max}]")
    print(f"  Points in zoom range: {len(d_zoom)}")
    
    if len(d_zoom) < 2:
        print("ERROR: Not enough points in zoom range!")
        return
    
    # Compute y = (n_expected - n_measured) × d' and errors
    y_zoom = (n_exp_zoom - n_meas_zoom) * d_prime
    y_err_zoom = n_err_zoom * d_prime
    
    # Weighted linear fit through origin: y = kappa * x
    # Weights: w = 1/σ²
    weights = 1.0 / (y_err_zoom**2 + 1e-12)
    kappa_est = np.sum(weights * d_zoom * y_zoom) / np.sum(weights * d_zoom**2)
    
    # Standard error of kappa
    kappa_var = 1.0 / np.sum(weights * d_zoom**2)
    kappa_err = np.sqrt(kappa_var)
    
    z_95 = 1.96  # 95% confidence interval
    kappa_ci_95 = z_95 * kappa_err
    
    print(f"\nKappa estimation:")
    print(f"  κ = {kappa_est:.4f} ± {kappa_ci_95:.4f} (95% CI)")
    print(f"  Relative uncertainty: {kappa_ci_95/kappa_est*100:.1f}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points with error bars
    ax.errorbar(d_zoom, y_zoom, yerr=z_95*y_err_zoom, 
               fmt='o', color='C0', markersize=8, 
               capsize=5, capthick=2, elinewidth=2,
               label=f'Data (95% CI, N={len(d_zoom)})')
    
    # Fit line with confidence band
    d_fit = np.linspace(0, d_max * 1.05, 200)
    y_fit = kappa_est * d_fit
    y_fit_upper = (kappa_est + z_95*kappa_err) * d_fit
    y_fit_lower = (kappa_est - z_95*kappa_err) * d_fit
    
    ax.plot(d_fit, y_fit, 'r-', linewidth=2.5, 
           label=f'κ = {kappa_est:.4f} ± {kappa_ci_95:.4f}')
    ax.fill_between(d_fit, y_fit_lower, y_fit_upper, 
                   color='red', alpha=0.2, label='95% CI')
    
    # Formatting
    ax.set_xlabel('d (intrinsic death rate)', fontsize=12, fontweight='bold')
    ax.set_ylabel('(n$_{\\mathrm{expected}}$ - n$_{\\mathrm{measured}}$) × d\'', 
                 fontsize=12, fontweight='bold')
    ax.set_title(f'Kappa Extraction (Zoomed): σ = {sigma}, d ∈ [{d_min}, {d_max}]', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_xlim(d_min, d_max * 1.02)
    
    # Add text box with fit info
    textstr = f'Fit: y = κ·d\nκ = {kappa_est:.5f} ± {kappa_ci_95:.5f}\nσ = {sigma}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(data_file).parent / output_file
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved zoomed plot to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create zoomed kappa extraction plot")
    parser.add_argument('--data', '-d', type=str, 
                       default='SSA/experiments/small_d/results/reduced_dispersal/kappa_results.npz',
                       help='Path to kappa_results.npz file')
    parser.add_argument('--range', '-r', type=float, nargs=2, default=[0, 0.01],
                       help='Zoom range [d_min, d_max] (default: 0 0.01)')
    parser.add_argument('--output', '-o', type=str, default='kappa_extraction_zoomed.png',
                       help='Output filename (default: kappa_extraction_zoomed.png)')
    
    args = parser.parse_args()
    
    plot_kappa_zoomed(args.data, zoom_range=tuple(args.range), output_file=args.output)
