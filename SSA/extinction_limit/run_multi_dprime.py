"""
Run extinction scaling experiments for all d' values (1.0, 0.1, 0.01) in parallel.
Uses the refactored extinction_scaling.py API.
"""
from pathlib import Path
import sys

# Import from the refactored extinction_scaling module
from extinction_scaling import (
    create_parameter_grid,
    run_all_simulations_parallel,
    load_results_csv,
    plot_summary,
    generate_diagnostic_plots,
    print_results_summary,
    DPRIME_VALUES
)

def main():
    """Run all d' values in parallel with diagnostic data saving and plotting."""
    import argparse
    parser = argparse.ArgumentParser(description="Run multi-d' extinction scaling experiments")
    parser.add_argument('--target-pop', '-p', type=int, default=1000,
                       help="Target equilibrium population (default: 1000)")
    parser.add_argument('--jobs', '-j', type=int, default=32,
                       help="Number of parallel jobs (default: 32)")
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    
    print("="*60)
    print("EXTINCTION SCALING: ALL d' VALUES")
    print("="*60)
    print(f"d' values: {DPRIME_VALUES}")
    print(f"Target population: {args.target_pop}")
    print(f"Parallel jobs: {args.jobs}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Create parameter grid for all d' values
    params = create_parameter_grid(dprime_values=DPRIME_VALUES)
    print(f"\nTotal simulations: {len(params)}")
    for dp in DPRIME_VALUES:
        n_dp = sum(1 for p in params if p.d_prime == dp)
        n_fine = sum(1 for p in params if p.d_prime == dp and p.is_fine_grid)
        print(f"  d'={dp}: {n_dp} points ({n_fine} fine grid)")
    
    # Run all simulations in parallel
    results = run_all_simulations_parallel(
        params, 
        csv_path,
        target_pop=args.target_pop,
        n_jobs=args.jobs,
        verbose=False
    )
    
    # Print summary
    print_results_summary(results)
    
    # Generate all plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Summary plot (4 panels)
    fit_results = plot_summary(results, output_dir)
    
    # Diagnostic plots (population dynamics, density distribution, PCF)
    generate_diagnostic_plots(results, output_dir)
    
    # Print power-law fit results
    print("\n" + "="*60)
    print("POWER-LAW FIT RESULTS")
    print("="*60)
    for d_prime, fit in sorted(fit_results.items()):
        if fit.get('d_ext') and fit.get('beta'):
            print(f"d' = {d_prime}:")
            print(f"  d_ext = {fit['d_ext']:.4f}")
            print(f"  β = {fit['beta']:.3f}")
            print(f"  R² = {fit['r_squared']:.3f}")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Results saved to: {csv_path}")
    print(f"Plots saved to: {output_dir}")
    print(f"Diagnostic plots saved to: {output_dir}/diagnostics/")

if __name__ == "__main__":
    main()
