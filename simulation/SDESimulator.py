import numpy as np


def simulate_spatial_birth_death(
        L=10.0,  # domain length
        N=1000,  # number of spatial grid points
        dt=0.01,  # time step
        T=10.0,  # final time
        b=2.0,  # birth rate
        d=1.0,  # intrinsic death rate
        dd=0.1,  # competition rate
        lam=1.0,  # initial density (individuals per unit length)
        sigma_m=1.0,  # standard deviation for dispersal (birth) kernel
        sigma_w=0.1,  # standard deviation for competition kernel
):
    """
    Simulate a spatial birth-death-competition model with multiplicative noise,
    using an Euler-Maruyama scheme with FFT-based convolution for the kernels.

    Parameters
    ----------
    L : float
        Domain length.
    N : int
        Number of spatial grid points.
    dt : float
        Time step.
    T : float
        Final simulation time.
    b : float
        Birth rate.
    d : float
        Intrinsic death rate.
    dd : float
        Competition rate.
    lam : float
        Initial density (individuals per unit length).
    sigma_m : float
        Standard deviation of the dispersal (birth) kernel.
    sigma_w : float
        Standard deviation of the competition kernel.

    Returns
    -------
    results : dict
        Dictionary with keys:
          'final_phi': list of final density arrays (length num_runs)
          'x': spatial grid (array)
    """
    # Derived parameters:
    dx = L / N
    steps = int(T / dt)

    # Spatial grid: x from -L/2 to L/2 (endpoint excluded)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)

    # Build dispersal kernel: Gaussian normalized as a probability density.
    kernel_disp = np.exp(-x ** 2 / (2 * sigma_m ** 2))
    kernel_disp /= (np.sum(kernel_disp) * dx)

    # Build competition kernel: raw Gaussian (amplitude scales as 1/(sqrt(2pi)*sigma_w))
    kernel_comp = np.exp(-x ** 2 / (2 * sigma_w ** 2)) / (np.sqrt(2 * np.pi) * sigma_w)
    # Do not renormalize kernel_comp further.

    # Shift kernels for FFT convolution (centered at index 0)
    kernel_disp_fft = np.fft.fft(np.fft.ifftshift(kernel_disp))
    kernel_comp_fft = np.fft.fft(np.fft.ifftshift(kernel_comp))

    def fft_convolve(phi, kernel_fft):
        """
        Compute the convolution of phi with a kernel provided in Fourier space,
        using periodic boundary conditions.
        """
        phi_fft = np.fft.fft(phi)
        conv = np.real(np.fft.ifft(phi_fft * kernel_fft)) * dx
        return conv

    # Initial condition: sample from a Poisson point process (PPP)
    n_init = np.random.poisson(lam * dx, size=N)
    # Convert to density (individuals per unit length)
    phi = n_init.astype(float) / dx

    # Time integration: Euler-Maruyama scheme
    for n in range(1, steps):
        # Convolution terms via FFT
        birth_conv = fft_convolve(phi, kernel_disp_fft)
        comp_conv = fft_convolve(phi, kernel_comp_fft)

        # Deterministic drift
        drift = b * birth_conv - d * phi - dd * phi * comp_conv

        # Noise amplitude (variance per dt)
        Gamma = b * birth_conv + d * phi + dd * phi * comp_conv
        Gamma = np.maximum(Gamma, 0)

        # Generate multiplicative Gaussian noise (independent per grid cell)
        noise = np.sqrt(Gamma) * np.random.normal(0, 1, size=N)

        # Euler-Maruyama update
        phi = phi + dt * drift + np.sqrt(dt) * noise

        # Enforce non-negativity of density
        phi = np.maximum(phi, 0)

    results = {
        'final_phi': phi,
        'x': x,
    }

    return results
