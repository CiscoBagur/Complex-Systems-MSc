import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

# Function to obtain the density profile of a system
def get_density_profile(positions, L, bin_width):
    """Calculate the density profile along the y-axis."""
    Ly = L[1]
    n_bins = int(Ly / bin_width)
    density_profile = np.zeros(n_bins)
    
    for pos in positions:
        bin_index = int(pos[1] / bin_width)
        if 0 <= bin_index < n_bins:
            density_profile[bin_index] += 1
    
    # Normalize by area of each bin
    area_per_bin = L[0] * bin_width
    density_profile /= area_per_bin
    
    bin_centers = (np.arange(n_bins) + 0.5) * bin_width
    return bin_centers, density_profile

# JIT-compiled helper functions for maximum performance
@jit(nopython=True)
def check_overlaps_numba(positions, new_position, index, sigma, L):
    """JIT-compiled overlap checking with hard walls."""
    radius = sigma / 2
    
    # Check hard wall boundaries first (fast rejection)
    if (new_position[0] < radius or new_position[0] > L[0] - radius or
        new_position[1] < radius or new_position[1] > L[1] - radius):
        return True
    
    # Check particle overlaps
    for j in range(len(positions)):
        if j != index:
            dx = new_position[0] - positions[j, 0]
            dy = new_position[1] - positions[j, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < sigma*sigma:
                return True
    return False

@jit(nopython=True)
def check_overlaps_periodic_numba(positions, new_position, index, sigma, L_scalar):
    """JIT-compiled overlap checking with periodic boundaries."""
    for j in range(len(positions)):
        if j != index:
            # Minimum image convention
            dx = new_position[0] - positions[j, 0]
            dy = new_position[1] - positions[j, 1]
            
            dx = dx - L_scalar * np.round(dx / L_scalar)
            dy = dy - L_scalar * np.round(dy / L_scalar)
            
            dist_sq = dx*dx + dy*dy
            if dist_sq < sigma*sigma:
                return True
    return False

@jit(nopython=True)
def metropolis_move_numba(positions, N, delta, sigma, L, g, T, use_periodic):
    """JIT-compiled Metropolis move."""
    i = np.random.randint(N)
    move = (np.random.rand(2) - 0.5) * delta
    if g != 0:
        acceptance_prob = min(1,np.exp(-g * move[1] / T))
        if np.random.rand() > acceptance_prob:
            return positions  # Reject move based on gravity
    
    if use_periodic:
        # Periodic boundary conditions (scalar L only)
        new_position = (positions[i] + move) % L[0]
        overlap = check_overlaps_periodic_numba(positions, new_position, i, sigma, L[0])
    else:
        # Hard walls
        new_position = positions[i] + move
        overlap = check_overlaps_numba(positions, new_position, i, sigma, L)
    
    if not overlap:
        positions[i] = new_position
    
    return positions

@jit(nopython=True)
def run_simulation_numba(positions, N, delta, sigma, L, g, T, steps, use_periodic):
    """JIT-compiled simulation loop for maximum performance."""
    trajectory = np.empty((steps, N, 2), dtype=np.float32)
    
    for step in range(steps):
        positions = metropolis_move_numba(positions, N, delta, sigma, L, g, T,use_periodic)
        trajectory[step] = positions.copy()
    
    return trajectory, positions

@jit(nopython=True)
def run_chill_simulation(positions, N, delta, sigma, L, g, T, steps, use_periodic):
    """JIT-compiled simulation loop for chill mode (no trajectory saving)."""
    for step in range(steps):
        positions = metropolis_move_numba(positions, N, delta, sigma, L, g, T,use_periodic)
    
    return positions

class Trajectory:
    def __init__(self, positions_array, sigma, L, phi):
        """
        Args:
            positions_array: NumPy array of shape (n_steps, N, 2)
            sigma: Particle diameter
            L: Box dimensions (scalar or array)
            phi: Packing fraction
        """
        self.positions = positions_array
        self.sigma = sigma
        self.L = np.array(L) if not isinstance(L, np.ndarray) else L
        self._unwrapped_cache = None
        self.N = positions_array.shape[1]
        self.phi = phi
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, index):
        return self.positions[index]
    
    def _unwrap_trajectory(self):
        """Unwrap trajectory to account for periodic boundary conditions."""
        if self._unwrapped_cache is not None:
            return self._unwrapped_cache
        
        n_steps = len(self.positions)
        
        # Initialize unwrapped positions
        unwrapped = np.zeros_like(self.positions)
        unwrapped[0] = self.positions[0]
        
        # Vectorized unwrapping
        for t in range(1, n_steps):
            delta = self.positions[t] - self.positions[t-1]
            delta -= self.L * np.round(delta / self.L)
            unwrapped[t] = unwrapped[t-1] + delta
        
        self._unwrapped_cache = unwrapped
        return unwrapped
    
    def visualize(self):
        """Visualize the trajectory animation."""
        plt.ion()
        
        # Adjust figure size based on box dimensions
        if self.L.ndim == 0:  # Scalar L (square box)
            fig, ax = plt.subplots(figsize=(6, 6))
            xlim, ylim = (0, self.L), (0, self.L)
        else:  # Array L (rectangular box)
            aspect = self.L[1] / self.L[0]
            fig, ax = plt.subplots(figsize=(8, 8 * aspect))
            xlim, ylim = (0, self.L[0]), (0, self.L[1])
        
        try:
            for t in range(len(self.positions)):
                if not plt.fignum_exists(fig.number):
                    print("Window closed, stopping visualization")
                    break
                    
                ax.clear()
                
                # Draw box boundaries
                if self.L.ndim > 0:
                    from matplotlib.patches import Rectangle
                    wall = Rectangle((0, 0), self.L[0], self.L[1], 
                                   fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(wall)
                
                # Draw all particles at once using scatter for speed
                positions = self.positions[t]
                for pos in positions:
                    circle = plt.Circle(pos, self.sigma/2, color='blue', alpha=0.5)
                    ax.add_artist(circle)
                         
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(f'Time step: {t}/{len(self.positions)-1} (Close window to stop)')
                
                plt.draw()
                plt.pause(0.05)
                
        except KeyboardInterrupt:
            print("\nVisualization stopped by Ctrl+C")
        finally:
            plt.ioff()
            plt.close(fig)
            
    def plot(self, timestep):
        """Plot a given timestep of the trajectory."""
        if self.L.ndim == 0:  # Scalar L (square box)
            plt.figure(figsize=(8,8))
            xlim, ylim = (0, self.L), (0, self.L)
        else:  # Array L (rectangular box)
            plt.figure(figsize=(8, 8 * self.L[1]/self.L[0]))
            xlim, ylim = (0, self.L[0]), (0, self.L[1])
            
            # Draw hard walls
            from matplotlib.patches import Rectangle
            wall = Rectangle((0, 0), self.L[0], self.L[1], 
                            fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(wall)
        
        # Draw particles
        positions = self.positions[timestep]
        for pos in positions:
            circle = plt.Circle(pos, self.sigma/2, color='blue', alpha=0.5)
            plt.gca().add_artist(circle)
        
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.title(f'Hard Box: N={self.N}, Lx={self.L[0]:.2f}, Ly={self.L[1]:.2f} at t={timestep}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
            
    def get_msd(self):
        """Calculate mean squared displacement (MSD) over time."""
        positions = self._unwrap_trajectory()
        # Vectorized MSD calculation
        msd = np.mean(np.sum((positions - positions[0])**2, axis=2), axis=1)
        return msd
    
    def get_diffusion_coefficient(self, fit_range=None):
        """Estimate diffusion coefficient from MSD."""

        msd = self.get_msd()
        time_steps = np.arange(len(msd))
            
        # Use middle portion for fit (avoid initial transients and noise at end)
        if fit_range is None:
            start = len(time_steps) // 4
            end = 3 * len(time_steps) // 4
        else:
            start, end = fit_range
            
        # Get fit parameters and covariance matrix
        params, cov = np.polyfit(time_steps[start:end], msd[start:end], 1, cov=True)
        slope, intercept = params
        
        # Extract standard errors from diagonal of covariance matrix
        slope_err = np.sqrt(cov[0, 0])
        # intercept_err = np.sqrt(cov[1, 1])
        
        # Calculate D and its error
        D = slope / 4  # In 2D, D = slope/4
        D_err = slope_err / 4  # Error propagates linearly
        
        return D, D_err

class Box:
    def __init__(self, phi, N, sigma=1):
        self.phi = phi
        self.N = N
        self.L = np.sqrt(N*sigma**2*np.pi/(4*phi))
        self.sigma = sigma
        self.positions = np.empty((0, 2), dtype=np.float32)  # NumPy array from start
    
    def _add_random_disk(self):
        """Add a random disk without overlaps."""
        max_attempts = 1000
        for _ in range(max_attempts):
            new_disk = np.random.rand(2) * self.L
            
            if len(self.positions) == 0:
                self.positions = new_disk.reshape(1, 2)
                return True
            
            # Vectorized distance check
            distances = np.linalg.norm(self.positions - new_disk, axis=1)
            if np.all(distances >= self.sigma):
                self.positions = np.vstack([self.positions, new_disk])
                return True
        
        print("Failed to place disk after maximum attempts")
        return False
    
    def initialize_random_disks(self):
        """Initialize with random disk positions."""
        while len(self.positions) < self.N:
            if not self._add_random_disk():
                break
    
    def initialize_triangular_lattice(self):
        """Initialize with triangular lattice."""
        n_side = int(np.ceil(np.sqrt(self.N)))
        spacing = self.L/n_side
        positions_list = []
        
        for i in range(n_side):
            for j in range(n_side):
                if len(positions_list) < self.N:
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing
                    positions_list.append([x, y])
        
        self.positions = np.array(positions_list, dtype=np.float32)
    
    
    def run_simulation(self, delta, steps, save_interval=1):
        """Run optimized simulation, saving every 'save_interval' steps."""
        start_time = time.time()
        
        print("Running simulation with Numba JIT compilation...")
        print(f"N={self.N}, steps={steps}, delta={delta}")
        
        # Use JIT-compiled simulation
        L_array = np.array([self.L, self.L])
        trajectory, final_positions = run_simulation_numba(
            self.positions.copy(), self.N, delta, self.sigma, 
            L_array, 0.0, 1.0 ,steps, use_periodic=True
        )
        
        # Downsample trajectory if requested
        if save_interval > 1:
            trajectory = trajectory[::save_interval]
            print(f"  Downsampled: {len(trajectory)} frames saved (1/{save_interval} of total)")
        
        
        self.positions = final_positions
        self.trajectory = Trajectory(trajectory, self.sigma, self.L, self.phi)
        self.simulation_time = time.time() - start_time

        print(f"✓ Simulation completed in {self.simulation_time:.2f} seconds")
        print(f"  Performance: {steps/self.simulation_time:.0f} steps/second")
        return self.trajectory
    
    def run_chill_simulation(self, delta, steps):
        """Run simulation in chill mode (no trajectory saving)."""
        start_time = time.time()
        
        print("Running chill simulation with Numba JIT compilation...")
        print(f"N={self.N}, steps={steps}, delta={delta}")
        
        L_array = np.array([self.L, self.L])
        final_positions = run_chill_simulation(
            self.positions.copy(), self.N, delta, self.sigma, 
            L_array, 0.0, 1.0 ,steps, use_periodic=True
        )
        
        self.positions = final_positions
        self.simulation_time = time.time() - start_time

        print(f"✓ Chill simulation completed in {self.simulation_time:.2f} seconds")
        print(f"  Performance: {steps/self.simulation_time:.0f} steps/second")
        return self.positions 
    
    def plot(self):
        """Plot current configuration."""
        plt.figure(figsize=(6,6))
        for pos in self.positions:
            circle = plt.Circle(pos, self.sigma/2, color='blue', alpha=0.5)
            plt.gca().add_artist(circle)
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.title(f'Hard Spheres: N={self.N}, φ={self.phi:.3f}')
        plt.show()
    
class HardBox(Box):
    def __init__(self, phi=None, N=None, sigma=1, Lx=None, Ly=None, g=0, T=1):
        """
        Hard box with hard walls (non-periodic boundaries).
        
        Usage:
        ------
        # Option 1: Use phi (square box)
        box = HardBox(phi=0.05, N=100, sigma=1)

        # Option 2: Specify dimensions directly
        box = HardBox(N=100, sigma=1, Lx=20, Ly=10)
        """
        self.g = g
        self.T = T
        
        if Lx is not None and Ly is not None:
            self.N = N
            self.sigma = sigma
            self.L = np.array([Lx, Ly])
            self.phi = N * sigma**2 * np.pi / (4 * Lx * Ly)
            self.positions = np.empty((0, 2))
        else:
            if phi is None:
                raise ValueError("Must provide either (phi) or (Lx, Ly)")
            super().__init__(phi, N, sigma)
            self.L = np.array([self.L, self.L])
    
    def _add_random_disk(self):
        """Add random disk respecting hard walls."""
        max_attempts = 1000
        radius = self.sigma / 2
        
        for _ in range(max_attempts):
            new_disk = np.array([
                np.random.rand() * (self.L[0] - 2*radius) + radius,
                np.random.rand() * (self.L[1] - 2*radius) + radius
            ])
            
            if len(self.positions) == 0:
                self.positions = new_disk.reshape(1, 2)
                return True
            
            distances = np.linalg.norm(self.positions - new_disk, axis=1)
            if np.all(distances >= self.sigma):
                self.positions = np.vstack([self.positions, new_disk])
                return True
        
        print("Failed to place disk after maximum attempts")
        return False
    
    def run_simulation(self, delta, steps, save_interval=1):
        """Run optimized simulation with hard walls, and save every 'save_interval' steps."""
        start_time = time.time()
        
        print(f"N={self.N}, steps={steps}, delta={delta}, g={self.g}")
        
        # Use JIT-compiled simulation
        trajectory, final_positions = run_simulation_numba(
            self.positions.copy(), self.N, delta, self.sigma, 
            self.L, self.g, self.T ,steps, use_periodic=False
        )
        
        # Downsample trajectory if requested
        if save_interval > 1:
            trajectory = trajectory[::save_interval]
            print(f"  Downsampled: {len(trajectory)} frames saved (1/{save_interval} of total)")
        
        
        self.positions = final_positions
        self.trajectory = Trajectory(trajectory, self.sigma, self.L, self.phi)
        self.simulation_time = time.time() - start_time

        print(f"✓ Simulation completed in {self.simulation_time:.2f} seconds")
        print(f"  Performance: {steps/self.simulation_time:.2f} steps/second")
        return self.trajectory
    
    def run_chill_simulation(self, delta, steps):
        """Run chill simulation with hard walls (no trajectory saving)."""
        start_time = time.time()
        
        print(f"N={self.N}, steps={steps}, delta={delta}, g={self.g}")
        
        final_positions = run_chill_simulation(
            self.positions.copy(), self.N, delta, self.sigma, 
            self.L, self.g, self.T ,steps, use_periodic=False
        )
        
        self.positions = final_positions
        self.simulation_time = time.time() - start_time

        print(f"✓ Chill simulation completed in {self.simulation_time:.2f} seconds")
        print(f"  Performance: {steps/self.simulation_time:.2f} steps/second")
        return self.positions
    
    def plot(self):
        """Plot with rectangular box."""
        plt.figure(figsize=(8, 8 * self.L[1]/self.L[0]))
        
        # Draw hard walls
        from matplotlib.patches import Rectangle
        wall = Rectangle((0, 0), self.L[0], self.L[1], 
                        fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(wall)
        
        # Draw particles
        for pos in self.positions:
            circle = plt.Circle(pos, self.sigma/2, color='blue', alpha=0.5)
            plt.gca().add_artist(circle)
        
        plt.xlim(-0.5, self.L[0] + 0.5)
        plt.ylim(-0.5, self.L[1] + 0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.title(f'Hard Box: N={self.N}, Lx={self.L[0]:.2f}, Ly={self.L[1]:.2f}, φ={self.phi:.3f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


# Example usage and performance comparison
if __name__ == "__main__":
    phi = 0.1
    N = 1000
    steps = 100000000
    save_interval = N
    Lx= 16
    Ly = 10*Lx
    
    box = HardBox(N=N, sigma=1, Lx=Lx, Ly=Ly, g=10, T=1)
    box.initialize_random_disks()
    positions = box.positions
    final_positions = run_chill_simulation(
        positions.copy(), box.N, delta=0.3, sigma=box.sigma, L=box.L, g=box.g, 
        T=box.T, steps=steps, use_periodic=False)
    
    box.positions = final_positions
    box.plot()
    