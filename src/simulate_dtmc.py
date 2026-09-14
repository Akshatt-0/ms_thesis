#!/usr/bin/env python3
"""
Stochastic Approximation Simulation for DTMC Asymptotic Variance Estimation.

This script implements and simulates the recursive O(1) computation and memory
asymptotic variance estimator proposed in Shubhada Agrawal, Prashanth L.A.,
and Siva Theja Maguluri (2024), "Markov Chain Variance Estimation:
A Stochastic Approximation Approach".

It includes:
1. Irreducible and aperiodic DTMC transition and trajectory simulation.
2. The linear SA-based recursive tabular estimator (Algorithm 1).
3. Exact analytical calculations for comparison:
   - Stationary distribution (pi)
   - Stationary expectation (f_bar)
   - Solution of the Poisson equation (V*) orthogonal to 1
   - Asymptotic variance (kappa) using multiple independent formulations.
4. Calculation of theoretically valid constants (c1, c2, c3) from spectral gap.
5. Detailed convergence analysis and comparative evaluation with constant
   and diminishing step sizes.
6. Optional matplotlib-based visualization of convergence behavior.
"""

import argparse
import time
import numpy as np

# Try importing matplotlib, handle failure gracefully
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DiscreteTimeMarkovChain:
    """Class to represent and simulate a Discrete Time Markov Chain (DTMC)."""

    def __init__(self, P, state_names=None):
        """
        Initialize the DTMC with transition matrix P.

        Args:
            P (np.ndarray): S x S transition probability matrix.
            state_names (list, optional): List of state names/labels.
        """
        self.P = np.array(P, dtype=np.float64)
        self.S = self.P.shape[0]

        if self.P.shape[1] != self.S:
            raise ValueError("Transition matrix P must be square.")

        # Validate that P is a valid transition matrix
        if not np.allclose(np.sum(self.P, axis=1), 1.0):
            raise ValueError("Each row of transition matrix P must sum to 1.0.")
        if np.any(self.P < 0.0):
            raise ValueError("Transition matrix P cannot contain negative probabilities.")

        self.state_names = state_names if state_names else [str(i) for i in range(self.S)]

    def compute_stationary_distribution(self):
        """
        Computes the unique stationary distribution pi of the Markov Chain.
        Solves pi^T P = pi^T subject to sum(pi) = 1.

        Returns:
            np.ndarray: Stationary distribution vector pi.
        """
        # We solve the overdetermined system:
        # (P^T - I) pi = 0
        # sum(pi) = 1
        A = np.vstack([self.P.T - np.eye(self.S), np.ones(self.S)])
        b = np.append(np.zeros(self.S), 1.0)

        # Solve using least squares (very robust for small S)
        pi = np.linalg.lstsq(A, b, rcond=None)[0]

        # Enforce non-negativity and sum-to-one
        pi = np.maximum(pi, 0.0)
        pi /= np.sum(pi)
        return pi

    def sample_trajectory(self, n_steps, start_state=None, seed=None):
        """
        Samples a trajectory of state indices X_0, X_1, ..., X_n.

        Args:
            n_steps (int): Total number of transitions to simulate (trajectory length is n_steps + 1).
            start_state (int, optional): Initial state index. If None, samples from stationary distribution.
            seed (int, optional): Random seed.

        Returns:
            np.ndarray: Trajectory of state indices.
        """
        if seed is not None:
            np.random.seed(seed)

        trajectory = np.zeros(n_steps + 1, dtype=np.int32)

        # Initialize start state
        if start_state is None:
            pi = self.compute_stationary_distribution()
            trajectory[0] = np.random.choice(self.S, p=pi)
        else:
            trajectory[0] = start_state

        # Sample transitions
        for k in range(n_steps):
            current_state = trajectory[k]
            probs = self.P[current_state]
            trajectory[k + 1] = np.random.choice(self.S, p=probs)

        return trajectory


class TabularVarianceEstimator:
    """
    Implements Algorithm 1 from the paper for tabular setting:
    Estimating asymptotic variance kappa(f) along a single trajectory.
    """

    def __init__(self, num_states, c1, c2, c3, step_size_func):
        """
        Initialize the estimator state.

        Args:
            num_states (int): S, the size of state space.
            c1 (float): Constant for stationary mean f_bar estimation.
            c2 (float): Constant for stationary mean of V estimation (V_bar).
            c3 (float): Constant for asymptotic variance kappa estimation.
            step_size_func (callable): Function k -> alpha_k returning the step-size.
        """
        self.S = num_states
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.step_size_func = step_size_func

        # Initializations at step 0 (Eq. at top of page 8)
        self.f_bar = 0.0
        self.V = np.zeros(self.S, dtype=np.float64)
        self.V_bar = 0.0
        self.kappa = 0.0

        # Metrics history
        self.history = {
            "f_bar": [],
            "V": [],
            "V_bar": [],
            "kappa": []
        }

    def update(self, k, X_k, f_X_k, X_kp1):
        """
        Perform a single iteration update at time-step k (1-indexed).

        Args:
            k (int): Iteration index (1 to n).
            X_k (int): Current state.
            f_X_k (float): Function value at current state.
            X_kp1 (int): Next observed state.
        """
        alpha_k = self.step_size_func(k)

        # Cache values of current estimates before updates (Remark 2.2)
        old_f_bar = self.f_bar
        old_V = np.copy(self.V)
        old_V_bar = self.V_bar
        old_kappa = self.kappa

        # 1. Average reward f_bar estimation (Equation 10)
        self.f_bar = old_f_bar + self.c1 * alpha_k * (f_X_k - old_f_bar)

        # 2. V estimation (Equations 11 & 12)
        delta_k = f_X_k - old_f_bar + old_V[X_kp1] - old_V[X_k]

        # Compact vector update for Equation 12:
        # V_k+1(x) = V_k(x) - alpha_k * delta_k / S for all x,
        # plus alpha_k * delta_k on X_k.
        self.V = old_V - (alpha_k * delta_k / self.S)
        self.V[X_k] += alpha_k * delta_k

        # 3. V_bar estimation (Equation 13)
        self.V_bar = old_V_bar + self.c2 * alpha_k * (old_V[X_k] - old_V_bar)

        # 4. Asymptotic Variance kappa estimation (Equation 14)
        sa_term = (
            2.0 * f_X_k * old_V[X_k]
            - 2.0 * f_X_k * old_V_bar
            - f_X_k**2
            + f_X_k * old_f_bar
        )
        self.kappa = (1.0 - self.c3 * alpha_k) * old_kappa + self.c3 * alpha_k * sa_term

        # Save history
        self.history["f_bar"].append(self.f_bar)
        self.history["V"].append(np.copy(self.V))
        self.history["V_bar"].append(self.V_bar)
        self.history["kappa"].append(self.kappa)


def compute_spectral_gap(P, pi):
    """
    Computes the mixing-related parameter Delta_1 as defined on page 9:
    Delta_1 = min { v^T D_pi (I - P) v | v in R^S, ||v||_2 = 1, v^T 1 = 0 }

    Args:
        P (np.ndarray): Transition matrix.
        pi (np.ndarray): Stationary distribution.

    Returns:
        float: Delta_1.
    """
    S = P.shape[0]
    # Q = I - 11^T / S (projection onto orthogonal complement of all-ones vector)
    Q = np.eye(S) - np.ones((S, S)) / S

    D_pi = np.diag(pi)
    # Symmetric part of D_pi (I - P)
    M = 0.5 * (D_pi @ (np.eye(S) - P) + (np.eye(S) - P).T @ D_pi)

    # Project the symmetric part: Q M Q
    QMQ = Q @ M @ Q

    # Compute eigenvalues (since QMQ is symmetric, eigh is highly stable)
    eigenvalues = np.linalg.eigh(QMQ)[0]

    # The eigenvalues are sorted in ascending order.
    # The smallest eigenvalue is 0 (corresponding to the all-ones eigenvector).
    # The second smallest eigenvalue is the spectral gap Delta_1 on the subspace orthogonal to 1.
    delta_1 = eigenvalues[1]
    return delta_1


def compute_theoretical_constants(delta_1):
    """
    Computes theoretically valid step-size constants (c1, c2, c3)
    as defined in Theorem 2.1 (page 9) and Appendix D.1.3 (page 56).

    Args:
        delta_1 (float): The spectral gap parameter Delta_1.

    Returns:
        tuple: (c1, c2, c3) satisfying the theorem conditions.
    """
    # 1. c1 >= 1 / (2*Delta_1) + Delta_1 / 2
    c1 = 1.0 / (2.0 * delta_1) + delta_1 / 2.0

    # 2. c3 range: [5/249 * (5 - 2*sqrt(2)) * Delta_1,  5/249 * (5 + 2*sqrt(2)) * Delta_1]
    factor_lower = 5.0 / 249.0 * (5.0 - 2.0 * np.sqrt(2.0))
    factor_upper = 5.0 / 249.0 * (5.0 + 2.0 * np.sqrt(2.0))

    c3_lower = factor_lower * delta_1
    c3_upper = factor_upper * delta_1
    # Choose midpoint of c3
    c3 = 0.5 * (c3_lower + c3_upper)

    # 3. c2 range (from page 9 theorem or page 56, using c3)
    c2_lower = c3 - (498.0 * c3**2) / (7.0 * delta_1) + (7.0 * delta_1) / 498.0

    # Ensure c2 is strictly positive as required by Algorithm 1
    # The mathematical interval can extend below zero, but the algorithm needs c2 > 0 for correct update direction.
    # Therefore we project the interval to be strictly positive.
    c2_lower_pos = max(1e-4, c2_lower)

    inside_sqrt = 498.0 * c3 * delta_1 - 17.0 * delta_1**2
    if inside_sqrt >= 0.0:
        c2_upper = -3.0 * c3 + (5.0 / 83.0) * np.sqrt(inside_sqrt)
        c2_upper_pos = max(c2_lower_pos + 1e-4, c2_upper)
        c2 = 0.5 * (c2_lower_pos + c2_upper_pos)
    else:
        # Fallback to robust empirical choice if theoretical bound is complex
        c2 = 2.0 * delta_1

    return c1, c2, c3


def compute_analytical_ground_truth(P, pi, f):
    """
    Computes exact analytical ground truths for the DTMC and function f.

    Solves:
    1. Stationary expectation of f: f_bar = pi^T f
    2. Poisson equation: (I - P) V = f - f_bar * 1, orthogonal to 1 (V_star)
    3. Stationary expectation of V: V_bar = pi^T V_star
    4. Asymptotic variance kappa(f) using two independent mathematical formulations:
       - Form 1 (Equation 5): 2 * E_pi[(f(X) - f_bar) * V(X)] - E_pi[(f(X) - f_bar)^2]
       - Form 2 (Equation 6): E_pi[V^2(X)] - E_pi[(P V)^2(X)]

    Returns:
        dict: Exact ground truth values.
    """
    S = P.shape[0]
    f_bar = np.dot(pi, f)

    # Solve the Poisson equation (I - P) V = f - f_bar * 1 subject to 1^T V = 0
    # Setup as linear least squares for overdetermined system
    A_poisson = np.vstack([np.eye(S) - P, np.ones(S)])
    b_poisson = np.append(f - f_bar, 0.0)
    V_star = np.linalg.lstsq(A_poisson, b_poisson, rcond=None)[0]

    # Stationary expectation of V*
    V_bar_star = np.dot(pi, V_star)

    # Form 1 of Asymptotic Variance (Equation 5)
    term1_f1 = 2.0 * np.sum(pi * (f - f_bar) * V_star)
    term2_f1 = np.sum(pi * (f - f_bar)**2)
    kappa_form1 = term1_f1 - term2_f1

    # Form 2 of Asymptotic Variance (Equation 6)
    term1_f2 = np.sum(pi * V_star**2)
    PV_star = P @ V_star
    term2_f2 = np.sum(pi * PV_star**2)
    kappa_form2 = term1_f2 - term2_f2

    # Assert that both formulations are mathematically equivalent and yield identical results
    assert np.allclose(kappa_form1, kappa_form2, atol=1e-10), (
        f"Verification failed: Form 1 ({kappa_form1:.6f}) and Form 2 ({kappa_form2:.6f}) do not match!"
    )

    return {
        "f_bar": f_bar,
        "V_star": V_star,
        "V_bar_star": V_bar_star,
        "kappa": kappa_form1
    }


def generate_plots(history_dim, history_const, ground_truth, save_path):
    """Generates convergence plots for both diminishing and constant step sizes."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping plot generation.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Diminishing details
    steps = len(history_dim["kappa"])
    x_dim = np.arange(1, steps + 1)
    x_const = np.arange(1, len(history_const["kappa"]) + 1)

    # 1. Asymptotic Variance kappa(f)
    axs[0, 0].plot(x_dim, history_dim["kappa"], label="Diminishing step-size", color="blue", alpha=0.8)
    axs[0, 0].plot(x_const, history_const["kappa"], label="Constant step-size", color="orange", alpha=0.8)
    axs[0, 0].axhline(y=ground_truth["kappa"], color="red", linestyle="--", label="True kappa(f)")
    axs[0, 0].set_title("Estimation of Asymptotic Variance $\kappa(f)$")
    axs[0, 0].set_xlabel("Transitions ($k$)")
    axs[0, 0].set_ylabel("Estimate")
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)

    # 2. Stationary Mean f_bar
    axs[0, 1].plot(x_dim, history_dim["f_bar"], label="Diminishing step-size", color="blue", alpha=0.8)
    axs[0, 1].plot(x_const, history_const["f_bar"], label="Constant step-size", color="orange", alpha=0.8)
    axs[0, 1].axhline(y=ground_truth["f_bar"], color="red", linestyle="--", label="True $\\bar{f}$")
    axs[0, 1].set_title("Estimation of Stationary Expectation $\\bar{f}$")
    axs[0, 1].set_xlabel("Transitions ($k$)")
    axs[0, 1].set_ylabel("Estimate")
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)

    # 3. Stationary Mean of V (V_bar)
    axs[1, 0].plot(x_dim, history_dim["V_bar"], label="Diminishing step-size", color="blue", alpha=0.8)
    axs[1, 0].plot(x_const, history_const["V_bar"], label="Constant step-size", color="orange", alpha=0.8)
    axs[1, 0].axhline(y=ground_truth["V_bar_star"], color="red", linestyle="--", label="True $\\bar{V}^*$")
    axs[1, 0].set_title("Estimation of $V$ Stationary Expectation $\\bar{V}^*$")
    axs[1, 0].set_xlabel("Transitions ($k$)")
    axs[1, 0].set_ylabel("Estimate")
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)

    # 4. Error metrics over time (L2 Norm of V - V_star)
    V_err_dim = [np.linalg.norm(v - ground_truth["V_star"]) for v in history_dim["V"]]
    V_err_const = [np.linalg.norm(v - ground_truth["V_star"]) for v in history_const["V"]]
    axs[1, 1].loglog(x_dim, V_err_dim, label="Diminishing step-size", color="blue", alpha=0.8)
    axs[1, 1].loglog(x_const, V_err_const, label="Constant step-size", color="orange", alpha=0.8)
    axs[1, 1].set_title("L2-Error of Value Function $V_k$ (Log-Log Scale)")
    axs[1, 1].set_xlabel("Transitions ($k$)")
    axs[1, 1].set_ylabel("$||V_k - V^*||_2$")
    axs[1, 1].legend()
    axs[1, 1].grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nConvergence plots successfully saved to: {save_path}")


def run_simulation(dtmc, f, ground_truth, args, steps, c1, c2, c3):
    """Runs simulation for both step-size setups and returns estimates and histories."""
    # Generate single trajectory for both algorithms to ensure fair comparison
    print(f"Sampling single trajectory of length {steps} from DTMC...")
    trajectory = dtmc.sample_trajectory(steps, seed=args.seed)

    # 1. Diminishing Step Size: alpha_k = alpha / (k + h)
    def step_size_dim(k):
        return args.alpha / (k + args.h)

    # 2. Constant Step Size: alpha_k = constant_alpha
    def step_size_const(k):
        return args.constant_alpha

    print(f"\nRunning Algorithm 1 with Diminishing Step Size (alpha={args.alpha}, h={args.h})...")
    est_dim = TabularVarianceEstimator(dtmc.S, c1, c2, c3, step_size_dim)
    t_start = time.time()
    for k in range(1, steps + 1):
        X_k = trajectory[k - 1]
        f_X_k = f[X_k]
        X_kp1 = trajectory[k]
        est_dim.update(k, X_k, f_X_k, X_kp1)
    t_dim = time.time() - t_start

    print(f"Running Algorithm 1 with Constant Step Size (alpha={args.constant_alpha})...")
    est_const = TabularVarianceEstimator(dtmc.S, c1, c2, c3, step_size_const)
    t_start = time.time()
    for k in range(1, steps + 1):
        X_k = trajectory[k - 1]
        f_X_k = f[X_k]
        X_kp1 = trajectory[k]
        est_const.update(k, X_k, f_X_k, X_kp1)
    t_const = time.time() - t_start

    # Print comparative report
    print("\n" + "=" * 80)
    print("                      SIMULATION CONVERGENCE REPORT")
    print("=" * 80)
    print(f"{'Metric':<15} | {'Ground Truth':<15} | {'Diminishing Estimate':<22} | {'Constant Estimate':<20}")
    print("-" * 80)

    # Print f_bar comparison
    print(f"{'mean(f)':<15} | {ground_truth['f_bar']:<15.6f} | {est_dim.f_bar:<22.6f} | {est_const.f_bar:<20.6f}")

    # Print V_bar comparison
    print(f"{'mean(V*)':<15} | {ground_truth['V_bar_star']:<15.6f} | {est_dim.V_bar:<22.6f} | {est_const.V_bar:<20.6f}")

    # Print kappa comparison
    err_dim = abs(est_dim.kappa - ground_truth["kappa"])
    err_const = abs(est_const.kappa - ground_truth["kappa"])
    print(f"{'kappa(f)':<15} | {ground_truth['kappa']:<15.6f} | "
          f"{est_dim.kappa:<22.6f} | {est_const.kappa:<20.6f}")
    print("-" * 80)
    print(f"{'Absolute Error':<15} | {'':<15} | {err_dim:<22.6e} | {err_const:<20.6e}")
    print(f"{'Execution Time':<15} | {'':<15} | {t_dim:<19.3f}s | {t_const:<17.3f}s")
    print("=" * 80)

    print("\nExact Value Function V* Comparison:")
    print(f"{'State':<10} | {'True V*':<15} | {'Diminishing V_n':<22} | {'Constant V_n':<20}")
    print("-" * 80)
    for i in range(dtmc.S):
        print(f"{dtmc.state_names[i]:<10} | {ground_truth['V_star'][i]:<15.6f} | "
              f"{est_dim.V[i]:<22.6f} | {est_const.V[i]:<20.6f}")
    print("=" * 80)

    return est_dim.history, est_const.history


def main():
    parser = argparse.ArgumentParser(description="Simulate DTMC Asymptotic Variance SA algorithm.")
    parser.add_argument("--steps", type=int, default=100000, help="Number of simulation transition steps.")
    parser.add_argument("--states", type=int, default=4, help="Size of state space.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--alpha", type=float, default=25.0, help="Numerator step size for diminishing sequence.")
    parser.add_argument("--h", type=float, default=200.0, help="Shift constant for diminishing step size.")
    parser.add_argument("--constant_alpha", type=float, default=0.005, help="Step-size for constant sequence.")
    parser.add_argument("--use_empirical_constants", action="store_true",
                        help="Skip spectral gap calculations and use empirical c1=1, c2=1, c3=1.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Let's design an interesting S-state irreducible and aperiodic Markov chain
    # Standard Dirichlet rows generate random stochastic vectors
    S = args.states
    if S == 4:
        # Define a fixed non-trivial 4-state Markov Chain for reproducibility
        P = np.array([
            [0.1, 0.7, 0.2, 0.0],
            [0.0, 0.2, 0.5, 0.3],
            [0.4, 0.0, 0.1, 0.5],
            [0.6, 0.1, 0.0, 0.3]
        ])
        f = np.array([-0.8, 0.5, -0.2, 1.0])  # Bounded in [-1, 1]
    else:
        # Randomly generate transition matrix and function values
        P = np.random.dirichlet(np.ones(S) * 0.5, size=S)
        # Mix in a small identity to ensure aperiodic and irreducible walk
        P = 0.8 * P + 0.2 * np.eye(S)
        # Random reward values in [-1.0, 1.0]
        f = np.random.uniform(-1.0, 1.0, size=S)

    dtmc = DiscreteTimeMarkovChain(P)
    pi = dtmc.compute_stationary_distribution()

    print(f"Initialized DTMC with {S} states:")
    print("Transition Probability Matrix P:")
    print(P)
    print("\nStationary Distribution pi:")
    print(pi)
    print("\nFunction f(X) defined on state space:")
    print(f)

    # Verify mixing properties and spectral gap Delta_1
    delta_1 = compute_spectral_gap(P, pi)
    print(f"\nSpectral Gap parameter Delta_1: {delta_1:.6f}")

    # Select step size scale constants c1, c2, c3
    if args.use_empirical_constants:
        print("Using empirical constants: c1 = 1.0, c2 = 1.0, c3 = 1.0")
        c1, c2, c3 = 1.0, 1.0, 1.0
    else:
        # Calculate theoretically guaranteed constants from Theorem 2.1
        c1, c2, c3 = compute_theoretical_constants(delta_1)
        print("Calculated theoretically valid constants:")
        print(f"  c1 = {c1:.6f} (Requirement: c1 >= {1.0 / (2 * delta_1) + delta_1 / 2.0:.6f})")
        print(f"  c2 = {c2:.6f}")
        print(f"  c3 = {c3:.6f}")

    # Compute ground truth analytically
    ground_truth = compute_analytical_ground_truth(P, pi, f)
    print(f"\nAnalytical Ground Truth Values:")
    print(f"  Stationary expectation mean(f) = {ground_truth['f_bar']:.6f}")
    print(f"  Stationary expectation mean(V) = {ground_truth['V_bar_star']:.6f}")
    print(f"  Poisson solution vector V*     = {ground_truth['V_star']}")
    print(f"  Asymptotic Variance kappa(f)   = {ground_truth['kappa']:.6f}")

    # Run simulations
    history_dim, history_const = run_simulation(
        dtmc, f, ground_truth, args, args.steps, c1, c2, c3
    )

    # Save visualization plot
    generate_plots(history_dim, history_const, ground_truth, "src/convergence_plot.png")


if __name__ == "__main__":
    main()
