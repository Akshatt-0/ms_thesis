#!/usr/bin/env python3
"""
Unit Tests for DTMC Variance Estimation Simulation.

This module uses Python's standard `unittest` framework to verify the correctness,
stability, and mathematical consistency of the classes and functions implemented
in `src/simulate_dtmc.py`.
"""

import unittest
import numpy as np
from simulate_dtmc import (
    DiscreteTimeMarkovChain,
    TabularVarianceEstimator,
    compute_spectral_gap,
    compute_theoretical_constants,
    compute_analytical_ground_truth
)


class TestDiscreteTimeMarkovChain(unittest.TestCase):
    """Test suite for the DiscreteTimeMarkovChain class."""

    def setUp(self):
        # A simple, well-known 2-state Markov chain
        # State 0 -> State 1 with prob 0.6, stays with 0.4
        # State 1 -> State 0 with prob 0.8, stays with 0.2
        self.P_2state = np.array([
            [0.4, 0.6],
            [0.8, 0.2]
        ])
        self.dtmc_2state = DiscreteTimeMarkovChain(self.P_2state, state_names=["A", "B"])

    def test_validation(self):
        """Verify that invalid transition matrices are correctly rejected."""
        # Non-square matrix
        with self.assertRaises(ValueError):
            DiscreteTimeMarkovChain(np.array([[0.5, 0.5, 0.0], [0.1, 0.9]]))

        # Row doesn't sum to 1.0
        with self.assertRaises(ValueError):
            DiscreteTimeMarkovChain(np.array([[0.5, 0.4], [0.2, 0.8]]))

        # Negative probability
        with self.assertRaises(ValueError):
            DiscreteTimeMarkovChain(np.array([[1.1, -0.1], [0.3, 0.7]]))

    def test_stationary_distribution(self):
        """Verify stationary distribution solving."""
        pi = self.dtmc_2state.compute_stationary_distribution()
        self.assertEqual(len(pi), 2)
        self.assertAlmostEqual(np.sum(pi), 1.0)
        # Analytical pi: pi_0 * 0.6 = pi_1 * 0.8 => pi_0 = 4/3 pi_1
        # pi_0 + pi_1 = 1 => 7/3 pi_1 = 1 => pi_1 = 3/7, pi_0 = 4/7
        self.assertAlmostEqual(pi[0], 4.0 / 7.0)
        self.assertAlmostEqual(pi[1], 3.0 / 7.0)

    def test_sample_trajectory(self):
        """Verify trajectory sampling constraints."""
        steps = 500
        trajectory = self.dtmc_2state.sample_trajectory(steps, start_state=0, seed=123)
        self.assertEqual(len(trajectory), steps + 1)
        self.assertEqual(trajectory[0], 0)
        # Verify all elements are valid state indices
        self.assertTrue(np.all((trajectory == 0) | (trajectory == 1)))


class TestTabularVarianceEstimator(unittest.TestCase):
    """Test suite for the TabularVarianceEstimator class."""

    def test_orthogonal_conservation(self):
        """
        Verify that V remains exactly orthogonal to 1 at every single step,
        even with large updates, which is a key mathematical feature of the projection-free step.
        """
        S = 4
        # Set up a random estimator
        c1, c2, c3 = 1.0, 1.0, 1.0
        estimator = TabularVarianceEstimator(
            num_states=S,
            c1=c1,
            c2=c2,
            c3=c3,
            step_size_func=lambda k: 0.1 / k
        )

        # Confirm V starts as zero vector (orthogonal to 1)
        self.assertAlmostEqual(np.sum(estimator.V), 0.0)

        # Run random transition updates and check orthogonal conservation at each step
        np.random.seed(42)
        for k in range(1, 100):
            X_k = np.random.randint(0, S)
            X_kp1 = np.random.randint(0, S)
            f_X_k = np.random.uniform(-1.0, 1.0)
            estimator.update(k, X_k, f_X_k, X_kp1)
            # sum of V components must remain exactly 0.0 (within numerical precision)
            self.assertAlmostEqual(np.sum(estimator.V), 0.0, places=12)


class TestAnalyticalAndTheoreticalSolvers(unittest.TestCase):
    """Test suite for Delta_1 spectral gap and analytical ground truth solvers."""

    def setUp(self):
        # A 3-state circular random walk (irreducible, aperiodic)
        self.P = np.array([
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.7, 0.1, 0.2]
        ])
        self.f = np.array([-0.5, 0.0, 0.9])
        self.dtmc = DiscreteTimeMarkovChain(self.P)
        self.pi = self.dtmc.compute_stationary_distribution()

    def test_spectral_gap_and_constants(self):
        """Test spectral gap Delta_1 computation and valid constants derivation."""
        delta_1 = compute_spectral_gap(self.P, self.pi)
        self.assertGreater(delta_1, 0.0, "Spectral gap Delta_1 must be strictly positive.")

        c1, c2, c3 = compute_theoretical_constants(delta_1)
        # Check requirements
        self.assertGreaterEqual(c1, 1.0 / (2.0 * delta_1) + delta_1 / 2.0)
        self.assertGreater(c2, 0.0, "Scale constant c2 must be strictly positive.")
        self.assertGreater(c3, 0.0, "Scale constant c3 must be strictly positive.")

    def test_analytical_ground_truth(self):
        """Test analytical ground truth solving and double-check equivalence of formulations."""
        gt = compute_analytical_ground_truth(self.P, self.pi, self.f)
        self.assertAlmostEqual(np.dot(self.pi, gt["V_star"]), gt["V_bar_star"])
        self.assertGreaterEqual(gt["kappa"], 0.0, "Asymptotic variance kappa must be non-negative.")


if __name__ == "__main__":
    unittest.main()
