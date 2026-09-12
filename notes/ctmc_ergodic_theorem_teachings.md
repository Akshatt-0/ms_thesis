# Ergodic Theorem for Continuous-Time Markov Chains (CTMC)

This document synthesizes the foundational concepts, mathematical settings, norms, and conditions regarding the Ergodic Theorem for Continuous-Time Markov Chains (CTMCs).

---

## 1. Core Statement of the Ergodic Theorem

The ergodic theorem for a CTMC bridges the gap between a time average (long-run behavior of a single trajectory) and an ensemble average (spatial expectation with respect to the stationary distribution).

Let $(X_t)_{t \ge 0}$ be an irreducible, positive-recurrent continuous-time Markov chain on a countable state space $S$ with a unique stationary distribution $\pi = (\pi_i)_{i \in S}$.

For any function $f: S \to \mathbb{R}$ that is either bounded, non-negative, or integrable with respect to $\pi$, the time average converges almost surely to the spatial average:

$$\lim_{T \to \infty} \frac{1}{T} \int_0^T f(X_t) \, dt = \sum_{i \in S} f(i) \pi_i \quad \text{with probability 1}$$

### The Indicator Special Case
For the indicator function $f(X_t) = \mathbb{I}_{\{X_t = i\}}$, the theorem dictates that the long-run proportion of time spent in a specific state $i$, denoted $R_i(T)$, exactly matches its stationary probability $\pi_i$:

$$\lim_{T \to \infty} R_i(T) = \lim_{T \to \infty} \frac{1}{T} \int_0^T \mathbb{I}_{\{X_t = i\}} \, dt = \pi_i = \frac{1}{q_i \mathbb{E}[T_i]}$$

Where:
* $\mathbb{E}[T_i]$ is the expected return time to state $i$.
* $q_i$ is the total transition rate out of state $i$.

---

## 2. Necessary Chain Conditions

To guarantee that the unique stationary distribution exists and that the limit converges, the CTMC must satisfy two primary structural properties:

* **Irreducibility:** Every state in the state space $S$ can communicate with every other state (it is possible to reach any state from any other state in finite time).
* **Positive Recurrence:** The expected return time to any state is finite ($\mathbb{E}[T_i] < \infty$). This ensures the probabilities do not leak away to infinity (as in transience) or flatten out to zero over infinite states (as in null recurrence).

---

## 3. Function Criteria: Bounded vs. Non-Negative

The theorem adapts to different function properties to ensure the limit and integrations are well-behaved:

| Condition Type | Mathematical Context | Operational Impact |
| :--- | :--- | :--- |
| **Non-negative** | $f(i) \ge 0$ for all $i \in S$ | The integrals and infinite sums are always well-defined (even if they equal $+\infty$). If either side is finite, both are finite and equal. |
| **Bounded** | $|f(i)| \le M$ for all $i \in S$ | Boundedness prevents sample path trajectories from oscillating wildly or escaping to infinity, guaranteeing finite limits. |

### General Alternative: $L^1(\pi)$ Integrability
If a function is neither strictly non-negative nor uniformly bounded, the theorem still holds as long as the function is **integrable** with respect to the stationary distribution $\pi$. This requires a finite $L^1(\pi)$ norm:
$$\|f\|_{L^1(\pi)} = \sum_{i \in S} |f(i)| \pi_i < \infty$$

---

## 4. The Role of the Boundedness Norm ($M$)

When a function is stated to be bounded by a real number $M$ ($|f(i)| \le M$), it utilizes the **Supremum Norm** (also known as the $L^\infty$ norm or uniform norm):

$$\|f\|_\infty = \sup_{i \in S} |f(i)|$$

### Why the Supremum Norm is Used
1. **Uniform Control:** It places a global ceiling on the function's evaluation. No matter which state $X_t$ the Markov chain transitions into, $f(X_t)$ can never exceed $M$.
2. **Dominated Convergence:** Having $|f(X_t)| \le M$ uniformly over time allows the use of Lebesgue's Dominated Convergence Theorem, validating the interchange of limits and integrals during mathematical proofs.
3. **Finite Spaces Exception:** If the state space $S$ is finite, **every** real-valued function $f$ automatically yields a finite supremum norm ($\|f\|_\infty < \infty$).

---

## 5. Continuity and Topology

**Continuity is not required** for the function $f$ in the Ergodic Theorem for CTMCs. 

* **Discrete Topology:** Because the state space $S$ is countable and discrete, every function defined on it is automatically continuous by definition. Explicitly demanding continuity is redundant.
* **Discontinuous Sample Paths:** While time $t$ is continuous, the state trajectory $X_t$ changes via sudden, discrete jumps. The composition $f(X_t)$ forms a piecewise constant step function that is explicitly discontinuous at every transition point. 
* **Integration Legality:** Because the chain experiences at most a countable number of jumps in any finite window $[0, T]$, the integral $\int_0^T f(X_t)\,dt$ remains perfectly Riemann or Lebesgue integrable.

The true underlying measure-theoretic prerequisite is **Borel measurability**, which is trivially satisfied by all functions on a countable discrete state space.