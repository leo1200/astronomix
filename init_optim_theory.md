# Reduced- vs Full-Space Optimization for PDE-Constrained Initial-Value Inverse Problems

*A three-axis theory, with worked examples on Kuramoto–Sivashinsky and 3D compressible Euler turbulence.*

---

## 1. Setup

Recover the initial state (and/or parameters) of a known PDE from data by minimizing a discrete loss. Methods sit on the classical **reduced-space ↔ full-space** axis of PDE-constrained optimization (Biros & Ghattas 2005; Hinze et al. 2009) — distinguished by *how many time-states are free variables*:

- **reduced-space / single shooting** — control = initial condition; the differentiable-solver approach;
- **multiple shooting** — control = `m` segment-start states + continuity (consistency) constraints; the middle rung of the optimal-control transcription ladder (Bock & Plitt 1984; Kelly 2017);
- **full-space / all-at-once** — control = the whole space–time field; the penalty formulation of van Leeuwen & Herrmann (2016), ODIL (Karnakov et al. 2024), PINNs (Raissi et al. 2019).

A *second* axis is **representation**: local grid (ODIL) vs global network (PINN); both are full-space.

A step-defect dynamics residual makes the limits exact: `m = 1` multiple shooting ≡ single shooting, and `m = N_t` ≡ full-space.

---

## 2. Three independent failure axes

Which one binds is problem-dependent.

1. **Conditioning** (local Hessian / gradient). Set by the singular spectrum of the linearized propagator over the horizon, plus operator stiffness. Unstable/chaotic dynamics give exploding gradients (`σ_max ∼ e^{λT}`); dissipative/damped modes give the vanishing/under-determined regime. The growth of the *variational* Hessian's condition number with window length is the 4D-Var conditioning analysis (Gratton et al. 2007; El-Said 2015). **Mild ⇒ slow; severe ⇒ fatal** (overflow/NaN, numerically singular solves, gradients flushed to zero), with the slow↔fatal threshold set by floating-point precision (`~10^16` float64, `~10^7` float32) and by the optimizer (first-order vs affine-invariant Newton).

2. **Identifiability** (global, statistical). Does data + prior determine the solution? Sparse data, non-injective forward maps, and damped/unobserved directions leave it under-determined. Resolved only by more data or a prior — a MAP estimate (data pins the observable/unstable subspace, the prior fills the stable one) or posterior sampling. Optimizer- and method-independent. Bayesian view: Stuart (2010); in DA, 4D-Var corrects growing but not decaying modes (Johnson et al. 2005; Trevisan & Uboldi 2004).

3. **Basin / non-convexity** (global, landscape). From forward-map nonlinearity compounded over the horizon; a **linear** forward map gives a convex loss and *no* basin issue. In DA this is the long-window multiple-minima problem, whose minima multiply with window length and are cured by continuation (quasi-static variational assimilation; Miller et al. 1994; Pires et al. 1996).

---

## 3. Central claim

These three are distinct, and **conditioning is not the failure mode.** A method can be well-conditioned but multimodal, unimodal but numerically singular, or perfectly solvable yet non-unique. Single shooting's condition number is often *lower* than full-space's, yet single shooting is the one that fails past the predictability horizon — because the killer is **multimodality (axis 3)**, not conditioning (axis 1). The exploding gradient is a real first-order obstacle but, on its own, a survivable one (until it becomes numerically fatal, per axis 1). The condition number is the *least* diagnostic of the three.

---

## 4. The method axis — what it attacks

Lifting interior states into free variables attacks conditioning and basin together, via two **separable knobs**:

- **Segmentation** (how many states are freed) removes the exploding gradient *structurally*: freeing the interior states zeroes the slaved-state chain-rule product (`∏ Jacobians ∼ e^{λT}`), leaving per-segment gradients `∼ e^{λT/m}`. This is independent of constraint stiffness, and is the now-standard fix for training through long/chaotic trajectories — with the explicit observation that multiple-shooting models have *simpler loss landscapes* (Turan & Jäschke 2022; Massaroli et al. 2021; Iakovlev et al. 2023).
- **Softness** of the consistency coupling (the penalty weight) expands the search space and smooths the landscape — the weak-constraint / "expand the search space" effect (van Leeuwen & Herrmann 2013, 2016).

Mechanistically, terminal data reaches the initial state **not** through one adjoint but by *relaying* through the consistency constraints — one segment per optimizer sweep, each hop a single well-conditioned segment Jacobian — a boundary-value relaxation. The price is more variables and `~m` relay iterations; the backprop-through-time memory of the reduced approach is itself mitigated by checkpointing (Griewank & Walther 2000).

Full-space is the limit: it removes the *dynamic* conditioning but trades it for a large, **static, preconditionable** conditioning set by the penalty weight × stiffness — addressable by the control-variable transform / multigrid (El-Said 2015; Karnakov et al. 2024) and by the saddle-point, time-parallel formulation (Fisher & Gürol 2017; Daužickaitė et al. 2021). None of these touch axis 2.

---

## 5. Least Squares Shadowing — adjacent, not on the spectrum

LSS targets the sensitivity of *long-time averages* in chaos (Wang et al. 2014; Ni & Wang 2017), where the conventional tangent/adjoint diverges — but structurally it is the *same* all-at-once move, replacing the exploding adjoint with a global least-squares over the whole trajectory. It corroborates the axis-1 fix from a third community.

---

## 6. Unification

One spectrum, four vocabularies:

| Axis | Reduced end | Full end | Community |
|---|---|---|---|
| state-space | single shooting | all-at-once | PDE-constrained optimization (Biros & Ghattas 2005) |
| constraint | strong constraint | weak constraint | 4D-Var (Le Dimet & Talagrand 1986; Trémolet 2006; Fisher et al. 2005) |
| transcription | single shooting | collocation | optimal control (Bock & Plitt 1984; Kelly 2017) |
| representation | differentiable solver | ODIL / PINN | SciML (Raissi et al. 2019; Karnakov et al. 2024) |

No single community states all three failure axes together.

---

## 7. Worked example I — Kuramoto–Sivashinsky (chaotic, demonstrated)

**Problem.** KS at `L = 22`, `N = 64`, `dt = 0.1` (chaotic regime), one-step IMEX (Crank–Nicolson linear + explicit nonlinear); `λ ≈ 0.05`, `T_λ ≈ 20`. The *same* discrete kernel generates the reference and drives single shooting, multiple shooting, and ODIL on a shared step-defect loss, so `m = 1 ≡ SS` and `m = N_t ≡ ODIL` exactly.

**Conditioning (parameter-free, evaluated at the truth — the robust core).** The per-segment sensitivity follows the clean law: `σ_max ∼ e^{λT}` for single shooting, `∼ e^{λT/m}` per segment, monotone in `m`. ODIL is **flat in `T`**; its large absolute level (`~10^13`) is set by penalty × stiffness, not chaos, and is preconditionable. A subtlety: the *realized* strong-constraint GN conditioning is **non-monotone** in `m` — an `m = 2` spike *above* both endpoints, because a stiff penalty couples still-long segments — so the clean `e^{2λT/m}` law belongs to the local per-segment sensitivity, not to the global conditioning.

**Recovery (cold start, fixed observation coverage).** Single shooting reconstructs to ~1% until a *few* Lyapunov times, then **cliffs** (~100×) — a basin loss at a perturbation-dependent horizon `t_cliff ≈ λ⁻¹ ln(1/ε₀)`, *not* at `1 T_λ`. Multiple shooting and ODIL recover across all horizons given a sound init. With a fair init and adequate budget, **ODIL is the most accurate** method but the **most init-sensitive** (cold/climatology starts trap it near `≈1`; it needs homotopy). At fixed budget the recovery error is non-monotone in `m` and init-direction-fragile near the basin boundary, so there is **no intermediate-`m` accuracy sweet spot** — the case for intermediate `m` is cost × robustness (cost is U-shaped in `m`), not lower error.

**What it shows.** The three axes cleanly separate, and the headline paradox dissolves: single shooting's condition number is *lower* than ODIL's, yet single shooting fails — the killer is multimodality (axis 3), the exploding gradient is the first-order symptom (axis 1), and identifiability (axis 2) is held fixed by the observation design.

---

## 8. Worked example II — 3D compressible Euler turbulence (proposed)

**Why large-scale structure.** Pure small-scale turbulence is *unidentifiable* (washed out and exponentially sensitive), so the meaningful target is a large-scale structure that seeds or coexists with turbulence — e.g. the compressible Taylor–Green vortex transitioning to turbulence (clean demonstrator), or a Richtmyer–Meshkov interface whose initial perturbation is inferred from the late mixing layer (application). Make the control the large-scale band (low-`k`, or a few physical parameters, or full IC with a strong smoothness prior) so identifiability (axis 2) is controlled and not confounded with the gradient story.

**Why single shooting fails — and why it contaminates even a large-scale target.** The unstable directions live at the *small* scales, with local rate `∼ 1/τ_η` (the Kolmogorov time), far faster than the large-eddy turnover `T_L`. Single shooting computes the large-scale gradient by propagating the adjoint backward through the *entire* turbulent trajectory, where it grows and piles up at high `k`; that amplified small-scale adjoint leaks into and swamps the large-scale gradient within a fraction of one turnover. This is the regime that diverges the conventional adjoint and motivated shadowing (Wang et al. 2014; Ni & Wang 2017). Multiple shooting caps the backprop to one segment, so the large-scale gradient survives.

**Setup and diagnostics.** Same differentiable shock-capturing scheme for all methods; sparse, filtered late-time observations at fixed coverage; compare single shooting, multiple shooting (free full 3D intermediate states + consistency defects), and full-space. The turbulence-specific diagnostic is the **scale-resolved adjoint energy spectrum vs. time during the backward pass**: in single shooting it migrates to high `k` (visible small-scale blow-up); in multiple shooting it resets at each segment boundary. Pair with the money plot (large-scale recovery error vs `T/T_L`) and a **scale-decomposed recovery error** (low-`k` recovered, high-`k` pinned at the prior level for *every* method — the visual proof that turbulence inversion only makes sense at large scales).

**Predictions.** Single shooting cliffs within `~1` turnover; multiple shooting and full-space recover the large scales across many; small scales are unrecoverable for all methods (axis 2, independent of method). Because the relevant rate is `1/τ_η`, the gradient blows up far faster than in KS, so segments must be short — turbulence pushes hard toward **many segments / full-space** (and the long-window weak-constraint formulations), with a genuine memory U-curve in `m` (multiple shooting stores `m` full 3D fields; single shooting must checkpoint the whole trajectory).

**Caveats.** Differentiating through shock capturing is the practical wall (non-smooth limiters give biased AD gradients at discontinuities — prefer smooth limiters, entropy-stable DG, or a spectral scheme with hyperviscosity). The consistency defects can only close on the large scales, so a **scale-dependent** penalty (strong on large scales, weak on small) is the right softness choice. A clean first cut is the scale-resolved adjoint-spectrum experiment on the Taylor–Green vortex before committing to the full inverse problem.

---

## References

*Verified to primary source.*

- Biros, G. & Ghattas, O. (2005). Parallel Lagrange–Newton–Krylov–Schur methods for PDE-constrained optimization, Part I & II. *SIAM J. Sci. Comput.* 27(2):687–713 & 714–739. doi:10.1137/S106482750241565X, 10.1137/S1064827502415661.
- Carrassi, A., Bocquet, M., Bertino, L. & Evensen, G. (2018). Data assimilation in the geosciences: an overview of methods, issues, and perspectives. *WIREs Climate Change* 9(5):e535. doi:10.1002/wcc.535.
- Daužickaitė, I., Lawless, A. S., Scott, J. A. & van Leeuwen, P. J. (2021). Randomised preconditioning for the forcing formulation of weak-constraint 4D-Var. *QJRMS* 147. arXiv:2101.07249.
- El-Said, A. (2015). *Conditioning of the weak-constraint variational data assimilation problem for numerical weather prediction.* PhD thesis, University of Reading.
- Fisher, M., Leutbecher, M. & Kelly, G. A. (2005). On the equivalence between Kalman smoothing and weak-constraint 4D-Var. *QJRMS* 131(610):3235–3246. doi:10.1256/qj.04.142.
- Fisher, M. & Gürol, S. (2017). Parallelization in the time dimension of 4D-Var. *QJRMS* 143(703):1136–1147.
- Gratton, S., Lawless, A. S. & Nichols, N. K. (2007). Approximate Gauss–Newton methods for nonlinear least squares problems. *SIAM J. Optim.* 18(1):106–132.
- Griewank, A. & Walther, A. (2000). Algorithm 799: revolve — checkpointing for the reverse/adjoint mode of computational differentiation. *ACM Trans. Math. Softw.* 26(1):19–45. doi:10.1145/347837.347846.
- Iakovlev, V., Yildiz, C., Heinonen, M. et al. (2023). Latent neural ODEs with sparse Bayesian multiple shooting. *ICLR 2023.* arXiv:2210.03466.
- Johnson, C., Hoskins, B. J. & Nichols, N. K. (2005). A singular vector perspective of 4D-Var: filtering and interpolation. *QJRMS* 131(605):1–19. doi:10.1256/qj.03.231.
- Karnakov, P., Litvinov, S. & Koumoutsakos, P. (2024). Solving inverse problems in physics by optimizing a discrete loss. *PNAS Nexus* 3(1):pgae005. doi:10.1093/pnasnexus/pgae005. arXiv:2205.04611.
- Kelly, M. P. (2017). An introduction to trajectory optimization: how to do your own direct collocation. *SIAM Review* 59(4):849–904. doi:10.1137/16M1062569.
- Mikhaeil, J. M., Monfared, Z. & Durstewitz, D. (2022). On the difficulty of learning chaotic dynamics with RNNs. *NeurIPS 2022.* arXiv:2110.07238.
- Miller, R. N., Ghil, M. & Gauthiez, F. (1994). Advanced data assimilation in strongly nonlinear dynamical systems. *J. Atmos. Sci.* 51(8):1037–1056.
- Ni, A. & Wang, Q. (2017). Sensitivity analysis on chaotic dynamical systems by non-intrusive least squares shadowing (NILSS). *J. Comput. Phys.* 347:56–77. doi:10.1016/j.jcp.2017.06.033.
- Pires, C., Vautard, R. & Talagrand, O. (1996). On extending the limits of variational assimilation in nonlinear chaotic systems. *Tellus A* 48(1):96–121.
- Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). Physics-informed neural networks. *J. Comput. Phys.* 378:686–707. doi:10.1016/j.jcp.2018.10.045.
- Stuart, A. M. (2010). Inverse problems: a Bayesian perspective. *Acta Numerica* 19:451–559. doi:10.1017/S0962492910000061.
- Trémolet, Y. (2006). Accounting for an imperfect model in 4D-Var. *QJRMS* 132(621):2483–2504. doi:10.1256/qj.05.224.
- Trevisan, A. & Uboldi, F. (2004). Assimilation of standard and targeted observations within the unstable subspace. *J. Atmos. Sci.* 61(1):103–113.
- van Leeuwen, T. & Herrmann, F. J. (2013). Mitigating local minima in full-waveform inversion by expanding the search space. *Geophys. J. Int.* 195(1):661–667. doi:10.1093/gji/ggt258.
- van Leeuwen, T. & Herrmann, F. J. (2016). A penalty method for PDE-constrained optimization in inverse problems. *Inverse Problems* 32(1):015007. doi:10.1088/0266-5611/32/1/015007. arXiv:1504.02249.
- Wang, Q., Hu, R. & Blonigan, P. (2014). Least squares shadowing sensitivity analysis of chaotic limit cycle oscillations. *J. Comput. Phys.* 267:210–224.

*Confirmed via consistent secondary citation (verify page numbers before submission):* Bock, H. G. & Plitt, K. J. (1984), *Proc. 9th IFAC World Congress*; Le Dimet, F.-X. & Talagrand, O. (1986), *Tellus A* 38(2):97–110; Lea, D. J., Allen, M. R. & Haine, T. W. N. (2000), *Tellus A* 52(5):523–532; Massaroli, S., Poli, M. et al. (2021), *NeurIPS 2021*; Turan, E. M. & Jäschke, J. (2022), *IEEE Control Systems Letters* 6:1897–1902.

*Cited from standard knowledge, not re-verified to primary source — confirm before use:* Haber, E. & Ascher, U. M. (2001), *Inverse Problems* 17(6):1847–1864 (all-at-once parameter estimation); textbooks: Hinze, Pinnau, Ulbrich & Ulbrich (2009, Springer); Biegler et al., eds. (2003, Springer); Betts (2010, SIAM); Asch, Bocquet & Nodet (2016, SIAM); Law, Stuart & Zygalakis (2015, Springer).