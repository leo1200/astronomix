## Step 1 — Understand the Sod shock tube

The Sod problem starts with two regions separated at x = 0.5:

```
left:  ρ=1.0, v=0, P=1.0   (high pressure)
right: ρ=0.125, v=0, P=0.1  (low pressure)
```

When released, three structures emerge and move rightward by t=0.2:

```
x:     0        0.2       0.5      0.65      0.87      1.0
       |---------|---------|--------|---------|---------|
S:     1.0      1.0→      1.0      3.4       3.2       3.2
                rarefaction        ^         ^
                (isentropic,       contact   shock
                S stays flat)      disc      (small
                                   (large    jump)
                                   jump,
                                   initial
                                   conditions)
```

```
x=0          0.2-0.5        0.67          0.87       1.0
|____________|______________|_____________|___________|
  undisturbed   rarefaction   contact disc   SHOCK    undisturbed
  left state    wave          (density jump) front    right state
```

Each structure has a distinct physical signature. Knowing where each one sits is your ground truth for evaluating the plots.

---

## Step 2 — Reading Image 1 (fluid state)

Go panel by panel and ask: **does this match what physics predicts for each structure?**

**Density (top left)**

```
x < 0.2  : ρ ≈ 1.0   → undisturbed left state           ✓
0.2-0.5  : ρ falling → rarefaction fan, smooth decrease  ✓
0.5-0.67 : ρ ≈ 0.47  → post-rarefaction plateau         ✓
0.67     : ρ drops   → contact discontinuity             ✓
0.67-0.87: ρ ≈ 0.22  → post-shock compressed gas        ✓
x > 0.87 : ρ ≈ 0.125 → undisturbed right state          ✓
```

The contact discontinuity at x≈0.67 is where density jumps but pressure does NOT jump — two fluids of different entropy in pressure equilibrium. The shock at x≈0.87 is where both density AND pressure jump.

**Velocity (top right)**

```
x < 0.2  : v = 0     → undisturbed, not yet reached      ✓
0.2-0.67 : v rising  → rarefaction accelerates gas       ✓
0.67-0.87: v ≈ 0.85  → uniform post-shock velocity       ✓
x > 0.87 : v = 0     → undisturbed right state           ✓
```

Key physical fact: velocity is **continuous across the contact discontinuity** (both sides move at the same speed) but **jumps sharply at the shock**. You can see both in the plot.

**Entropy (bottom left)**

```
x < 0.67 : S ≈ 1.0   → adiabatic process, no entropy change  ✓
x ≈ 0.67 : S jumps   → contact discontinuity, two fluids      ✓
0.67-0.87: S ≈ 3.4   → post-shock entropy rise               ✓
x > 0.87 : S ≈ 3.2   → right state entropy                   ✓
```

Key physical fact: **entropy only rises at shocks** (irreversible process). The rarefaction is isentropic (reversible), so entropy stays flat through it. If entropy rose in the rarefaction region that would be a serious bug. It does not — correct.

**Pressure (bottom right)**

```
x < 0.5  : P ≈ 1.0   → left plateau                    ✓
0.2-0.5  : P falling → rarefaction, smooth drop         ✓
0.5-0.87 : P ≈ 0.30  → post-shock pressure plateau      ✓
x > 0.87 : P ≈ 0.125 → undisturbed right state         ✓
```

Key physical fact: **pressure is continuous across the contact discontinuity** — only density jumps there, not pressure. You can verify: the pressure curve shows no feature at x≈0.67, only at x≈0.87. Correct.

**Red dashed line at x≈0.87**

This is where the shock finder placed the surface cell. It sits exactly at the sharp joint between the post-shock plateau and the undisturbed right state — correct in all four panels simultaneously. If it were misplaced onto the rarefaction or contact discontinuity that would be wrong.

---

## Step 3 — Reading Image 2 (the three criteria)

Now for each criterion ask: **where should it fire given what we just learned about the flow structure?**

**What you expect physically:**

| Region | ∇·v | ∇T·∇ρ | M > 1.3 |
|---|---|---|---|
| Undisturbed left | 0 | 0 | No |
| Rarefaction | negative (expanding) | depends | No |
| Contact disc x≈0.67 | ≈0 | could fire | No |
| Shock x≈0.87 | strongly negative | Yes | Yes |
| Undisturbed right | 0 | 0 | No |

**Criterion 1 — ∇·v < 0 (orange)**

Should fire: at the shock (strong compression) and possibly weakly at the rarefaction tail (gas is still accelerating, so some convergence).

Should NOT fire: broadly across the entire middle region with rapid oscillations.

What you see: rapid oscillation from x≈0.45 to x≈0.87. This is numerical noise — the HLLC solver leaves small velocity oscillations behind the rarefaction that the central difference picks up and amplifies. Each cell flips sign relative to its neighbor. **This is a red flag** — not physically wrong enough to break the final result (criterion 3 saves it) but indicates the gradient stencil is too sensitive.

**Criterion 2 — ∇T·∇ρ > 0 (green)**

Should fire: at the shock (both T and ρ jump in the same direction) and possibly at the contact discontinuity (ρ jumps but T may too slightly).

Should NOT fire: in the rarefaction interior, or oscillating rapidly.

What you see: fires at x≈0.2 (start of rarefaction — false positive), oscillates in the middle, then correctly fires near x≈0.87. The rarefaction hit is physically explainable — at the rarefaction head, both ρ and T start decreasing together so the dot product is positive. Technically not a shock but the criterion cannot distinguish it alone. Again criterion 3 saves it.

**Criterion 3 — M > 1.3 (red)**

Should fire: only at the shock, nowhere else. The rarefaction and contact discontinuity are subsonic processes. The Mach number is only super-1.3 at an actual shock.

What you see: fires at exactly one location near x≈0.87, zero everywhere else. **This is correct and clean.** This criterion is doing all the heavy lifting — it alone correctly localizes the shock.

---

## Step 4 — The key diagnostic question

When evaluating any shock finder output, ask these four questions in order:

**Q1: Does the shock surface land on the sharp discontinuity?**
Look at velocity and pressure — the shock is where both have a sharp jump (not a smooth ramp). The red line should sit there. ✓ here.

**Q2: Is the shock zone 3–4 cells wide, centered on the surface?**
The zone should be a compact band. If it spans 50 cells something is wrong with criterion 1 or 2. Here the oscillating criteria make this hard to judge from the individual panels, but the AND combination fixes it.

**Q3: Is the Mach number physically reasonable?**
For the Sod problem the exact solution gives M ≈ 1.75 at the shock. Your finder should return something close to that. Check `new_result.mach_numbers[new_result.shock_surface_cells]` against this.

**Q4: Are there false positives?**
Are any surface cells marked at the rarefaction (x≈0.2–0.5) or contact discontinuity (x≈0.67)? Those would be wrong. From Image 1 there is only one red line at x≈0.87 — no false positives. ✓

---

## Summary judgment

| | Correct? | Why |
|---|---|---|
| Shock location | ✓ | Sits on the sharp discontinuity |
| No false positives | ✓ | Only one surface cell |
| Criterion 3 | ✓ | Clean, fires only at shock |
| Criterion 1 | ⚠ | Noisy, but saved by criterion 3 |
| Criterion 2 | ⚠ | False positive at rarefaction head, saved by criterion 3 |
| Mach number | needs checking | Compare to exact M≈1.75 |

The final result is correct. The individual criteria are noisier than they should be, which is a robustness concern — if you ever run a case where criterion 3 is borderline (M just above 1.3), the noise in criteria 1 and 2 could cause false negatives at the real shock or false positives nearby.