# PLAN_learned_norm.md — Learned Per-Event Envelope (Amortized Normalization)

Status: design finalized, implementation not started.
Depends on: NormFn3d.py (superseded in part), causal_loglo_4d_v2.py,
gubser_bjorken_harness.py, trainer_adapter.py, combined_causal_loglo_loss.py.
Related: STAGED_PLAN.md Stages 1 and 4; conservation projection layer; FV hybrid.

---

## 1. What it is

A learned, per-event, analytic multiplicative envelope that replaces the
NormFn3d trend-removal role and lives **inside** the model. The physical field
is factorized as

    u(τ, x, y, η) = M_φ(τ, η) ⊙ ũ(τ, x, y, η)

where a tiny encoder reads the τ₀ slice and outputs O(10) coefficients of a
smooth, separable log-envelope

    log M = − Σ_j a_j φ_j(ln τ/τ₀) + Σ_k b_k ψ_k(η)

with **fixed** smooth bases φ_j, ψ_k. The trunk (any variant) operates on the
quasi-stationary residual ũ; the output is remodulated before losses and
delivery.

**Mechanism.** The Fourier layers implicitly assume stationary, periodic
signals (Li et al., arXiv:2010.08895); a non-periodic secular power law along τ
leaks across all τ modes. Dividing out a per-event, learned mean profile makes
the residual quasi-stationary while keeping the τ/η dependence inside the
differentiable graph — it is learned in an interpretable, low-dimensional,
exactly invertible form rather than stripped by preprocessing.

**Relation to TC-QGP AdaGN.** Structured, physics-constrained special case of
FiLM machinery, applied at the input/output boundary: modulation is restricted
to a smooth, separable, invertible function of (τ, η) on the *physical field*,
not free per-channel gains on hidden features. TC-QGP's stable training at the
250-event / IP-Glasma+MUSIC scale is transferable evidence that the
encoder→modulation mechanism trains in this regime; the goal here differs
(stationarize the residual vs. represent time in a one-shot decoder).

---

## 2. Fixed design decisions (from design discussion)

- **D1 — Frozen at τ₀.** The encoder reads the true initial slice once; the
  coefficients define a fixed, invertible change of variables for the whole
  trajectory. Never re-encode from predicted states during rollout (avoids
  error feedback, preserves the semigroup structure in ũ).
- **D2 — Rollout role: integrating factor.** In autoregressive prediction the
  physical step factorizes as the exact analytic ratio
  M(τ_{n+1})/M(τ_n) times the learned map on ũ. The mean cooling is applied
  exactly per step; network errors accumulate only on the O(1) residual.
  Per-step trend bias no longer compounds geometrically over the rollout.
- **D3 — Channel scope.** Apply only to channels with a genuine secular
  profile: energy density first; channel-specific exponents where warranted
  later. Flow velocities and π^{μν} stay entirely with the operator (TC-QGP's
  per-field error table is the cautionary evidence for global gains on flow).
- **D4 — Keep the static pointwise transform.** e^{3/4} (or log) is retained
  as parameter-free dynamic-range conditioning; the envelope removes the
  coordinate trend, not the core–edge contrast in the transverse plane.
- **D5 — Embeddings stay mandatory.** The envelope removes the first moment
  only; fluctuation dynamics remain non-autonomous in τ and η. The better the
  envelope works, the more the trunk needs the explicit coordinate signal
  (arXiv:2512.01421; same argument as for NormFn3d).
- **D6 — Continuous in (τ, η).** M is analytic, evaluable at arbitrary
  coordinates → preserves discretization invariance (unlike the (n_η, n_τ)
  ref/shape_ref tables), compatible with log-τ, MLP(τ)/CKConv conditioning,
  train-coarse/evaluate-fine curricula, and mild τ extrapolation.
- **D7 — Per-event encoding = centrality/system generalization.** The τ₀
  slice already carries centrality (total energy, size, eccentricity), so the
  coefficients become smooth implicit functions of centrality and system
  without labels or binning — including pAu/dAu, where the fixed τ^{4/3}
  exponent is the known failure mode.
- **D8 — a(τ), not constant a.** 2–3 basis coefficients in ln τ so the
  encoder can anticipate the centrality-dependent steepening of the cooling
  law once transverse expansion sets in.
- **D9 — One owner for the mean dynamics per architecture.** Envelope for the
  pure spectral models; exact Milne geometric sources for the FV hybrid
  (where the envelope's role shrinks to input conditioning of the closure
  network). Never both on the same channel.
- **D10 — Do not extend into transverse structure.** No radii/eccentricity
  terms in M; that is fluctuating physics the FNO must own. The envelope's
  job is variance reduction, not exactness.
- **D11 — Envelope inside the model boundary.** forward(u₀) encodes,
  demodulates, runs the trunk, remodulates: external contract stays
  physical-in / physical-out. All ref/shape_ref .npy files and external inv_*
  calls disappear; the encoder travels in the state_dict, so checkpoints are
  self-contained (removes the stale-ref silent-wrong-answer failure mode).
- **D12 — Masks defined on physical fields.** FireballMask / crop thresholds
  computed before demodulation or after remodulation, never on ũ, so the mask
  definition does not depend on envelope parameters mid-training.

---

## 3. What it replaces and what it does not

Replaced (redundant once trained):
- etau75 τ-power **as trend removal** (power kept per D4 as range conditioning)
- cosh_etau75 analytic η correction
- etau75_eta / etau75_eta_shape data-driven ref and shape_ref tables,
  including their computation from the training split, versioning, and
  inference-time matching

Unchanged and still required:
- sinusoidal τ/η embeddings (D5)
- FireballMask + static conservative crop (orthogonal; D12)
- relative L² / H¹ / spectral loss machinery
- option 2 (physics loss) remains a complement, not a substitute — it lives in
  the loss and removes nothing from the forward pass. Synergy: with the
  envelope inside the graph the output is already physical, so the PDE/ODE
  residual is computed without back-propagating through an external inverse
  normalization.

---

## 4. Implementation steps

### Step 1 — EnvelopeNet module (new file `envelope_net.py`, ~150 lines)

- Bases: φ_j = low-order polynomials in ln(τ/τ₀), 2–3 terms (D8);
  ψ_k = smooth even functions of η, optional odd terms for asymmetric systems.
- Encoder: small strided CNN on the τ₀ energy-density slice → global average
  pool → MLP → coefficients (a_j, b_k).
- **Critical initialization:** zero-init final-layer weights, bias set to
  (a₁ = 1, rest 0), so that in the e^{3/4} variable the envelope starts as
  exactly (τ/τ₀)^{−1} — the model at initialization *is* the current etau75
  pipeline; training learns only departures.
- API: `coeffs = encode(u0)`; `M = evaluate(coeffs, tau, eta)` for arbitrary
  coordinate tensors (D6); exact `demodulate`/`remodulate` pair.

### Step 2 — Model integration

- Wrap the trunk:
  `forward = encode → pointwise transform → demodulate → trunk → remodulate →
  inverse pointwise transform`.
- Because remodulation is inside `forward`, the neuralop 2.0 Trainer contract
  and `trainer_adapter.py` are untouched; losses receive physical-trend fields
  as now; FireballMask / relative norms unchanged.
- Additionally expose `encode` / `step` / `decode` for the rollout driver
  (Step 5).

### Step 3 — Gates before training (extend `gubser_bjorken_harness.py`)

- **G-init:** with the zero-init head, the full pipeline reproduces the
  current etau75 path to machine precision. Proves the seam is wired
  correctly before any training.
- **G-Bjorken:** on Bjorken events, learned exponent recovers a₁ → 1
  (effective e-exponent 4/3) and ũ is τ-constant to tolerance. Tests the
  *decomposition*, not just the product.
- **G-Gubser:** b_k → 0 (boost invariance, SO(3)_q ⊗ SO(1,1)); residual
  quasi-stationary.
- **G-invert:** demodulate–remodulate round trip at float precision; evaluate
  M on a refined τ grid against the analytic form (discretization-invariance
  check, D6).

### Step 4 — Bookkeeping sweep (the caveat)

Two variable systems now exist: physical u and residual ũ = u/M. Every
physical-variable touchpoint must include remodulation or machine-precision
gates fail for trivial reasons:
- conservation checks (∫ τ T^ττ between slices)
- Cooper–Frye / T_fo contour extraction
- per-field metrics, banded fRMSE, spectral diagnostics
- analytic self-tests in the harness
Label every diagnostic with its variable system (u vs. ũ); residual-space
diagnostics (Jacobian spectra for the Stage-4 stability target, residual
spectral content) are often *more* informative in ũ but must be marked as
such or cross-experiment comparisons become meaningless. Bjorken gate gains
the second assertion (envelope recovers the exponent) separate from the
product-level check.

### Step 5 — Rollout driver

- Encode M once at τ₀ (D1); iterate entirely in ũ; remodulate only at emitted
  τ values (D2). Not a numerical-safety issue (round trip is exact to float
  precision) but keeps the semigroup in one variable system and avoids
  pointless work.
- Consistency test: one-shot prediction vs. teacher-forced iteration agree at
  shared τ slices.
- Training consistency is trivial: teacher-forced pairs (u_n, u_{n+1})
  demodulated with the same per-event envelope → one encoder gradient path
  per event regardless of rollout length; pushforward-style losses unchanged.

### Step 6 — Ablation ladder (frozen evaluation harness)

    fixed etau75
      → etau75_eta_shape
        → envelope, constant a
          → envelope, a(τ)

on the mixed-centrality set. Headline metrics:
- rollout drift of the volume-averaged cooling curve
- banded fRMSE near the freeze-out region
- per-field errors (watch velocities: the envelope must not degrade them, D3)
Decisive test: held-out centrality classes; then pAu/dAu when available — the
regime where the fixed exponent fails (D7).

### Step 7 — Optional couplings (in order of leverage)

1. **Option-2 loss term:** exact volume-averaged Milne conservation residual
   d/dτ ∫ τ T^ττ = − τ² ∫ T^ηη, computed on remodulated fields. Exact at all
   centralities and times (flux terms are total derivatives vanishing at the
   boundary — no Bjorken assumption). This is the k = 0 soft-loss counterpart
   of the conservation projection layer: loss form during training,
   projection layer at rollout. PINO evidence for data-scarce gains:
   arXiv:2111.03794; regularization argument: arXiv:2512.01421. Mind
   multi-term loss balancing (SoftAdapt / ReLoBRaLo, arXiv:2512.01421).
2. **θ-FiLM reuse:** the same encoder→coefficients pattern later feeds the
   η/s(T), ζ/s(T) FiLM conditioning — shared mechanism, different target
   (hidden features), per the foundation-model path.

---

## 5. Interactions summary

| Component | Interaction |
|---|---|
| Plain FNO / joint 4D FFT | Largest gain: removes the non-periodic τ power law that leaks across all τ modes |
| CausalFNO4D / CausalLogLo | Shared 3D spectral weights see statistically similar slices at every τ — the homogeneity weight-sharing assumes |
| Tucker/TFNO factorization | More uniform operator family across slices → lower effective rank; low-rank regularization bites less |
| LOGLO local branch | Patch-level FFT statistics equalized across τ |
| FireballMask / crop | Orthogonal; M > 0 moves no boundary; thresholds on physical fields (D12) |
| Conservation projection | Applies after remodulation (physical variables); M differentiable and exactly invertible |
| FV hybrid | Ownership rule D9: analytic Milne sources own the mean; envelope only conditions the closure input |
| Pretraining ladder / k_η = 0 lifting | Built-in curriculum: 2+1D teaches a_j (b_k ≡ 0 by boost invariance); 3+1D fine-tunes b_k — cheap data teaches the shared part |
| Geo-FNO deformation | Complementary: coordinates vs. amplitude; define M on computational coordinates to fix ordering |
| Log-τ grid, MLP(τ)/CKConv | Compatible by D6 (continuous evaluation) |

---

## 6. Risks and mitigations

- **Frozen envelope drifts late in rollout** (steepening once transverse
  expansion sets in, centrality-dependent): mitigated by a(τ) (D8); leftover
  weak trend is absorbed by trunk + embeddings — envelope is variance
  reduction, not exactness.
- **Envelope competing with the operator:** prevented by D3 + D10 (channel
  scope, no transverse structure) and by the O(10) coefficient budget.
- **Silent unit mismatches (u vs. ũ):** Step 4 sweep + G-init gate + labeled
  diagnostics.
- **Encoder overfitting at 250 events:** coefficient head is tiny and
  zero-initialized to the known physics; ablation ladder (Step 6) isolates
  its marginal contribution; if unstable, freeze at the analytic init and
  train coefficients last (warm-up via Trainer subclass `on_epoch_start`).

---

## 7. Expected outcomes

- Rollout drift of the mean cooling curve reduced (trend handled by exact
  analytic ratio, not learned multiplicatively per step).
- Ref-table machinery deleted; checkpoints self-contained; discretization
  invariance restored relative to etau75_eta / shape_ref.
- Single model spans centralities and (later) systems; learned a(centrality)
  reproduces known phenomenology, with 1 + c_s² in the ideal boost-invariant
  limit — an interpretable physics check unavailable to AdaGN-style
  conditioning.
