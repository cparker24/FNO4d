"""
combined_loss_test_mlx.py
=========================
MLX port of combined_loss_test.py — same analytical invariants, MLX arrays.

Run with:  python combined_loss_test_mlx.py
"""

import math
import sys

import mlx.core as mx

from neuralop_mlx.training.losses import CombinedTemporalLoss

mx.random.seed(0)

PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = '') -> None:
    global passed, failed
    if condition:
        print(f"  {PASS}  {name}")
        passed += 1
    else:
        print(f"  {FAIL}  {name}  ←  {detail}")
        failed += 1


def make_array(*shape, val=0.0):
    return mx.full(shape, val)


# Shape used throughout: (B=1, C=1, X=4, Y=4, Z=4, T=1)
B, C, X, Y, Z, T = 1, 1, 4, 4, 4, 1

# ---------------------------------------------------------------------------
# 1. Zero loss for identical tensors
# ---------------------------------------------------------------------------
print("\n=== 1. Perfect prediction gives zero ===")

target = mx.random.normal((B, C, X, Y, Z, T))

for mode in ('relative', 'absolute'):
    for preset in ('l2', 'h1', 'grad_focus'):
        fn   = CombinedTemporalLoss(preset=preset, mode=mode)
        loss = fn(target, target).item()
        check(f"preset={preset}, mode={mode}", loss < 1e-10,
              f"got {loss:.2e}")

# ---------------------------------------------------------------------------
# 2. Relative L² — known analytical value
# ---------------------------------------------------------------------------
print("\n=== 2. Relative L² — analytical value ===")

c, eps_val = 3.0, 0.5
target_const = make_array(B, C, X, Y, Z, T, val=c)
pred_const   = make_array(B, C, X, Y, Z, T, val=c + eps_val)
expected_rel_l2 = (eps_val / c) ** 2

fn  = CombinedTemporalLoss(preset='l2', mode='relative')
got = fn(pred_const, target_const).item()
check(
    f"rel_L²: expected {expected_rel_l2:.6f}, got {got:.6f}",
    abs(got - expected_rel_l2) < 1e-5,
    f"error = {abs(got - expected_rel_l2):.2e}"
)

# ---------------------------------------------------------------------------
# 3. Absolute L² — known analytical value
# ---------------------------------------------------------------------------
print("\n=== 3. Absolute L² — analytical value ===")

expected_abs_l2 = eps_val ** 2
fn  = CombinedTemporalLoss(preset='l2', mode='absolute')
got = fn(pred_const, target_const).item()
check(
    f"abs_L²: expected {expected_abs_l2:.6f}, got {got:.6f}",
    abs(got - expected_abs_l2) < 1e-5,
    f"error = {abs(got - expected_abs_l2):.2e}"
)

# ---------------------------------------------------------------------------
# 4. Relative mode is scale-invariant
# ---------------------------------------------------------------------------
print("\n=== 4. Relative mode: scale invariance ===")

pred_r   = mx.random.normal((B, C, X, Y, Z, T))
target_r = mx.random.normal((B, C, X, Y, Z, T))

for preset in ('l2', 'h1', 'grad_focus'):
    fn     = CombinedTemporalLoss(preset=preset, mode='relative')
    loss1  = fn(pred_r,         target_r        ).item()
    loss5  = fn(pred_r * 5,     target_r * 5    ).item()
    loss01 = fn(pred_r * 0.1,   target_r * 0.1  ).item()
    check(
        f"preset={preset}: ×5  unchanged",
        abs(loss5 - loss1) < 1e-4,
        f"loss1={loss1:.5f}, loss5={loss5:.5f}"
    )
    check(
        f"preset={preset}: ×0.1 unchanged",
        abs(loss01 - loss1) < 1e-4,
        f"loss1={loss1:.5f}, loss0.1={loss01:.5f}"
    )

# ---------------------------------------------------------------------------
# 5. Absolute mode scales with error magnitude
# ---------------------------------------------------------------------------
print("\n=== 5. Absolute mode: scales with error magnitude ===")

target_a = mx.random.normal((B, C, X, Y, Z, T))
error    = mx.random.normal((B, C, X, Y, Z, T))

for preset in ('l2', 'h1'):
    fn     = CombinedTemporalLoss(preset=preset, mode='absolute')
    loss1  = fn(target_a + error,     target_a).item()
    loss2  = fn(target_a + 2 * error, target_a).item()
    check(
        f"preset={preset}: 2× error → ~4× loss",
        abs(loss2 / loss1 - 4.0) < 0.01,
        f"ratio = {loss2/loss1:.4f}, expected 4.0"
    )

# ---------------------------------------------------------------------------
# 6. Component isolation: β=0 kills gradient term
# ---------------------------------------------------------------------------
print("\n=== 6. Component isolation ===")

pred_i   = mx.random.normal((B, C, X, Y, Z, T))
target_i = mx.random.normal((B, C, X, Y, Z, T))

fn_l2   = CombinedTemporalLoss(preset='l2',       mode='relative')
fn_grad = CombinedTemporalLoss(preset='boundary',  mode='relative')
fn_h1   = CombinedTemporalLoss(preset='h1',        mode='relative')

d_l2   = fn_l2.forward_dict(pred_i,   target_i)
d_grad = fn_grad.forward_dict(pred_i, target_i)
d_h1   = fn_h1.forward_dict(pred_i,   target_i)

check("β=0 → grad component = 0",  d_l2['grad'].item()  < 1e-10,
      f"got {d_l2['grad'].item():.2e}")
check("α=0 → l2 component = 0",    d_grad['l2'].item()  < 1e-10,
      f"got {d_grad['l2'].item():.2e}")
expected_h1 = d_l2['l2'].item() + d_grad['grad'].item()
check(
    f"h1 total = l2 + grad  ({expected_h1:.5f})",
    abs(d_h1['total'].item() - expected_h1) < 1e-5,
    f"got {d_h1['total'].item():.5f}"
)

# ---------------------------------------------------------------------------
# 7. Temporal weight isolation
# ---------------------------------------------------------------------------
print("\n=== 7. Temporal weight isolation ===")

T_multi  = 8
pred_t   = mx.random.normal((B, C, X, Y, Z, T_multi))
target_t = mx.random.normal((B, C, X, Y, Z, T_multi))

for t_focus in (0, 3, 7):
    # MLX arrays are built from lists; item assignment is not supported.
    w = mx.array([1.0 if i == t_focus else 0.0 for i in range(T_multi)])

    fn_weighted = CombinedTemporalLoss(preset='h1', mode='relative', tau_weights=w)
    fn_single   = CombinedTemporalLoss(preset='h1', mode='relative')

    loss_weighted = fn_weighted(pred_t, target_t).item()
    loss_single   = fn_single(
        pred_t[..., t_focus:t_focus + 1],
        target_t[..., t_focus:t_focus + 1],
    ).item()

    check(
        f"weight at t={t_focus} only == single-slice loss",
        abs(loss_weighted - loss_single) < 1e-5,
        f"weighted={loss_weighted:.6f}, single={loss_single:.6f}"
    )

# ---------------------------------------------------------------------------
# 8. Monotonicity — larger errors produce larger loss
# ---------------------------------------------------------------------------
print("\n=== 8. Monotonicity ===")

target_m = mx.zeros((B, C, X, Y, Z, T))
errors   = [0.01, 0.1, 0.5, 1.0, 2.0]

for mode in ('relative', 'absolute'):
    fn     = CombinedTemporalLoss(preset='h1', mode=mode)
    losses = [fn(target_m + e, target_m).item() for e in errors]
    is_increasing = all(losses[i] < losses[i + 1] for i in range(len(losses) - 1))
    check(
        f"mode={mode}: losses strictly increase with error magnitude",
        is_increasing,
        f"losses={[f'{v:.4f}' for v in losses]}"
    )

# ---------------------------------------------------------------------------
# 9. Gradient flow through both components
# ---------------------------------------------------------------------------
# MLX uses mx.grad() instead of .backward(); the gradient is the return value.
print("\n=== 9. Gradient flow ===")

for mode in ('relative', 'absolute'):
    p      = mx.random.normal((B, C, X, Y, Z, T))
    target_g = mx.random.normal((B, C, X, Y, Z, T))
    fn     = CombinedTemporalLoss(preset='h1', mode=mode)

    grad = mx.grad(lambda p_: fn(p_, target_g))(p)
    mx.eval(grad)

    no_nan    = not mx.isnan(grad).any().item()
    grad_norm = mx.sqrt(mx.sum(grad ** 2)).item()
    no_zero   = grad_norm > 1e-10
    check(f"mode={mode}: no NaN gradients",  no_nan)
    check(f"mode={mode}: non-zero gradients", no_zero,
          f"norm={grad_norm:.2e}")

# ---------------------------------------------------------------------------
# 10. Mixed mode: per-component l2_mode / grad_mode
# ---------------------------------------------------------------------------
print("\n=== 10. Mixed mode (l2=relative, grad=absolute) ===")

pred_mx   = mx.random.normal((B, C, X, Y, Z, T))
target_mx = mx.random.normal((B, C, X, Y, Z, T))
k = 5.0

fn_mix = CombinedTemporalLoss(preset='h1', l2_mode='relative', grad_mode='absolute')
d1 = fn_mix.forward_dict(pred_mx,         target_mx)
dk = fn_mix.forward_dict(pred_mx * k,     target_mx * k)

check(
    "mixed: l2 component unchanged under scale",
    abs(d1['l2'].item() - dk['l2'].item()) < 1e-4,
    f"d1_l2={d1['l2'].item():.5f}, dk_l2={dk['l2'].item():.5f}"
)
expected_ratio = k ** 2
actual_ratio   = dk['grad'].item() / (d1['grad'].item() + 1e-12)
check(
    f"mixed: grad component scales as k²={expected_ratio:.1f}",
    abs(actual_ratio - expected_ratio) < 0.1,
    f"actual ratio={actual_ratio:.3f}"
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*40}")
print(f"  {passed} passed    {failed} failed")
print(f"{'='*40}\n")
sys.exit(0 if failed == 0 else 1)
