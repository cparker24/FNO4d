"""Wraps a neuralop data loss (LpLoss/H1Loss) to stop penalizing/rewarding
predictions in each event's zero-padded tail.

read_3d_root.py pads every event's tau axis out to a common `choose_ntau`
with exact zeros past that event's own `ntau_freezeout` (events have
different physical lifetimes). Training against that padding with an
unmasked relative loss teaches the model a single dataset-average "shutoff
tau" rather than each event's own freeze-out time -- see
PARTICLIZATION_INVESTIGATION.md's model-vs-ground-truth diagnostic: events
whose true freeze-out falls later than that learned average get their
predicted energy density collapsed to near-zero *before* real freeze-out,
which is the dominant driver of the live-rollout yield deficit.

The fix: zero out *both* prediction and target at every (event, tau) slice
that is pure padding before computing the loss. Padding is exact zero
across every channel and every spatial point (read_3d_root.py's
`np.zeros(...)` fill) while real physics -- even right at the freeze-out
threshold -- has nonzero temperature/velocity content, so a tiny epsilon
cleanly separates the two. Zeroing both sides (not just excluding them from
a sum) makes diff = pred_masked - true_masked = 0 there regardless of the
model's raw output, so no gradient flows from the padded region at all.
"""
import torch


class TailMaskedLoss:
    """Wraps a neuralop loss so padded (event, tau) slices are excluded.

    Parameters
    ----------
    base_loss : callable
        A neuralop-style loss, e.g. LpLoss(d=4, p=2) or H1Loss(d=4). Called
        as ``base_loss(y_pred, y, **kwargs)``.
    eps : float
        Threshold (in the loss's own, normalized-space units) below which a
        (event, tau) slice's max absolute value across channels and all
        spatial points is treated as padding, not real (dilute) physics.
    """

    def __init__(self, base_loss, eps: float = 1e-8):
        self.base_loss = base_loss
        self.eps = eps

    @property
    def name(self):
        base_name = getattr(self.base_loss, 'name', self.base_loss.__class__.__name__)
        return f'TailMasked_{base_name}'

    def _mask(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, C, nx, ny, neta, T) -> mask (B, 1, 1, 1, 1, T), broadcastable
        # over both y_pred and y.
        valid = y.detach().abs().amax(dim=(1, 2, 3, 4), keepdim=True) > self.eps
        return valid.to(y.dtype)

    def __call__(self, y_pred, y, **kwargs):
        mask = self._mask(y)
        return self.base_loss(y_pred * mask, y * mask, **kwargs)
