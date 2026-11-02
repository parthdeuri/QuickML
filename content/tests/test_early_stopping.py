import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "early-stopping.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
es = ns["early_stop"]
rng = np.random.default_rng(0)
t = np.arange(60)
# U-shaped validation curve with noise: falls, then overfits
losses = 1.0 / (1 + 0.3 * t) + 0.004 * (t - 15).clip(0) ** 1.3 + rng.normal(0, 0.01, 60)


def expected(losses, patience):
    """Independent derivation (min_delta = 0): improvement epochs are the
    strict prefix minima; stop at the first epoch whose distance to the most
    recent improvement reaches patience."""
    L = np.asarray(losses)
    prefix_min = np.minimum.accumulate(L)
    improved = np.r_[True, L[1:] < prefix_min[:-1]]
    last = np.maximum.accumulate(np.where(improved, np.arange(len(L)), -1))
    hit = np.flatnonzero(np.arange(len(L)) - last >= patience)
    stop = int(hit[0]) if len(hit) else len(L) - 1
    return stop, int(last[stop]), float(L[last[stop]])


for patience in (1, 2, 3, 5, 8, 20, 100):
    stop, best_epoch, best = es(losses, patience)
    e_stop, e_best, e_val = expected(losses, patience)
    assert (stop, best_epoch) == (e_stop, e_best), patience
    assert np.isclose(best, e_val) and np.isclose(best, losses[best_epoch])
    if stop < len(losses) - 1 or patience <= stop - best_epoch:
        assert stop - best_epoch >= patience or stop == len(losses) - 1

# fixed sequence with simple structure
seq = [5, 4, 3, 3.5, 3.4, 3.6, 2.9, 3.0]
assert expected(seq, 3) == (5, 2, 3.0) and es(seq, 3) == (5, 2, 3.0)
assert es(seq, 4)[:2] == (7, 6)           # improvement at epoch 6 resets the counter
assert es([3, 2, 1], 5) == (2, 2, 1)      # never triggers: runs to the end
# min_delta ignores tiny gains: epoch 2 improves by less than min_delta
stop, be, b = es([1.0, 0.5, 0.499, 0.498, 0.497], 2, min_delta=0.01)
assert (stop, be) == (3, 1) and b == 0.5
# best loss at the returned epoch is never worse than any earlier loss
for patience in (2, 4):
    stop, be, b = es(losses, patience)
    assert b <= min(losses[: stop + 1]) + 1e-12
print("early-stopping OK")
