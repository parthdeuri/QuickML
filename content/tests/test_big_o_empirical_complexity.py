import bisect, numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "big-o-empirical-complexity.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
slope = ns["fit_loglog_slope"]

rng = np.random.default_rng(0)
sizes = [256, 512, 1024, 2048, 4096]
data = {n: [int(v) for v in rng.integers(0, 10**6, n)] for n in sizes}

# correctness against Python builtins
for n in (0, 1, 2, 7, 100):
    a = [int(v) for v in rng.integers(0, 50, n)]
    assert ns["insertion_sort"](a)[0] == sorted(a)
    assert ns["merge_sort"](a)[0] == sorted(a)
    s = sorted(a)
    for x in (-1, 0, 13, 25, 49, 60):
        i, _ = ns["binary_search"](s, x)
        j = bisect.bisect_left(s, x)
        assert (i == -1) == (j == len(s) or s[j] != x)
        if i != -1:
            assert s[i] == x and i == j
        k, _ = ns["linear_search"](a, x)
        assert (k == -1) == (x not in a)
        if k != -1:
            assert a[k] == x

# costs at growing n (worst case search: missing key)
lin = [ns["linear_search"](data[n], -1)[1] for n in sizes]
binc = [ns["binary_search"](sorted(data[n]), -1)[1] for n in sizes]
ins = [ns["insertion_sort"](data[n])[1] for n in sizes]
mer = [ns["merge_sort"](data[n])[1] for n in sizes]

assert lin == sizes                                   # exactly n comparisons
s_lin, s_ins, s_mer, s_bin = (slope(sizes, c) for c in (lin, ins, mer, binc))
assert abs(s_lin - 1.0) < 0.01, s_lin                 # O(n)
assert abs(s_ins - 2.0) < 0.05, s_ins                 # O(n^2)
assert 1.05 < s_mer < 1.25, s_mer                     # O(n log n)
assert s_bin < 0.2, s_bin                             # O(log n) ~ flat
assert s_lin < s_mer < s_ins

# binary search cost / log2 n is roughly constant, and under the bound
ratios = [c / np.log2(n) for c, n in zip(binc, sizes)]
assert max(ratios) / min(ratios) < 1.1, ratios
assert all(c <= int(np.ceil(np.log2(n + 1))) for c, n in zip(binc, sizes))
# merge sort never exceeds n * ceil(log2 n) comparisons
assert all(c <= n * int(np.ceil(np.log2(n))) for c, n in zip(mer, sizes))
# insertion sort on random data is about n^2 / 4
assert all(abs(c / (n * n / 4) - 1) < 0.1 for c, n in zip(ins, sizes))
# determinism
assert ns["merge_sort"](data[256])[1] == mer[0]
print(f"slopes lin={s_lin:.3f} ins={s_ins:.3f} merge={s_mer:.3f} bin={s_bin:.3f}")
print("big-o-empirical-complexity OK")
