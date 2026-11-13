import yaml
d = yaml.safe_load(open("content/patterns/compare-mha-vs-gqa-vs-sliding-window.yaml", encoding="utf-8"))
ns = {}
exec(d["template"], ns)
kb, af, aw = ns["kv_bytes"], ns["attended_full"], ns["attended_window"]
assert kb(32, 32, 128, 4096) == 4 * kb(32, 8, 128, 4096)
assert kb(32, 32, 128, 4096) == 2 * 32 * 32 * 128 * 4096 * 2
assert kb(1, 1, 1, 1, 1) == 2
assert aw(100, 100) == af(100) and aw(100, 500) == af(100)
assert aw(4096, 512) < af(4096) / 4
T, W = 1000, 100
assert aw(T, W) == W * (W + 1) // 2 + (T - W) * W
print("pattern compare-mha-vs-gqa-vs-sliding-window OK")
