import yaml
d = yaml.safe_load(open("content/patterns/compare-tensor-vs-pipeline-parallelism.yaml", encoding="utf-8"))
ns = {}
exec(d["template"], ns)
b, s, h = 1, 2048, 4096
act = b * s * h * 2
assert ns["tp_allreduce_bytes"](b, s, h, 1) == 0
assert abs(ns["tp_layer_bytes"](b, s, h, 8) - 2 * 2 * 7 / 8 * act) < 1e-6
assert ns["tp_layer_bytes"](b, s, h, 8) > ns["pp_boundary_bytes"](b, s, h)
assert ns["pp_boundary_bytes"](b, s, h) == act
assert abs(ns["bubble_fraction"](4, 4) - 3 / 7) < 1e-12
assert ns["bubble_fraction"](1, 8) == 0
assert ns["bubble_fraction"](4, 32) < ns["bubble_fraction"](4, 4)
print("pattern compare-tensor-vs-pipeline-parallelism OK")
