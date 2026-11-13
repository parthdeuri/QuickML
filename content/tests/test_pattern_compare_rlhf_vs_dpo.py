import numpy as np
import yaml
d = yaml.safe_load(open("content/patterns/compare-rlhf-vs-dpo.yaml"))
ns = {}
exec(d["template"], ns)
assert abs(ns["at_ref"] - np.log(2)) < 1e-12
assert ns["now"] < ns["at_ref"]
# DPO equals the Bradley-Terry loss on the implicit rewards
imp = ns["implicit"]
bt = ns["bt_loss"](imp(ns["pi_w"], ns["ref_w"]), imp(ns["pi_l"], ns["ref_l"]))
assert abs(bt - ns["now"]) < 1e-12
# RLHF: drifting from the reference lowers the penalised objective
obj = ns["rlhf_objective"]
assert obj(1.0, -8.0, -10.0, 0.1) < obj(1.0, -10.0, -10.0, 0.1)
print("pattern compare-rlhf-vs-dpo OK")
