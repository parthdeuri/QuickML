import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-domain-finetune.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
prepare, format_example = ns["prepare"], ns["format_example"]
check_config, release_gate = ns["check_config"], ns["release_gate"]

recs = [{"prompt": f"q{i}", "response": f"answer number {i} here",
         "group": f"g{i % 8}"} for i in range(16)]
recs.append(dict(recs[0], response="Answer number 0 here!"))  # near-exact dup
recs.append({"prompt": "a", "response": "no", "group": "g1"})  # too short
tr, va = prepare(recs)
assert len(tr) + len(va) == 16 and tr and va
assert not {r["group"] for r in tr} & {r["group"] for r in va}
assert prepare(recs) == (tr, va)  # deterministic

toks, mask = format_example("hi there", "go now")
assert toks == ["<user>", "hi", "there", "<assistant>", "go", "now", "<end>"]
assert mask == [0, 0, 0, 0, 1, 1, 1]
assert sum(mask) == 3 and len(mask) == len(toks)

cfg = dict(epochs=3, n=1000, batch_size=32, warmup_frac=0.03, params=7e9,
           method="full", lora_frac=0.01, act_gb=6, gpu_gb=24)
full = check_config(cfg)
assert full["steps"] == 3 * 32 == 96 and full["warmup"] == 3
assert full["mem_gb"] == 118.0 and not full["fits"]  # 7e9*16/1e9 + 6
assert "118.0" in full["reason"] and "24" in full["reason"]
lora = check_config(dict(cfg, method="lora"))
assert lora["fits"] and lora["mem_gb"] == 21.1  # 14 + 1.12 + 6
assert "reason" not in lora

good = dict(ft_loss=1.1, base_loss=1.5, wins=30, losses=10, ties=10,
            gen_base=0.70, gen_ft=0.69)
g = release_gate(good)
assert g["decision"] == "ship" and g["reasons"] == [] and g["p"] < 0.05
assert abs(g["p"] - 0.0011107) < 1e-6
h = release_gate(dict(good, gen_ft=0.60))
assert h["decision"] == "hold" and "forgetting" in h["reasons"][0]
h = release_gate(dict(good, wins=12, losses=8))
assert h["decision"] == "hold" and h["p"] > 0.05 and "sign test" in h["reasons"][0]
h = release_gate(dict(good, ft_loss=1.6))
assert h["decision"] == "hold" and "loss" in h["reasons"][0]
# ties do not change the p-value
assert release_gate(dict(good, ties=0))["p"] == g["p"]
print("project-domain-finetune OK")
