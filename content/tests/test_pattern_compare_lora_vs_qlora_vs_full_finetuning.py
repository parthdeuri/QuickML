import yaml
d = yaml.safe_load(open("content/patterns/compare-lora-vs-qlora-vs-full-finetuning.yaml"))
ns = {}
exec(d["template"], ns)
r = ns["res"]
assert r["full"][0] == 4096 * 4096
assert r["lora"][0] == r["qlora"][0] == 16 * 8192
assert r["lora"][0] / r["full"][0] < 0.01
assert r["qlora"][1] < r["lora"][1] < r["full"][1]
assert r["qlora"][1] < 0.6 * r["lora"][1]
print("pattern compare-lora-vs-qlora-vs-full-finetuning OK")
