import yaml
d = yaml.safe_load(open("content/patterns/compare-bpe-vs-wordpiece-vs-unigram.yaml"))
ns = {}
exec(d["template"], ns)
assert ns["bpe"] == ("u", "g") and ns["two"][ns["bpe"]] == 20
assert ns["wp"] == ("g", "s")
assert ns["bpe"] != ns["wp"]
cost = ns["cost"]
# the pair BPE picks is never needed by the Unigram likelihood
assert cost["ug"] < 1e-9
assert ns["uni"] == "gs" and cost["gs"] > 10
print("pattern compare-bpe-vs-wordpiece-vs-unigram OK")
