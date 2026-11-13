import yaml
d = yaml.safe_load(open("content/patterns/compare-rag-vs-finetuning.yaml"))
ns = {}
exec(d["template"], ns)
assert ns["fresh"] == ("12 dollars", "source:price")
assert ns["stale"] == ("10 dollars", None)
c = ns["choose"]
assert c(True, False, False) == ["rag"]
assert c(False, True, False) == ["rag"]
assert c(False, False, True) == ["finetune"]
assert c(True, False, True) == ["rag", "finetune"]
assert c(False, False, False) == ["prompting"]
print("pattern compare-rag-vs-finetuning OK")
