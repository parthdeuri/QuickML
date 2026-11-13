import yaml
d = yaml.safe_load(open("content/patterns/compare-react-vs-plan-and-execute.yaml", encoding="utf-8"))
ns = {}
exec(d["template"], ns)
g = ns["GRAPH"]
assert ns["react_calls"](g) == 6 and ns["plan_execute_calls"](g) == 2
assert ns["critical_path"](g) == 3 < len(g)
chain = {"a": [], "b": ["a"], "c": ["b"]}
assert ns["critical_path"](chain) == len(chain)
flat = {"a": [], "b": [], "c": []}
assert ns["critical_path"](flat) == 1
print("pattern compare-react-vs-plan-and-execute OK")
