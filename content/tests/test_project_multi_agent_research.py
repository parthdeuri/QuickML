import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-multi-agent-research.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
run, score = ns["run_project"], ns["report_score"]

corpus = {
    "d1": "ml definition a model that learns patterns from data",
    "d2": "ml benefits automation and better predictions at scale",
    "d3": "ml risks bias drift and privacy leaks in deployed models",
}
r = run("ml", corpus, 20)
assert r["reason"] == "done" and len(r["report"]) == 3
assert [s["title"] for s in r["report"]] == ["ml definition", "ml benefits", "ml risks"]
assert score(r["report"], corpus) == 1.0
for s in r["report"]:
    assert any("[%s]" % d in s["body"] for d in corpus)

# critic rejects the first risks draft once, then accepts the revision
calls = r["board"]["calls"]
rs = [c for c in calls if c[0] == "researcher" and "ml risks" in c[1]]
assert len(rs) == 2
verdicts = [c[2] for c in calls if c[0] == "critic"]
assert sum(v.startswith("reject") for v in verdicts) == 1
assert verdicts[-1] == "accept"
assert len(calls) == 10  # 2 + 2S + 2R with S=3, R=1

# token cost equals an independent sum
indep = 0
for role, p, o in calls:
    indep += len(p.split())
    if isinstance(o, str):
        indep += len(o.split())
    else:  # planner list or writer list of dicts: count words in contents
        for item in o:
            if isinstance(item, str):
                indep += len(item.split())
            else:
                indep += len(item["title"].split()) + len(item["body"].split())
assert indep == r["board"]["tokens"], (indep, r["board"]["tokens"])

# turn limit stops the run with a reason and no report
r5 = run("ml", corpus, 5)
assert r5["reason"] == "max_turns" and r5["report"] is None
assert len(r5["board"]["calls"]) == 5

# determinism
assert run("ml", corpus, 20)["report"] == r["report"]
assert run("ml", corpus, 20)["board"]["tokens"] == r["board"]["tokens"]

# score counts only valid citations
bad = [{"title": "a", "body": "x [zz]"}, {"title": "b", "body": "y [d1]"}]
assert score(bad, corpus) == 0.5
# expected turns formula
S, p = 3, 1 / 3
assert abs(2 + 2 * S * (1 + p) - 10) < 1e-12
print("project-multi-agent-research OK")
