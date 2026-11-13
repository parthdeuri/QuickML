import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-research-assistant.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
R = ns["Researcher"]

corpus = {
    "d1": "Chunking splits documents into pieces. Overlap keeps context across cuts.",
    "d2": "Embeddings map text to vectors. Cosine similarity ranks chunks.",
    "d3": "Embeddings map text to vectors. Cosine similarity ranks chunks.",
    "d4": "Rerankers rescore the top candidates. They cost one call per pair.",
}
planner = lambda q: ["how does chunking work", "how are embeddings ranked", "what do rerankers cost"]

r = R(corpus, planner)
out = r.run("How does retrieval work?")
assert out["stop"] == "done" and out["steps"] == 3 and len(out["report"]) == 3
retrieved = {n["src"] for sq in planner("") for n in r.gather(sq)}
assert retrieved == {"d1", "d2", "d4"}
assert all(c["src"] in retrieved and c["ok"] for c in out["report"])

# duplicate sources merged: d3 never retrieved, alias recorded
assert r.alias == {"d3": "d2"} and set(r.docs) == {"d1", "d2", "d4"}

# deliberately unsupported claim and an uncited-source claim
def writer(notes):
    base = [(n["text"], n["src"]) for n in notes]
    return base + [("Chunking guarantees perfect accuracy.", "d1"),
                   ("Rerankers are free.", "d9")]
out2 = R(corpus, planner, writer).run("q")
claims = out2["report"]
assert all(c["src"] in retrieved for c in claims)       # d9 dropped
assert len(claims) == 4
bad = [c for c in claims if not c["ok"]]
assert [c["text"] for c in bad] == ["Chunking guarantees perfect accuracy."]
assert bad[0]["support"] == 0.25

# budgets stop the run with a reason
o = R(corpus, planner, max_steps=2).run("q")
assert o["stop"] == "step_budget" and o["steps"] == 2
o = R(corpus, planner, max_tokens=9).run("q")
assert o["stop"] == "token_budget" and o["steps"] < 3

# deterministic
assert R(corpus, planner).run("q") == R(corpus, planner).run("q")
print("project-research-assistant OK")
