import re
import yaml
import numpy as np
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-pdf-question-answering.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
ingest, answer, embed = ns["ingest"], ns["answer"], ns["embed"]

PAGES = [
    "Quarterly Report. The company sells solar panels and home batteries across three regions. Revenue grew steadily during the year.",
    "Operations. The main factory is located in Pune and employs four hundred workers. Shipping is handled by two logistics partners.",
    "Warranty. Every solar panel carries a twenty five year warranty on power output. Batteries are covered for ten years against defects.",
    "Outlook. Management plans to open a second factory in Chennai next year. Hiring will focus on engineers and technicians.",
]
idx = ingest(PAGES)

# metadata survives ingestion; ids are stable and unique
assert [c["id"] for c in idx] == list(range(len(idx)))
assert {c["page"] for c in idx} == {1, 2, 3, 4}
assert all(c["text"] and abs(np.linalg.norm(c["vec"]) - 1) < 1e-9 for c in idx)
# cleaning rejoins hyphenated line breaks
assert ns["clean"]("ware-\nhouse  stock") == "warehouse stock"

# a page-3 fact retrieves a page-3 chunk and cites [p3]
r = answer("how long is the warranty on solar panels", idx)
assert not r["abstained"] and r["citations"] == ["p3"], r
assert idx[r["retrieved"][0]]["page"] == 3 and "[p3]" in r["answer"]

# every citation points to a retrieved chunk, for all questions
LABELLED = [("how long is the warranty on solar panels", 3),
            ("where is the main factory located", 2),
            ("where will the second factory open", 4),
            ("what does the company sell", 1),
            ("how many workers does the factory employ", 2),
            ("how long are batteries covered", 3)]
hits = 0
for q, page in LABELLED:
    r = answer(q, idx, k=3)
    pages = {f"p{idx[i]['page']}" for i in r["retrieved"]}
    assert set(r["citations"]) <= pages and r["citations"], (q, r)
    hits += page in {idx[i]["page"] for i in r["retrieved"]}
recall = hits / len(LABELLED)
assert recall >= 0.8, recall  # stated floor; the toy set reaches 1.0

# out-of-scope questions abstain and cite nothing
for q in ["what is the capital of france", "how tall is mount everest",
          "who won the football world cup"]:
    r = answer(q, idx)
    assert r["abstained"] and r["citations"] == [], (q, r)

# threshold sits in the gap between in-scope and out-of-scope scores
top = lambda q: max(float(c["vec"] @ embed(q)) for c in idx)
lo_in = min(top(q) for q, _ in LABELLED)
hi_out = max(top(q) for q in ["what is the capital of france",
                              "how tall is mount everest",
                              "who won the football world cup"])
assert hi_out < 0.25 < lo_in, (hi_out, lo_in)
# deterministic
r1 = answer("where is the main factory located", idx)
r2 = answer("where is the main factory located", ingest(PAGES))
assert r1 == r2
print("project-pdf-question-answering OK")
