import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-rag-knowledge-assistant.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
Assistant, faithfulness, toks, IDK = (ns[n] for n in ("Assistant", "faithfulness", "toks", "IDK"))

HR = "Remote work requires manager approval. Salary bands are reviewed every March."
ENG = "Deploys freeze on Fridays after noon. The on-call rotation changes every Monday."
PUB = "The office opens at nine every weekday. Visitors must sign in at reception."

def build():
    a = Assistant()
    a.ingest("hr", HR, ["hr"])
    a.ingest("eng", ENG, ["eng", "hr"])
    a.ingest("pub", PUB, ["all", "eng", "hr"])
    return a

eng = {"groups": ["eng", "all"]}
hr = {"groups": ["hr", "all"]}
Q = "When are salary bands reviewed?"

# --- ACL hidden document is never retrieved or cited ----------------
a = build()
r = a.ask(eng, Q)
assert "hr#1" not in r["ctx"] and not any(c.startswith("hr") for c in r["cites"])
assert "salary" not in r["ctx"].lower() and "March" not in r["answer"]
assert r["abstained"] and r["answer"] == IDK
# the same question reaches it for an authorised user
r = a.ask(hr, Q)
assert not r["abstained"] and r["cites"] == ["hr#1"]
assert "March" in r["answer"]
# every citation, for every user and question, is readable by that user
for user in (eng, hr, {"groups": []}, {"groups": ["all"]}):
    for q in (Q, "When do deploys freeze?", "Where do visitors sign in?",
              "Who approves remote work?", "on-call rotation changes"):
        r = a.ask(user, q)
        for cid in r["cites"]:
            doc, i = cid.split("#")
            assert a.chunks[(doc, int(i))]["acl"] & set(user["groups"]), (user, q, cid)
assert a.ask({"groups": []}, "Where do visitors sign in?")["abstained"]

# --- answers contain only facts present in retrieved chunks ---------
for user, q in [(eng, "When do deploys freeze?"),
                (hr, Q), (eng, "Where do visitors sign in?")]:
    r = a.ask(user, q)
    assert not r["abstained"]
    body = r["answer"].split("] ", 1)[1]
    assert body in r["ctx"]
    assert faithfulness(r["answer"], r["ctx"]) == 1.0

# --- weak evidence abstains -----------------------------------------
r = a.ask(eng, "What is the parental leave policy?")
assert r["abstained"] and r["cites"] == [] and r["answer"] == IDK

# --- updating a document replaces its old facts ---------------------
a.ingest("eng", "Deploys freeze on Thursdays after noon.", ["eng", "hr"])
r = a.ask(eng, "When do deploys freeze?")
assert "Thursdays" in r["answer"] and "Fridays" not in r["ctx"]
assert not any(k[0] == "eng" and k[1] > 0 for k in a.chunks)
assert a.ask(eng, "How often does the on-call rotation change?")["abstained"]

# --- deleted documents vanish ---------------------------------------
a.remove("pub")
assert a.ask(eng, "Where do visitors sign in?")["abstained"]
assert all(k[0] != "pub" for k in a.chunks)
print("project-rag-knowledge-assistant OK")
