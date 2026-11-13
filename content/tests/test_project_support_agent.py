import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-support-agent.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
A = ns["SupportAgent"]()

# FAQ answered from the knowledge base with its source
s = {}
r = A.handle(s, "What is your returns policy?")
assert r["action"] == "answer" and r["source"] == "returns"
assert r["text"] == ns["KB"]["returns"]
r = A.handle({}, "how long does shipping take, what is the policy")
assert r["action"] == "answer" and r["source"] == "shipping", r

# order lookup is not gated
r = A.handle({}, "order status for A100 please")
assert r["action"] == "tool" and r["text"] == "shipped"

# refund: not executed until confirmed, then exactly once
s = {}
r = A.handle(s, "I want a refund for B200, mail me at jo@example.com")
assert r["action"] == "confirm" and A.refunds == []
r = A.handle(s, "yes")
assert r["action"] == "tool" and A.refunds == ["B200"]
A.handle(s, "yes")  # a stray repeat must not refund again
assert A.refunds == ["B200"]
# declining cancels
s = {}
A.handle(s, "refund A100")
r = A.handle(s, "no")
assert r["text"] == "Cancelled." and A.refunds == ["B200"]

# low confidence and upset customers escalate with transcript
s = {}
r = A.handle(s, "hmm my thing is weird")
assert r["action"] == "escalate" and r["why"] == "low confidence"
assert r["transcript"] == ["hmm my thing is weird"]
r = A.handle({}, "this is unacceptable, refund me")
assert r["action"] == "escalate" and r["why"] == "upset customer"
r = A.handle({}, "refund and order status")  # tie: c = 0.5 < tau
assert r["action"] == "escalate"

# logs never contain raw email or order ids
text = "\n".join(A.logs)
assert "jo@example.com" not in text and "<EMAIL>" in text
assert "B200" not in text and "A100" not in text

# unknown tool is denied; gate holds without confirmation
assert A.call("delete_account", {}).startswith("denied")
n = len(A.refunds)
assert A.call("request_refund", {"id": "A100"}) == "needs_confirmation"
assert len(A.refunds) == n

# math: expected cost per ticket
e, cb, ch = 0.25, 0.02, 4
assert abs((1 - e) * cb + e * ch - 1.015) < 1e-12
print("project-support-agent OK")
