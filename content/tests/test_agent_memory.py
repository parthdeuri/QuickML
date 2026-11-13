import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "agent-memory.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
decay, retrieve, window = ns["recency_decay"], ns["retrieve"], ns["window_with_summary"]

# decay
assert decay(0, 7.0) == 1.0
assert abs(decay(7.0, 7.0) - 0.5) < 1e-15
assert abs(decay(14.0, 7.0) - 0.25) < 1e-15

# hand-checked score: q=(1,0); m=(1,1)/sqrt2 -> cos = 1/sqrt2
q = np.array([1.0, 0.0])
m = {"text": "x", "vec": np.array([1.0, 1.0]), "t": 0.0, "imp": 0.8}
(s, txt), = retrieve([m], q, 1, (2.0, 3.0, 4.0), now=10.0, half_life=10.0)
assert abs(s - (2 * 2 ** -0.5 + 3 * 0.5 + 4 * 0.8)) < 1e-9 and txt == "x"

# weights flip the winner
mems = [
    {"text": "relevant-old", "vec": np.array([1.0, 0.0]), "t": 0.0, "imp": 0.1},
    {"text": "recent", "vec": np.array([0.0, 1.0]), "t": 99.0, "imp": 0.1},
    {"text": "important", "vec": np.array([0.0, 1.0]), "t": 0.0, "imp": 1.0},
]
top = lambda w: retrieve(mems, q, 1, w, now=100.0, half_life=5.0)[0][1]
assert top((1, 0, 0)) == "relevant-old"
assert top((0, 1, 0)) == "recent"
assert top((0, 0, 1)) == "important"

# retrieval returns k items sorted by score
res = retrieve(mems, q, 2, (1.0, 1.0, 1.0), now=100.0, half_life=5.0)
assert len(res) == 2 and res[0][0] >= res[1][0]
full = retrieve(mems, q, 10, (1.0, 1.0, 1.0), now=100.0, half_life=5.0)
assert len(full) == 3 and [x[0] for x in full] == sorted((x[0] for x in full), reverse=True)

# window
count = lambda msg: len(msg["content"].split())
summ = lambda dropped: "SUMMARY of %d turns" % len(dropped)  # 4 + digits tokens
msgs = [{"role": "user", "content": " ".join(["w"] * n)} for n in (5, 6, 7, 3, 4)]
for budget in (30, 20, 15, 11):
    out = window(msgs, budget, summ, count)
    assert sum(count(x) for x in out) <= budget, budget
    assert out[-1] == msgs[-1]  # newest turn kept
    if len(out) < len(msgs):
        assert out[0]["role"] == "system" and out[0]["content"].startswith("SUMMARY")
        kept = out[1:]
        n_drop = len(msgs) - len(kept)
        assert kept == msgs[n_drop:]
        assert out[0]["content"] == summ(msgs[:n_drop])
        assert sum(1 for x in out if x["role"] == "system") == 1
# fits entirely: unchanged, no summary
assert window(msgs, 1000, summ, count) == msgs
# keeps newest even when nothing else fits
assert window(msgs, 1, summ, count)[-1] == msgs[-1]
print("agent-memory OK")
