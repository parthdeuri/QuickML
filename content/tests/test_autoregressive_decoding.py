import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "autoregressive-decoding.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
V, d = 8, 5
emb, W = rng.normal(size=(V, d)), rng.normal(size=(d, V))
prompt = [1, 2]
full = ns["uncached"](emb, W)

# reference: iterate argmax of a full forward pass step by step
ref, EOS = list(prompt), 99
for _ in range(10):
    ref.append(int(np.argmax(emb[ref].mean(axis=0) @ W)))
out = ns["generate"](full, prompt, 10, EOS)
assert out == ref and len(out) == len(prompt) + 10  # max_new_tokens respected

# stops at EOS, and EOS is kept as the last token
eos = ref[len(prompt)]
out = ns["generate"](full, prompt, 10, eos)
assert out == prompt + [eos]
eos2 = next(t for t in range(V) if t not in ref)
assert ns["generate"](full, prompt, 0, eos2) == prompt

# cached logits equal uncached logits at every step, hence same output
assert ns["generate"](ns["cached"](emb, W), prompt, 10, EOS) == ref
tok, c = list(prompt), ns["cached"](emb, W)
for _ in range(6):
    assert np.allclose(c(tok), full(tok), rtol=0, atol=1e-12)
    tok.append(int(np.argmax(full(tok))))
print("autoregressive-decoding OK")
