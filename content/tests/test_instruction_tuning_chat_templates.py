import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "instruction-tuning-chat-templates.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# instruction record with and without input
r1 = {"instruction": "Translate", "input": "chat", "output": "cat"}
assert ns["to_text"](r1) == "<|user|>\nTranslate\nchat\n<|assistant|>\ncat\n</s>"
r2 = {"instruction": "Say hi", "input": "", "output": "hi"}
assert ns["to_text"](r2) == "<|user|>\nSay hi\n<|assistant|>\nhi\n</s>"
# multi-turn chat: EOS after every assistant turn
chat = {"messages": [{"role": "system", "content": "Be brief"},
                     {"role": "user", "content": "a"},
                     {"role": "assistant", "content": "b"},
                     {"role": "user", "content": "c"},
                     {"role": "assistant", "content": "d"}]}
exp = ("<|system|>\nBe brief\n<|user|>\na\n<|assistant|>\nb\n</s>"
       "<|user|>\nc\n<|assistant|>\nd\n</s>")
assert ns["to_text"](chat) == exp
# custom template
t = {"turn": "[{role}] {content};", "eos": "<eos>"}
assert ns["to_text"](r2, t) == "[user] Say hi;[assistant] hi;<eos>"

# mask is 1 exactly on assistant tokens and EOS
roles = ["system", "user", "user", "header", "assistant", "assistant", "eos",
         "user", "header", "assistant", "eos"]
m = ns["assistant_label_mask"](roles)
assert m == [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
assert all((x == 1) == (r in ("assistant", "eos")) for x, r in zip(m, roles))

# mixture weights
sizes = [1000, 100, 10]
w1 = ns["mixture_weights"](sizes, 1.0)
assert np.isclose(w1.sum(), 1) and np.allclose(w1, np.array(sizes) / sum(sizes))
w0 = ns["mixture_weights"](sizes, 0.0)
assert np.allclose(w0, 1 / 3)
wh = ns["mixture_weights"](sizes, 0.5)
assert np.isclose(wh.sum(), 1) and wh[2] > w1[2] and wh[0] < w1[0]
assert np.allclose(wh, np.sqrt(sizes) / np.sqrt(sizes).sum())

# seeded draw: empirical frequencies match weights
rng = np.random.default_rng(0)
N = 40000
draws = rng.choice(3, size=N, p=wh)
counts = np.bincount(draws, minlength=3)
assert np.allclose(counts, N * wh, atol=4 * np.sqrt(N * wh * (1 - wh)).max()), (counts, N * wh)
print("instruction-tuning-chat-templates OK")
