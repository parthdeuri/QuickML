import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "huggingface-generate-pipeline.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
tok, model = ns["build"](0)
encode, manual, decode = ns["encode"], ns["manual_greedy"], ns["decode"]
N = 6

def gen(ids, mask):
    return model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                          max_new_tokens=N, pad_token_id=0)

# single prompt
ids, mask = encode(tok, ["the cat sat on"])
assert mask.all()
ref = gen(ids, mask)
assert torch.equal(manual(model, ids, mask, N), ref)
assert ref.shape == (1, ids.shape[1] + N)

# left-padded batch of two different-length prompts
texts = ["the cat sat on", "a dog"]
ids, mask = encode(tok, texts, "left")
assert mask.tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]
assert ids[1, :2].tolist() == [0, 0]
ref = gen(ids, mask)
out = manual(model, ids, mask, N)
assert torch.equal(out, ref)
# batched result equals running each prompt alone
for i, t in enumerate(texts):
    i1, m1 = encode(tok, [t])
    alone = gen(i1, m1)[0, i1.shape[1]:]
    assert torch.equal(alone, ref[i, ids.shape[1]:]), (t, alone, ref[i])

# right padding changes the generation of the shorter prompt
ids_r, mask_r = encode(tok, texts, "right")
assert ids_r[1, -2:].tolist() == [0, 0]
ref_r = gen(ids_r, mask_r)
i1, m1 = encode(tok, [texts[1]])
alone = gen(i1, m1)[0, i1.shape[1]:]
assert not torch.equal(ref_r[1, ids_r.shape[1]:], alone), "right pad should differ"
# the longest prompt (no padding) is unaffected
assert torch.equal(ref_r[0, ids_r.shape[1]:], ref[0, ids.shape[1]:])

# slicing off the prompt leaves only new tokens
new = ref[:, ids.shape[1]:]
assert new.shape == (2, N)
assert torch.equal(ref[:, :ids.shape[1]], ids)
full = decode(tok, ref)
only_new = decode(tok, new)
assert "<pad>" not in full[1]                     # special tokens skipped
assert full[1].startswith("a dog")
assert full[1] == ("a dog " + only_new[1]).strip()
print("huggingface-generate-pipeline OK")
