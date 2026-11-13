import numpy as np, yaml, torch
from pathlib import Path
from transformers import GPT2Config, GPT2LMHeadModel

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gpt-decoder-only.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

torch.manual_seed(0)
cfg = GPT2Config(vocab_size=11, n_embd=8, n_head=2, n_layer=2, n_positions=6, n_ctx=6,
                 resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
model = GPT2LMHeadModel(cfg).double().eval()
sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
# make LayerNorm params and biases non-trivial
for k, v in model.named_parameters():
    if "ln" in k or "bias" in k:
        v.data += 0.1 * torch.randn_like(v)
sd = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}

params = {"wte": sd["transformer.wte.weight"], "wpe": sd["transformer.wpe.weight"],
          "gf": sd["transformer.ln_f.weight"], "bf": sd["transformer.ln_f.bias"], "blocks": []}
for i in range(cfg.n_layer):
    p = f"transformer.h.{i}."
    params["blocks"].append({
        "g1": sd[p + "ln_1.weight"], "b1": sd[p + "ln_1.bias"],
        "Wqkv": sd[p + "attn.c_attn.weight"], "bqkv": sd[p + "attn.c_attn.bias"],
        "Wo": sd[p + "attn.c_proj.weight"], "bo": sd[p + "attn.c_proj.bias"],
        "g2": sd[p + "ln_2.weight"], "b2": sd[p + "ln_2.bias"],
        "W1": sd[p + "mlp.c_fc.weight"], "c1": sd[p + "mlp.c_fc.bias"],
        "W2": sd[p + "mlp.c_proj.weight"], "c2": sd[p + "mlp.c_proj.bias"]})

# weight tying: lm_head is literally the embedding matrix
assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
assert np.array_equal(model.lm_head.weight.detach().numpy(), params["wte"])

ids = np.array([3, 1, 7, 0, 10, 4])
mine = ns["gpt_forward"](ids, params, heads=2)
with torch.no_grad():
    ref = model(torch.tensor(ids)[None]).logits[0].numpy()
assert mine.shape == (6, 11)
assert np.allclose(mine, ref, atol=1e-6), np.abs(mine - ref).max()

# causality: changing token j must not change logits before j
for j in range(1, 6):
    ids2 = ids.copy()
    ids2[j] = (ids2[j] + 1) % 11
    out2 = ns["gpt_forward"](ids2, params, heads=2)
    assert np.allclose(out2[:j], mine[:j], atol=1e-12)
    assert not np.allclose(out2[j:], mine[j:])

# shorter prefix gives same logits as the matching rows (decoding consistency)
assert np.allclose(ns["gpt_forward"](ids[:4], params), mine[:4], atol=1e-12)
print("gpt-decoder-only OK")
