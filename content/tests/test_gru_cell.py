import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "gru-cell.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
B, I, H = 4, 5, 7
cell = torch.nn.GRUCell(I, H).double()
x = rng.normal(size=(B, I))
h = rng.normal(size=(B, H))
Wx = cell.weight_ih.detach().numpy().T      # (I, 3H) rows r|z|n
Wh = cell.weight_hh.detach().numpy().T
bx = cell.bias_ih.detach().numpy()
bh = cell.bias_hh.detach().numpy()
# make the biases non-trivial so the n-gate bias placement is exercised
with torch.no_grad():
    cell.bias_ih.copy_(torch.tensor(rng.normal(size=3 * H)))
    cell.bias_hh.copy_(torch.tensor(rng.normal(size=3 * H)))
bx = cell.bias_ih.detach().numpy()
bh = cell.bias_hh.detach().numpy()
out = ns["gru_cell"](x, h, Wx, Wh, bx, bh)
ref = cell(torch.tensor(x), torch.tensor(h)).detach().numpy()
assert out.shape == (B, H)
assert np.allclose(out, ref)
# single (unbatched) sample works too
one = ns["gru_cell"](x[0], h[0], Wx, Wh, bx, bh)
assert np.allclose(one, ref[0])
# unrolled over a sequence stays in step with torch
hn, ht = h, torch.tensor(h)
for _ in range(5):
    xs = rng.normal(size=(B, I))
    hn = ns["gru_cell"](xs, hn, Wx, Wh, bx, bh)
    ht = cell(torch.tensor(xs), ht)
assert np.allclose(hn, ht.detach().numpy())
print("gru-cell OK")
