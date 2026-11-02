import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "rnn-cell.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
T, I, H, B = 7, 5, 4, 3
cell = torch.nn.RNNCell(I, H).double()
Wx = cell.weight_ih.detach().numpy()
Wh = cell.weight_hh.detach().numpy()
b = (cell.bias_ih + cell.bias_hh).detach().numpy()
X = rng.normal(size=(T, B, I))
h0 = rng.normal(size=(B, H))
out, hT = ns["rnn_forward"](X, h0, Wx, Wh, b)
h = torch.tensor(h0)
ref = []
for t in range(T):
    h = cell(torch.tensor(X[t]), h)
    ref.append(h.detach().numpy())
assert out.shape == (T, B, H)
assert np.allclose(out, np.stack(ref))
assert np.allclose(hT, ref[-1])
print("rnn-cell OK")
