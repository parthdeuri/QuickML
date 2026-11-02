import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "lstm-cell.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
T, I, H, B = 6, 5, 4, 3
cell = torch.nn.LSTMCell(I, H).double()
Wx = cell.weight_ih.detach().numpy()
Wh = cell.weight_hh.detach().numpy()
b = (cell.bias_ih + cell.bias_hh).detach().numpy()
X = rng.normal(size=(T, B, I))
h, c = rng.normal(size=(B, H)), rng.normal(size=(B, H))
ht, ct = torch.tensor(h), torch.tensor(c)
for t in range(T):
    h, c = ns["lstm_cell"](X[t], h, c, Wx, Wh, b)
    ht, ct = cell(torch.tensor(X[t]), (ht, ct))
    assert np.allclose(h, ht.detach().numpy()), t
    assert np.allclose(c, ct.detach().numpy()), t
assert h.shape == (B, H)
print("lstm-cell OK")
