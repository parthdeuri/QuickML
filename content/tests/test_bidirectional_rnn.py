import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "bidirectional-rnn.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
torch.manual_seed(0)
rng = np.random.default_rng(0)
T, I, H = 6, 4, 5
X = rng.normal(size=(T, I))

rnn = torch.nn.RNN(I, H, bidirectional=True, nonlinearity="tanh").double()
g = lambda name: getattr(rnn, name).detach().numpy()
fwd = (g("weight_ih_l0").T, g("weight_hh_l0").T, g("bias_ih_l0") + g("bias_hh_l0"))
bwd = (g("weight_ih_l0_reverse").T, g("weight_hh_l0_reverse").T,
       g("bias_ih_l0_reverse") + g("bias_hh_l0_reverse"))
out = ns["bi_rnn"](X, fwd, bwd)
ref, hn = rnn(torch.tensor(X)[:, None, :])        # (T, 1, 2H), (2, 1, H)
assert out.shape == (T, 2 * H)
assert np.allclose(out, ref[:, 0].detach().numpy())
assert np.allclose(out[-1, :H], hn[0, 0].detach().numpy())   # forward final
assert np.allclose(out[0, H:], hn[1, 0].detach().numpy())    # backward final

# second reference: two separate unidirectional nn.RNN passes
f = torch.nn.RNN(I, H).double()
b = torch.nn.RNN(I, H).double()
with torch.no_grad():
    f.weight_ih_l0.copy_(rnn.weight_ih_l0); f.weight_hh_l0.copy_(rnn.weight_hh_l0)
    f.bias_ih_l0.copy_(rnn.bias_ih_l0); f.bias_hh_l0.copy_(rnn.bias_hh_l0)
    b.weight_ih_l0.copy_(rnn.weight_ih_l0_reverse); b.weight_hh_l0.copy_(rnn.weight_hh_l0_reverse)
    b.bias_ih_l0.copy_(rnn.bias_ih_l0_reverse); b.bias_hh_l0.copy_(rnn.bias_hh_l0_reverse)
xt = torch.tensor(X)[:, None, :]
of, _ = f(xt)
ob, _ = b(xt.flip(0))
two = torch.cat([of, ob.flip(0)], dim=-1)[:, 0].detach().numpy()
assert np.allclose(out, two)
# the backward half at the last position only saw the last token
assert np.allclose(out[-1, H:], np.tanh(X[-1] @ bwd[0] + bwd[2]))
print("bidirectional-rnn OK")
