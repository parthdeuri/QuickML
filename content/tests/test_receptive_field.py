import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "receptive-field.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

for layers in [[(3, 1), (3, 2), (5, 1), (3, 2)], [(7, 2), (3, 1), (3, 3)], [(1, 1), (5, 2)]]:
    convs = []
    for k, s in layers:
        c = torch.nn.Conv2d(1, 1, k, stride=s, bias=False).double()
        torch.nn.init.constant_(c.weight, 1.0)  # positive weights: no cancellation
        convs.append(c)
    net = torch.nn.Sequential(*convs)
    rf, jump = ns["receptive_field"](layers)
    x = torch.zeros(1, 1, 80, 80, dtype=torch.float64, requires_grad=True)
    out = net(x)
    for (i, j) in [(0, 0), (2, 3), (out.shape[2] - 1, out.shape[3] - 1)]:
        x.grad = None
        net(x)[0, 0, i, j].backward(retain_graph=False)
        rows, cols = np.nonzero(x.grad[0, 0].numpy())
        lo_r, hi_r = ns["input_span"](layers, i)
        lo_c, hi_c = ns["input_span"](layers, j)
        assert (rows.min(), rows.max()) == (lo_r, hi_r), (layers, i)
        assert (cols.min(), cols.max()) == (lo_c, hi_c), (layers, j)
        assert hi_r - lo_r + 1 == rf and hi_c - lo_c + 1 == rf
    assert lo_r == (out.shape[2] - 1) * jump
print("receptive-field OK")
