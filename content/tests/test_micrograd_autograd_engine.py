import math, numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "micrograd-autograd-engine.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
Value = ns["Value"]

# 1. composed scalar expression vs torch.autograd (float64)
def expr(a, b, c):
    return (a * b + c).tanh() ** 2 + a.exp() - b.relu() * c + (a - b) * 0.5 + 2 * c

av, bv, cv = 0.7, 1.3, -0.4
a, b, c = Value(av), Value(bv), Value(cv)
out = expr(a, b, c)
out.backward()
ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (av, bv, cv))
tout = (ta * tb + tc).tanh() ** 2 + ta.exp() - tb.relu() * tc + (ta - tb) * 0.5 + 2 * tc
tout.backward()
assert math.isclose(out.data, tout.item(), rel_tol=1e-12)
for v, t in ((a, ta), (b, tb), (c, tc)):
    assert math.isclose(v.grad, t.grad.item(), rel_tol=1e-10, abs_tol=1e-12)

# relu with a negative input: zero gradient
n = Value(-2.0)
(n.relu() * 3).backward()
assert n.grad == 0.0

# 2. tiny two-layer MLP with MSE loss
rng = np.random.default_rng(0)
X = rng.normal(size=(4, 2))
Y = rng.normal(size=4)
W1, B1 = rng.normal(size=(2, 3)), rng.normal(size=3)
W2, B2 = rng.normal(size=3), rng.normal()
vW1 = [[Value(float(W1[i, j])) for j in range(3)] for i in range(2)]
vB1 = [Value(float(x)) for x in B1]
vW2 = [Value(float(x)) for x in W2]
vB2 = Value(float(B2))
loss = Value(0.0)
for x, y in zip(X, Y):
    hid = []
    for j in range(3):
        s = vB1[j]
        for i in range(2):
            s = s + vW1[i][j] * float(x[i])
        hid.append(s.tanh())
    o = vB2
    for j in range(3):
        o = o + hid[j] * vW2[j]
    loss = loss + (o - float(y)) ** 2
loss = loss * (1 / 4)
loss.backward()
tW1, tB1, tW2, tB2 = (torch.tensor(m, dtype=torch.float64, requires_grad=True) for m in (W1, B1, W2, B2))
tX, tY = torch.tensor(X), torch.tensor(Y)
tl = ((torch.tanh(tX @ tW1 + tB1) @ tW2 + tB2 - tY) ** 2).mean()
tl.backward()
assert math.isclose(loss.data, tl.item(), rel_tol=1e-12)
g1 = np.array([[vW1[i][j].grad for j in range(3)] for i in range(2)])
assert np.allclose(g1, tW1.grad.numpy(), atol=1e-12)
assert np.allclose([v.grad for v in vB1], tB1.grad.numpy(), atol=1e-12)
assert np.allclose([v.grad for v in vW2], tW2.grad.numpy(), atol=1e-12)
assert math.isclose(vB2.grad, tB2.grad.item(), rel_tol=1e-10)

# 3. a reused node receives the summed gradient: d(x*x + x)/dx = 2x + 1
x = Value(3.0)
y = x * x + x
y.backward()
assert x.grad == 2 * 3.0 + 1 == 7.0

# 4. topological order places every node after its parents
cap = {}
orig = Value.backward
def collect(self):
    order, seen = [], set()
    def visit(v):
        if v not in seen:
            seen.add(v)
            for p in v.parents:
                visit(p)
            order.append(v)
    visit(self)
    return order
a, b = Value(2.0), Value(-1.0)
z = ((a * b + a) ** 2 + b.exp()).tanh() * a
order = collect(z)
pos = {id(v): i for i, v in enumerate(order)}
assert order[-1] is z and len(order) == len(set(map(id, order)))
for v in order:
    assert all(pos[id(p)] < pos[id(v)] for p in v.parents)

# 5. backward twice without zeroing: gradients accumulate. For a single
# op the leaves exactly double; in a deeper graph the stale intermediate
# grads are pushed down again, so leaves are inflated beyond 2x.
p, q = Value(2.0), Value(5.0)
r = p * q
r.backward()
assert (p.grad, q.grad) == (5.0, 2.0)
r.backward()
assert (p.grad, q.grad, r.grad) == (10.0, 4.0, 1.0)
x = Value(2.0)
y = x * x * x
y.backward()
g1 = x.grad
assert g1 == 12.0
y.backward()
assert x.grad > 2 * g1 and y.grad == 1.0
x.grad = 0.0
for v in collect(y):
    v.grad = 0.0
y.backward()
assert x.grad == g1

print("micrograd-autograd-engine OK")
