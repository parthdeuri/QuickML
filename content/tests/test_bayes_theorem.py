import numpy as np, yaml
from fractions import Fraction
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "bayes-theorem.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# Disease prevalence 1/100, sensitivity 9/10, false-positive rate 1/10.
# Hand computation: (1/100 * 9/10) / (1/100 * 9/10 + 99/100 * 1/10) = 9/108 = 1/12
post = ns["posterior"]([0.01, 0.99], [0.9, 0.1])
assert np.isclose(post[0], 1 / 12) and np.isclose(post.sum(), 1.0)

# Exact rational arithmetic goes through the same code path.
exact = ns["posterior"]([Fraction(1, 100), Fraction(99, 100)],
                        [Fraction(9, 10), Fraction(1, 10)])
assert exact[0] == Fraction(1, 12)

# Two independent positive tests: odds 1/99 * 9 * 9 = 81/99 -> 81/180 = 9/20
two = ns["update"]([0.01, 0.99], [[0.9, 0.1], [0.9, 0.1]])
assert np.isclose(two[0], 9 / 20)

# Three hypotheses: posterior is proportional to prior * likelihood
p = ns["posterior"]([0.5, 0.3, 0.2], [0.1, 0.4, 0.5])
assert np.allclose(p, np.array([0.05, 0.12, 0.10]) / 0.27)
print("bayes-theorem OK")
