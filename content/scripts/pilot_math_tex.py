"""Pilot: appends `math_tex` to six cards (idempotent). Superseded by per-card edits later."""
import re
from pathlib import Path

import yaml

P = Path(__file__).resolve().parent.parent / "problems"

D = {
 "self-attention": [
  r"$Q = XW_q,\;\; K = XW_k,\;\; V = XW_v \qquad (X:\ T \times d)$",
  r"$S = \frac{QK^{\top}}{\sqrt{d_k}}$",
  r"Causal: set $S_{ij} = -\infty$ for $j > i$",
  r"$A = \mathrm{softmax}(S)$ row-wise",
  r"$\mathrm{Output} = A\,V$",
 ],
 "adam-optimizer": [
  r"$m \leftarrow \beta_1 m + (1-\beta_1)\,g$",
  r"$v \leftarrow \beta_2 v + (1-\beta_2)\,g^2$",
  r"$\hat{m} = \frac{m}{1-\beta_1^{\,t}},\;\; \hat{v} = \frac{v}{1-\beta_2^{\,t}}$",
  r"$w \leftarrow w - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}$",
  r"Defaults: $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$",
 ],
 "bayes-theorem": [
  r"$P(H\,|\,E) = \frac{P(E\,|\,H)\,P(H)}{P(E)}$",
  r"$P(E) = \sum_h P(E\,|\,h)\,P(h)$",
  r"$\mathrm{posterior} \propto \mathrm{likelihood} \times \mathrm{prior}$",
  r"$\mathrm{odds}(H\,|\,E) = \mathrm{odds}(H)\,\frac{P(E\,|\,H)}{P(E\,|\,\neg H)}$",
  r"Sequential: posterior after $E_1$ is the prior for $E_2$",
 ],
 "svd": [
  r"$A = U S V^{\top},\;\; U^{\top}U = I,\;\; V^{\top}V = I,\;\; s_1 \geq s_2 \geq \cdots \geq 0$",
  r"$A^{\top}A = V S^2 V^{\top}$ (eigendecomposition)",
  r"$u_i = \frac{A v_i}{s_i}$",
  r"Rank-$k$ truncation: $A_k = \sum_{i \leq k} s_i\,u_i v_i^{\top}$",
  r"$\|A - A_k\|_F^2 = \sum_{i > k} s_i^2$",
  r"$\|A - A_k\|_2 = s_{k+1}$",
 ],
 "lstm-cell": [
  r"$[i,f,g,o] = \mathrm{split}(W_x x_t + W_h h_{t-1} + b)$",
  r"$i_t = \sigma(i),\;\; f_t = \sigma(f),\;\; o_t = \sigma(o),\;\; g_t = \tanh(g)$",
  r"$c_t = f_t \odot c_{t-1} + i_t \odot g_t$",
  r"$h_t = o_t \odot \tanh(c_t)$",
  r"$\frac{\partial c_t}{\partial c_{t-1}} = f_t$ (direct path only)",
 ],
 "entropy-kl-divergence": [
  r"$H(p) = -\sum_x p(x)\log p(x)$",
  r"$H(p,q) = -\sum_x p(x)\log q(x)$",
  r"$\mathrm{KL}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$",
  r"$\mathrm{KL}(p\,\|\,q) = H(p,q) - H(p)$",
  r"$\mathrm{KL}(p\,\|\,q) \geq 0$, equality iff $p = q$",
  r"$\mathrm{KL}(p\,\|\,q) \neq \mathrm{KL}(q\,\|\,p)$ in general",
 ],
}

for slug, lines in D.items():
    f = P / f"{slug}.yaml"
    text = f.read_text(encoding="utf-8")
    text = re.sub(r"^math_tex:\n(?:- .*\n)+", "", text, flags=re.M)
    if not text.endswith("\n"):
        text += "\n"
    text += yaml.dump({"math_tex": lines}, allow_unicode=True, sort_keys=False, width=1000)
    f.write_text(text, encoding="utf-8", newline="")
print("pilot cards:", len(D))
