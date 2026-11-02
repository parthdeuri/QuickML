"""Pilot: wrap formulas in prose fields with $...$ LaTeX for three cards (exact, single-hit edits)."""
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "problems"

E = {
 "adam-optimizer": [
  ("(1 − βᵗ) removes that bias", r"$(1-\beta^t)$ removes that bias"),
  ("Keep m, v (zeros) and a step counter t per", r"Keep $m$, $v$ (zeros) and a step counter $t$ per"),
  ("Update m as", "Update $m$ as"),
  ("Update v as", "Update $v$ as"),
  ("Bias-correct both with 1 − βᵗ —", r"Bias-correct both with $1-\beta^t$ —"),
  ("Step by η·m̂ / (√v̂ + ε) —", r"Step by $\eta\,\hat{m}/(\sqrt{\hat{v}}+\epsilon)$ —"),
  ("Learning rate η", r"Learning rate $\eta$"),
  ("β₁ (decay", r"$\beta_1$ (decay"),
  ("β₂ (decay", r"$\beta_2$ (decay"),
  ("ε (numerical", r"$\epsilon$ (numerical"),
  ("What do m and v represent?", "What do $m$ and $v$ represent?"),
  ("m is an exponential average of gradients", "$m$ is an exponential average of gradients"),
  ("and v is an exponential average", "and $v$ is an exponential average"),
  ("dividing by 1 − βᵗ rescales", r"dividing by $1-\beta^t$ rescales"),
 ],
 "self-attention": [
  ("Dividing scores by √d keeps", r"Dividing scores by $\sqrt{d}$ keeps"),
  ("Project X into queries", "Project $X$ into queries"),
  ("Compute scores QKᵀ and divide by √d_k —", r"Compute scores $QK^{\top}$ and divide by $\sqrt{d_k}$ —"),
  ("Mask future positions with −∞ if causal", r"Mask future positions with $-\infty$ if causal"),
  ("Multiply the weights by V —", "Multiply the weights by $V$ —"),
  ("Forgetting the √d_k scaling", r"Forgetting the $\sqrt{d_k}$ scaling"),
  ("gives NaN with −∞;", r"gives NaN with $-\infty$;"),
  ("Key/query dimension d_k (sets the scale √d_k)", r"Key/query dimension $d_k$ (sets the scale $\sqrt{d_k}$)"),
  ("Why divide by √d_k?", r"Why divide by $\sqrt{d_k}$?"),
  ("Dot products of d_k-dimensional vectors have variance growing with d_k;",
   "Dot products of $d_k$-dimensional vectors have variance growing with $d_k$;"),
  ("So position i attends only to positions up to i,", "So position $i$ attends only to positions up to $i$,"),
 ],
 "bayes-theorem": [
  ("P(H, E) = P(E | H) P(H) = P(H | E) P(E).",
   r"$P(H,E) = P(E\,|\,H)\,P(H) = P(H\,|\,E)\,P(E)$."),
  ("Dividing by P(E), which", "Dividing by $P(E)$, which"),
  ("get the evidence P(E) —", "get the evidence $P(E)$ —"),
 ],
}

bad = 0
for slug, edits in E.items():
    f = P / f"{slug}.yaml"
    text = f.read_text(encoding="utf-8")
    for old, new in edits:
        n = text.count(old)
        if n == 1:
            # a double-quoted YAML scalar needs every backslash doubled
            at = text.index(old)
            line = text[text.rfind("\n", 0, at) + 1:text.find("\n", at)]
            if line.lstrip().startswith(('- "', '"')) or ': "' in line:
                new = new.replace("\\", "\\\\")
            text = text.replace(old, new)
        elif n == 0 and new in text:
            pass  # already applied
        else:
            bad += 1
            print("MISSING/AMBIGUOUS", slug, n, old)
    f.write_text(text, encoding="utf-8", newline="")
print("pilot inline math done;", bad, "problems")
