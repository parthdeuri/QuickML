#!/usr/bin/env python3
"""Appends the optional `diagram:` block to selected cards (idempotent).

Run: python content/scripts/add_diagrams.py
Lines must be <= 46 characters (phone width) and 3-16 lines per card.
"""
from pathlib import Path
import re

import yaml

PROBLEMS = Path(__file__).resolve().parent.parent / "problems"

D = {
 "backpropagation": [
  "x ─► [W1,b1] ─► tanh ─► [W2,b2] ─► ŷ",
  "                                   │",
  "                              loss L(ŷ, y)",
  "                                   │",
  "x ◄─ dW1,db1 ◄─ ∂tanh ◄─ dW2,db2 ◄─┘",
  "   forward: left to right",
  "   gradients: right to left (chain rule)",
 ],
 "convolution-2d": [
  "input (5×5)         kernel (3×3)",
  " ■ ■ ■ · ·            a b c",
  " ■ ■ ■ · ·      ∗     d e f",
  " ■ ■ ■ · ·            g h i",
  " · · · · ·",
  " · · · · ·",
  "slide the kernel: each stop gives one output",
  "out size = (n − k + 2p) / s + 1",
 ],
 "lstm-cell": [
  "h(t-1), x(t) ─► [W, b] ─► f, i, g, o",
  "",
  "c(t-1) ══(× f)══(+)══════════► c(t)",
  "                 ▲",
  "              (i ⊙ g)",
  "h(t) = o ⊙ tanh(c(t))",
  "",
  "f forget · i input · g candidate · o output",
 ],
 "residual-connections": [
  "x ───────────┐",
  "│            │",
  "▼            │",
  "[ F(x) ]     │ identity",
  "│            │ shortcut",
  "▼            │",
  "(+) ◄────────┘",
  "│",
  "▼",
  "y = x + F(x)",
 ],
 "self-attention": [
  "x ─► Wq ─► Q ─┐",
  "x ─► Wk ─► K ─┴─► A = softmax(QKᵀ/√d)",
  "x ─► Wv ─► V ─────────► out = A · V",
  "",
  "A[i, j] = how much position i reads from j",
 ],
 "multi-head-attention": [
  "x ─► split into h heads",
  "     head 1: attention(Q1,K1,V1) ─┐",
  "     head 2: attention(Q2,K2,V2) ─┼► concat",
  "     head h: attention(Qh,Kh,Vh) ─┘",
  "                                  ▼",
  "                               [Wo] ─► out",
 ],
 "transformer-block": [
  "x ────────────┐",
  "│             │",
  "LayerNorm     │",
  "Self-attention│",
  "│             │",
  "(+) ◄─────────┘  = x1",
  "x1 ───────────┐",
  "│             │",
  "LayerNorm     │",
  "MLP (GELU)    │",
  "│             │",
  "(+) ◄─────────┘  = out",
 ],
 "autoencoder": [
  "x ─► [encoder] ─► z (small) ─► [decoder] ─► x̂",
  "",
  "loss = ‖x − x̂‖²",
  "z is the bottleneck: it must keep what the",
  "decoder needs to rebuild x",
 ],
 "vae-reparameterization": [
  "x ─► encoder ─► μ, log σ²",
  "                  │",
  "         z = μ + σ ⊙ ε,  ε ~ N(0, I)",
  "                  │",
  "z ─► decoder ─► x̂",
  "",
  "loss = ‖x − x̂‖² + KL(N(μ,σ²) ‖ N(0,I))",
 ],
 "gan-minimax-loss": [
  "noise z ─► Generator ─► fake ─┐",
  "real data ────────────────────┤",
  "                              ▼",
  "                       Discriminator",
  "                              │",
  "                       real or fake?",
  "D tells them apart; G tries to fool D",
 ],
 "cross-validation": [
  "fold 1   VAL  tr   tr   tr   tr",
  "fold 2   tr   VAL  tr   tr   tr",
  "fold 3   tr   tr   VAL  tr   tr",
  "fold 4   tr   tr   tr   VAL  tr",
  "fold 5   tr   tr   tr   tr   VAL",
  "",
  "score = mean of the 5 validation scores",
 ],
 "mlp-forward-pass": [
  "x (d) ─► [W1, b1] ─► act ─► h (m)",
  "                              │",
  "                         [W2, b2]",
  "                              │",
  "                       logits ─► softmax ─► p",
 ],
 "kv-cache": [
  "step t: compute q_t, k_t, v_t for ONE token",
  "cache:  K = [k1 … kt]    V = [v1 … vt]",
  "",
  "out_t = softmax(q_t Kᵀ / √d) · V",
  "(older keys and values are never recomputed)",
 ],
 "lora-low-rank-adaptation": [
  "x ────► W (frozen) ───────────┐",
  " └─► A (d×r) ─► B (r×d) ─────►(+) ─► y",
  "",
  "y = x W + (α / r) · x A B",
  "r ≪ d, so only A and B are trained",
 ],
 "k-means": [
  "init centres",
  "   │",
  "   ▼",
  "assign each point to nearest centre ◄─┐",
  "   │                                  │",
  "   ▼                                  │",
  "move each centre to its cluster mean ─┘",
  "repeat until assignments stop changing",
 ],
 "diffusion-forward-process": [
  "x₀ ─► x₁ ─► x₂ ─► … ─► x_T    add noise",
  "clean                   pure noise",
  "",
  "q(x_t | x₀) = N(√ᾱ_t x₀, (1 − ᾱ_t) I)",
  "",
  "the model learns the reverse: x_T ─► … ─► x₀",
 ],
 "batch-normalization": [
  "            features →",
  "         ┌─────────────┐",
  " batch   │ x x x x x x │ ◄ layer norm:",
  "   ↓     │ x x x x x x │   each row",
  "         │ x x x x x x │",
  "         └─────────────┘",
  "           ▲ batch norm: each column",
 ],
}

if __name__ == "__main__":
    for slug, lines in D.items():
        assert 3 <= len(lines) <= 16, slug
        for ln in lines:
            assert len(ln) <= 46, (slug, len(ln), ln)
        path = PROBLEMS / f"{slug}.yaml"
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^diagram:\n", "", text, flags=re.M)
        text = re.sub(r"^- .*\n", "", text, flags=re.M)  # only diagram items sit at column 0
        if not text.endswith("\n"):
            text += "\n"
        text += yaml.dump({"diagram": lines}, allow_unicode=True,
                          sort_keys=False, width=1000)
        path.write_text(text, encoding="utf-8")
    print("diagrams added to", len(D), "cards")
