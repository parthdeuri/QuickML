"""Card diagrams (LLM batch B): fine-tuning, RLHF, parallelism, serving and scaling."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagram_lib import G, render  # noqa: E402


def sft_loss_masking(g):
    toks = ["What", "is", "2+2?", "It", "is", "4", ".", "pad"]
    fills = {}
    texts = {}
    for c, t in enumerate(toks):
        masked = c < 3 or c == 7
        fills[(0, c)] = "op" if masked else "io"
        fills[(1, c)] = "op" if masked else "hi"
        texts[(0, c)] = t
        texts[(1, c)] = "-100" if masked else "loss"
    g.grid(10, 72, 2, 8, 10, fills, texts)
    g.text(25, 77, "prompt", "muted")
    g.text(60, 77, "response", "muted")
    g.text(85, 77, "pad", "muted")
    g.text(50, 47, "only response tokens are scored", "text")
    mf = {}
    for r in range(6):
        for c in range(6):
            if c <= r and (r < 3) == (c < 3):
                mf[(r, c)] = "cool" if r < 3 else "hi"
    g.grid(10, 38, 6, 6, 6, mf)
    g.text(28, 42, "packed rows: attention mask", "muted", fs=7.5)
    g.text(70, 22, "block-diagonal\nand causal:\nexample 2 cannot\nsee example 1", "muted", fs=7.5)


def qlora_nf4_quantization(g):
    g.n("x", "x", 50, 92, 12, kind="io")
    g.n("w4", "W  4-bit NF4\n(frozen)", 27, 78, 30, 11, kind="cool")
    g.n("dq", "dequantise\nto 16-bit", 27, 62, 30, 11)
    g.n("mm", "x · W", 27, 47, 24)
    g.a(("w4", "b"), ("dq", "t"))
    g.a(("dq", "b"), ("mm", "t"))
    g.text(41, 54, "on the fly", "muted", fs=7)
    g.p([("x", "b"), (50, 87), (6, 87), (6, 47), ("mm", "l")])
    g.n("A", "A  16-bit\nLoRA", 76, 76, 26, 11, kind="param")
    g.n("B", "B  16-bit\nLoRA", 76, 59, 26, 11, kind="param")
    g.p([(50, 87), (76, 87), ("A", "t")])
    g.a(("A", "b"), ("B", "t"))
    g.n("add", "+", 50, 32, 8, circle=True, fs=11)
    g.p([("mm", "b"), (27, 32), ("add", "l")])
    g.p([("B", "b"), (76, 32), ("add", "r")])
    g.n("y", "y", 50, 18, 12, kind="io")
    g.a(("add", "b"), ("y", "t"))
    g.text(50, 6, "gradients reach only A and B", "muted", fs=7.5)


def adapter_layers(g):
    cx = 36
    g.n("x", "x", cx, 90, 14, kind="io")
    g.n("sub", "Attention / FFN\n(frozen)", cx, 76, 36, 11)
    g.n("dn", "down  d → r", cx, 60, 30, kind="param")
    g.n("act", "ReLU", cx, 49, 16)
    g.n("up", "up  r → d", cx, 38, 30, kind="param")
    g.n("add", "+", cx, 26, 8, circle=True, fs=11)
    g.n("ln", "LayerNorm", cx, 14, 28)
    g.a(("x", "b"), ("sub", "t"))
    g.a(("sub", "b"), ("dn", "t"))
    g.a(("dn", "b"), ("act", "t"))
    g.a(("act", "b"), ("up", "t"))
    g.a(("up", "b"), ("add", "t"))
    g.a(("add", "b"), ("ln", "t"))
    g.p([(cx, 69), (68, 69), (68, 26), ("add", "r")], kind="loss", dashed=True)
    g.text(85, 46, "adapter skip", "muted", fs=7.5)
    g.text(84, 62, "small\nbottleneck\nr ≪ d", "muted", fs=7.5)
    g.text(50, 4, "only the adapter is trained", "muted", fs=7.5)


def rlhf_kl_penalty(g):
    g.n("x", "prompt x", 50, 92, 22, kind="io")
    g.n("pol", "policy πθ\n(trained)", 28, 78, 32, 11, kind="param")
    g.n("y", "sampled response y", 28, 62, 40, kind="io")
    g.n("rm", "reward model\nr(x, y)", 28, 46, 32, 11, kind="param")
    g.n("ref", "reference π_ref\n(frozen)", 72, 62, 34, 11, kind="cool")
    g.n("kl", "KL = log πθ(y)\n− log π_ref(y)", 72, 46, 34, 11, kind="loss")
    g.n("fin", "reward − β · KL", 50, 28, 56, kind="hi")
    g.n("ppo", "PPO update of policy", 50, 14, 46)
    g.p([("x", "l"), (28, 92), ("pol", "t")])
    g.a(("pol", "b"), ("y", "t"))
    g.text(38, 70, "samples", "muted", fs=7)
    g.a(("y", "b"), ("rm", "t"))
    g.a(("y", "r"), ("ref", "l"))
    g.text(50, 70, "same y", "muted", fs=7)
    g.a(("ref", "b"), ("kl", "t"))
    g.p([("pol", "r"), (95, 78), (95, 46), ("kl", "r")])
    g.a(("rm", "b"), (28, 33.5))
    g.a(("kl", "b"), (72, 33.5))
    g.a(("fin", "b"), ("ppo", "t"))
    g.text(50, 4, "KL keeps the policy near the reference", "muted", fs=7.5)


def instruction_tuning_chat_templates(g):
    for nid, t, cx in [("f1", "system", 16), ("f2", "user", 40), ("f3", "assistant", 64)]:
        g.n(nid, t, cx, 80, 22, kind="io")
    g.n("tm", "chat template", 40, 66, 52, kind="param")
    for nid in ("f1", "f2", "f3"):
        g.a((nid, "b"), ("tm", "t"))
    g.n("s1", "<|system|> You are helpful", 40, 50, 60)
    g.n("s2", "<|user|> What is 2+2?", 40, 38, 60)
    g.n("s3", "<|assistant|> 4 <|end|>", 40, 26, 60, kind="hi")
    g.a(("tm", "b"), ("s1", "t"))
    g.n("ls", "loss on\nthis span", 87, 26, 22, 11, kind="loss")
    g.a(("ls", "l"), ("s3", "r"))
    g.text(87, 44, "masked", "muted", fs=7.5)
    g.text(50, 10, "same template at train and inference", "muted", fs=7.5)


def tensor_parallel_matmul(g):
    g.n("X", "X (same on both)", 50, 92, 40, kind="io")
    sub = "₁₂"
    for k, cx in [(1, 26), (2, 74)]:
        s = sub[k - 1]
        g.n(f"A{k}", f"GPU {k - 1}: A{s}\ncolumn slice", cx, 76, 36, 11, kind="param")
        g.n(f"H{k}", f"H{s} = act(X A{s})", cx, 60, 36)
        g.n(f"B{k}", f"B{s} row slice\nY{s} = H{s} B{s}", cx, 44, 36, 11, kind="param")
        g.p([("X", "b"), (50, 86), (cx, 86), (f"A{k}", "t")])
        g.a((f"A{k}", "b"), (f"H{k}", "t"))
        g.a((f"H{k}", "b"), (f"B{k}", "t"))
    g.n("ar", "all-reduce: Y = Y₁ + Y₂", 50, 26, 54, kind="hi")
    g.a(("B1", "b"), (26, 30.5))
    g.a(("B2", "b"), (74, 30.5))
    g.n("Y", "Y", 50, 12, 12, kind="io")
    g.a(("ar", "b"), ("Y", "t"))
    g.text(50, 3, "one all-reduce per block, nothing before", "muted", fs=7.5)


def pipeline_parallel_schedule(g):
    fills, texts = {}, {}
    for s in range(4):
        for t in range(7):
            m = t - s
            if 0 <= m < 4:
                fills[(s, t)] = "io"
                texts[(s, t)] = str(m + 1)
            else:
                fills[(s, t)] = "loss"
    g.grid(22, 64, 4, 7, 10, fills, texts, fs=8)
    for s in range(4):
        g.text(12, 64 - 10 * s - 5, f"stage {s}", "muted", fs=7)
    g.text(57, 70, "time →", "muted")
    g.text(50, 17, "number = micro-batch, red = idle bubble", "text", fs=7.5)
    g.text(50, 9, "bubble fraction = (p − 1) / (m + p − 1)", "muted", fs=7.5)


def continuous_batching(g):
    def draw(y0, plan, title):
        g.text(50, y0 + 4, title, "text")
        fills, texts = {}, {}
        for r, segs in enumerate(plan):
            c = 0
            for i, (name, ln) in enumerate(segs):
                for _ in range(ln):
                    if name is None:
                        fills[(r, c)] = "loss"
                    else:
                        fills[(r, c)] = "io" if i % 2 == 0 else "cool"
                        texts[(r, c)] = name
                    c += 1
            g.text(15, y0 - 8 * r - 4, f"slot {r + 1}", "muted", fs=7)
        g.grid(26, y0, 4, 8, 8, fills, texts)

    draw(92, [[("A", 3), (None, 5)], [("B", 8)], [("C", 5), (None, 3)], [("D", 2), (None, 6)]],
         "static batch: slots idle")
    draw(48, [[("A", 3), ("E", 5)], [("B", 8)], [("C", 5), ("F", 3)], [("D", 2), ("G", 6)]],
         "continuous: slots refill")
    g.text(60, 7, "time (decode steps) →", "muted", fs=7.5)


def scaling_laws(g):
    g.n("E", "floor E\n(irreducible)", 18, 88, 30, 11)
    g.n("Mt", "model term\nA / Nᵅ", 50, 88, 30, 11, kind="param")
    g.n("Dt", "data term\nB / Dᵝ", 82, 88, 30, 11, kind="cool")
    g.n("L", "loss L(N, D)", 50, 70, 36, kind="loss")
    for k in ("E", "Mt", "Dt"):
        g.a((k, "b"), ("L", "t"))
    g.n("mn", "minimise L\nsubject to 6ND = C", 38, 52, 54, 12, kind="hi")
    g.n("C", "budget\nC", 88, 52, 20, 12, kind="io")
    g.a(("L", "b"), ("mn", "t"))
    g.a(("C", "l"), ("mn", "r"))
    g.n("N", "N*  params", 26, 32, 34, kind="param")
    g.n("D", "D*  tokens", 70, 32, 34, kind="cool")
    g.a(("mn", "b"), ("N", "t"))
    g.a(("mn", "b"), ("D", "t"))
    g.text(48, 18, "bigger C: grow N and D together", "text", fs=7.5)
    g.text(50, 8, "too big a model on too little data wastes compute", "muted", fs=7)


DIAGRAMS = {
    "sft-loss-masking": (84, sft_loss_masking),
    "qlora-nf4-quantization": (98, qlora_nf4_quantization),
    "adapter-layers": (96, adapter_layers),
    "rlhf-kl-penalty": (100, rlhf_kl_penalty),
    "instruction-tuning-chat-templates": (88, instruction_tuning_chat_templates),
    "tensor-parallel-matmul": (100, tensor_parallel_matmul),
    "pipeline-parallel-schedule": (78, pipeline_parallel_schedule),
    "continuous-batching": (100, continuous_batching),
    "scaling-laws": (96, scaling_laws),
}

if __name__ == "__main__":
    render(DIAGRAMS, sys.argv[1:] or None)
