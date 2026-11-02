"""Group 1: wrap formulas in prose fields with $...$ LaTeX (exact, single-hit edits)."""
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "problems"

E = {
 "adaboost": [
  ("Set the vote weight α = ½ log((1−ε)/ε) —", r"Set the vote weight $\alpha = \frac{1}{2}\log\frac{1-\epsilon}{\epsilon}$ —"),
  ("Multiply each weight by exp(−α y h(x)) and", r"Multiply each weight by $\exp(-\alpha\, y\, h(x))$ and"),
  ("Repeat for T rounds and store (α, stump) pairs", r"Repeat for $T$ rounds and store $(\alpha, \mathrm{stump})$ pairs"),
  ("Predict the sign of the α-weighted sum", r"Predict the sign of the $\alpha$-weighted sum"),
  ("Number of rounds T —", "Number of rounds $T$ —"),
  ("Labels must be −1 and +1 in this code", "Labels must be $-1$ and $+1$ in this code"),
  ("sample by exp(−α y h(x)):", r"sample by $\exp(-\alpha\, y\, h(x))$:"),
 ],
 "batch-normalization": [
  ("the learned γ and β let the layer", r"the learned $\gamma$ and $\beta$ let the layer"),
  ("Divide the centred input by √(var + ε) — ε avoids", r"Divide the centred input by $\sqrt{\mathrm{var}+\epsilon}$ — $\epsilon$ avoids"),
  ("Apply γ and β — restores", r"Apply $\gamma$ and $\beta$ — restores"),
  ("ε (numerical floor", r"$\epsilon$ (numerical floor"),
  ("Why learn γ and β?", r"Why learn $\gamma$ and $\beta$?"),
  ("represent; γ and β let the network", r"represent; $\gamma$ and $\beta$ let the network"),
 ],
 "clip-contrastive-loss": [
  ("Form logits = I Tᵀ / τ, an N × N matrix — every image against every caption, τ sharpens",
   r"Form logits = $I T^{\top} / \tau$, an $N \times N$ matrix — every image against every caption, $\tau$ sharpens"),
  ("Use labels 0 … N−1 as targets", r"Use labels $0 \ldots N-1$ as targets"),
  ("all N − 1 other", "all $N-1$ other"),
  ("a separate N-way classification", "a separate $N$-way classification"),
  ("small τ makes", r"small $\tau$ makes"),
  ("Temperature τ (fixed or learned)", r"Temperature $\tau$ (fixed or learned)"),
  ("Batch size N (number of negatives)", "Batch size $N$ (number of negatives)"),
  ("Embedding dimension D", "Embedding dimension $D$"),
  ("Here τ is a constant", r"Here $\tau$ is a constant"),
 ],
 "depthwise-separable-convolution": [
  ("own k×k filter and sum", r"own $k \times k$ filter and sum"),
  ("with a F×C matrix — the pointwise 1×1 step produces F maps", r"with a $F \times C$ matrix — the pointwise 1×1 step produces $F$ maps"),
  ("Output channels F of the pointwise layer", "Output channels $F$ of the pointwise layer"),
  ("It uses C·k² spatial weights plus F·C mixing weights instead of F·C·k² for",
   r"It uses $C \cdot k^2$ spatial weights plus $F \cdot C$ mixing weights instead of $F \cdot C \cdot k^2$ for"),
 ],
 "entropy-kl-divergence": [
  ("average surprise of samples from p when", "average surprise of samples from $p$ when"),
  ("Cross-entropy H(p, q) is the average surprise if you use q's", "Cross-entropy $H(p, q)$ is the average surprise if you use $q$'s"),
  ("how badly q approximates p. Because H(p) is fixed", "how badly $q$ approximates $p$. Because $H(p)$ is fixed"),
  ("only when q equals p almost everywhere", "only when $q$ equals $p$ almost everywhere"),
  ("KL is asymmetric and infinite when q is zero where p is not", "KL is asymmetric and infinite when $q$ is zero where $p$ is not"),
  ("Drop zero-probability terms of p — 0·log 0 is defined as 0", r"Drop zero-probability terms of $p$ — $0 \cdot \log 0$ is defined as $0$"),
  ("Entropy is the negative sum of p·log p — average surprise under p", r"Entropy is the negative sum of $p \cdot \log p$ — average surprise under $p$"),
  ("Cross-entropy swaps in log q — cost of coding p's samples with q", r"Cross-entropy swaps in $\log q$ — cost of coding $p$'s samples with $q$"),
  ("(forward KL(p‖q) or reverse KL(q‖p))", r"(forward $\mathrm{KL}(p\|q)$ or reverse $\mathrm{KL}(q\|p)$)"),
  ("Smoothing epsilon added to q to avoid", "Smoothing epsilon added to $q$ to avoid"),
  ("q must be positive wherever p is,", "$q$ must be positive wherever $p$ is,"),
  ("Forward KL(p‖q) is mass-covering, so q spreads over all modes of p; reverse KL(q‖p) is mode-seeking, so q locks onto",
   r"Forward $\mathrm{KL}(p\|q)$ is mass-covering, so $q$ spreads over all modes of $p$; reverse $\mathrm{KL}(q\|p)$ is mode-seeking, so $q$ locks onto"),
 ],
 "gan-minimax-loss": [
  ("minimising log(1 − D(G(z))) or by", r"minimising $\log(1 - D(G(z)))$ or by"),
  ("maximising log D(G(z)).", r"maximising $\log D(G(z))$."),
  ("maximise log D(G(z)) has the same", r"maximise $\log D(G(z))$ has the same"),
  ("Let D output a logit and treat σ(logit) as P(real)", r"Let $D$ output a logit and treat $\sigma(\mathrm{logit})$ as $P(\mathrm{real})$"),
  ("D loss is softplus", "$D$ loss is softplus"),
  ("the minimax game, G minimises log(1 − D(G(z)))", r"the minimax game, $G$ minimises $\log(1 - D(G(z)))$"),
  ("the saturating one vanishes when D is confident", "the saturating one vanishes when $D$ is confident"),
  ("it does not model D or G networks", "it does not model $D$ or $G$ networks"),
  ("When D confidently rejects a fake, D(G(z)) is near 0 and log(1 − D) is flat there, so the gradient to G is almost zero when G is worst.",
   r"When $D$ confidently rejects a fake, $D(G(z))$ is near $0$ and $\log(1 - D)$ is flat there, so the gradient to $G$ is almost zero when $G$ is worst."),
  ("a few outputs that fool D, so", "a few outputs that fool $D$, so"),
  ("Adam β₁ for each network", r"Adam $\beta_1$ for each network"),
  ("D should be updated with fake samples detached from G's graph", "$D$ should be updated with fake samples detached from $G$'s graph"),
  ("A very strong D starves G of gradient", "A very strong $D$ starves $G$ of gradient"),
 ],
 "gradient-clipping": [
  ("Form the factor min(1, τ / (norm + ε)) —", r"Form the factor $\min(1,\, \tau / (\mathrm{norm} + \epsilon))$ —"),
  ("Max norm τ (or clip value κ)", r"Max norm $\tau$ (or clip value $\kappa$)"),
 ],
 "hyperparameter-search": [
  ("a grid with m values", "a grid with $m$ values"),
  ("yields only m distinct values", "yields only $m$ distinct values"),
  ("while n random trials yield n distinct values.", "while $n$ random trials yield $n$ distinct values."),
 ],
 "knowledge-distillation": [
  ("Multiplying the soft loss by T² keeps", "Multiplying the soft loss by $T^2$ keeps"),
  ("scale like 1/T².", "scale like $1/T^2$."),
  ("log-probabilities at temperature T —", "log-probabilities at temperature $T$ —"),
  ("cross-entropy of the student at T = 1 —", "cross-entropy of the student at $T = 1$ —"),
  ("Mix the two with weight α and scale the soft term by T² —", r"Mix the two with weight $\alpha$ and scale the soft term by $T^2$ —"),
  ('"Temperature T"', '"Temperature $T$"'),
  ("Mixing weight α between", r"Mixing weight $\alpha$ between"),
  ("Forgetting the T² factor", "Forgetting the $T^2$ factor"),
  ("Why multiply the soft loss by T²?", "Why multiply the soft loss by $T^2$?"),
  ("scale like 1/T², so T² keeps the soft", "scale like $1/T^2$, so $T^2$ keeps the soft"),
  ("KL(teacher ‖ student), so", r"$\mathrm{KL}(\mathrm{teacher} \,\|\, \mathrm{student})$, so"),
 ],
 "linear-regression": [
  ("Add a column of ones to X —", "Add a column of ones to $X$ —"),
  ("Form XᵀX and Xᵀy —", r"Form $X^{\top}X$ and $X^{\top}y$ —"),
  ("Solve the linear system for w —", "Solve the linear system for $w$ —"),
  ("Predict with Xw —", "Predict with $Xw$ —"),
  ("ridge adds λ)", r"ridge adds $\lambda$)"),
  ("Collinear features make XᵀX singular", r"Collinear features make $X^{\top}X$ singular"),
  ("A high R² on training data", "A high $R^2$ on training data"),
  ("instead of inverting XᵀX?", r"instead of inverting $X^{\top}X$?"),
  ("It adds λI to XᵀX, so", r"It adds $\lambda I$ to $X^{\top}X$, so"),
 ],
 "matrix-factorization-recommender": [
  ("non-convex in (U, V) but convex", "non-convex in $(U, V)$ but convex"),
  ("Rank k — number of latent factors", "Rank $k$ — number of latent factors"),
  ("Regularisation λ — controls", r"Regularisation $\lambda$ — controls"),
  ("R is a dense array with a 0/1 mask M here", "$R$ is a dense array with a 0/1 mask $M$ here"),
  ("Plain λ is not scaled", r"Plain $\lambda$ is not scaled"),
  ("(the weighted-λ variant does)", r"(the weighted-$\lambda$ variant does)"),
  ("optimising U and V jointly?", "optimising $U$ and $V$ jointly?"),
 ],
 "momentum-sgd": [
  ("Update v ← μ·v + g —", r"Update $v \leftarrow \mu \cdot v + g$ —"),
  ("direction: v for classical, g + μ·v for Nesterov —", r"direction: $v$ for classical, $g + \mu \cdot v$ for Nesterov —"),
  ("Move the weights by −η times that direction", r"Move the weights by $-\eta$ times that direction"),
  ('"Learning rate η"', r'"Learning rate $\eta$"'),
  ('"Momentum μ"', r'"Momentum $\mu$"'),
  ("when raising μ", r"when raising $\mu$"),
  ("direction, g + μ·v after updating v, which", r"direction, $g + \mu \cdot v$ after updating $v$, which"),
  ("geometric series in μ, so the effective step is larger by 1 / (1 − μ).",
   r"geometric series in $\mu$, so the effective step is larger by $1 / (1 - \mu)$."),
 ],
 "ppo-clipped-objective": [
  ("Form the unclipped term ρ·Â and the clipped term clip(ρ)·Â —",
   r"Form the unclipped term $\rho \cdot \hat{A}$ and the clipped term $\mathrm{clip}(\rho) \cdot \hat{A}$ —"),
  ("Clip range ε (smaller", r"Clip range $\epsilon$ (smaller"),
  ("settings (γ and λ) and", r"settings ($\gamma$ and $\lambda$) and"),
  ("the sign of Â decides", r"the sign of $\hat{A}$ decides"),
  ("above 1 plus ε, and", r"above 1 plus $\epsilon$, and"),
  ("below 1 minus ε.", r"below 1 minus $\epsilon$."),
 ],
 "receptive-field": [
  ("Each layer looks at a k-wide window", "Each layer looks at a $k$-wide window"),
  ("adds (k − 1) steps of", "adds $(k - 1)$ steps of"),
  ("(k − 1) times the product of all earlier", "$(k - 1)$ times the product of all earlier"),
  ("add (k − 1) · jump to the field", r"add $(k - 1) \cdot \mathrm{jump}$ to the field"),
  ("Kernel size k of each layer", "Kernel size $k$ of each layer"),
  ("Stride s of each layer", "Stride $s$ of each layer"),
  ("dilation replaces k by", "dilation replaces $k$ by"),
  ("adds (k − 1) times the product of earlier", "adds $(k - 1)$ times the product of earlier"),
 ],
 "reward-model-bradley-terry": [
  ("Take the difference d = r_c − r_r —", r"Take the difference $d = r_c - r_r$ —"),
  ("Apply −log σ(d), computed as log(1 + e^(−d)) —", r"Apply $-\log \sigma(d)$, computed as $\log(1 + e^{-d})$ —"),
  ("Why does the loss equal log(1 + e^(−d))?", r"Why does the loss equal $\log(1 + e^{-d})$?"),
  ("−log σ(d) = log(1 + e^(−d)). Computing", r"$-\log \sigma(d) = \log(1 + e^{-d})$. Computing"),
 ],
 "softmax-cross-entropy": [
  ("collapses to the simple form p − y,", "collapses to the simple form $p - y$,"),
  ("p − onehot(y) that does not", r"$p - \mathrm{onehot}(y)$ that does not"),
  ("Compute log Σ exp(z) − z_y per row", r"Compute $\log \sum \exp(z) - z_y$ per row"),
  ("so no log(0)", r"so no $\log(0)$"),
  ("from p and divide by n —", "from $p$ and divide by $n$ —"),
  ("gradient just p − y?", "gradient just $p - y$?"),
 ],
 "top-k-top-p-sampling": [
  ("to the k most likely tokens, or", "to the $k$ most likely tokens, or"),
  ("probability reaches p, then", "probability reaches $p$, then"),
  ("beyond the k-th to −∞ —", r"beyond the $k$-th to $-\infty$ —"),
  ("at least p — keeps the smallest nucleus reaching p", "at least $p$ — keeps the smallest nucleus reaching $p$"),
  ("zero weight to −∞", r"zero weight to $-\infty$"),
  ("Top-p threshold in (0, 1]", "Top-p threshold in $(0, 1]$"),
  ('"Temperature τ"', r'"Temperature $\tau$"'),
  ("reaching probability mass p, so", "reaching probability mass $p$, so"),
 ],
 "vision-transformer-patch-embed": [
  ("treating every P×P patch", r"treating every $P \times P$ patch"),
  ("grid of P×P patches — turns H×W pixels into N = HW/P² tokens",
   r"grid of $P \times P$ patches — turns $H \times W$ pixels into $N = HW/P^2$ tokens"),
  ("vector of length C·P² —", r"vector of length $C \cdot P^2$ —"),
  ("shared matrix W and add a bias — linear projection to model width D",
   "shared matrix $W$ and add a bias — linear projection to model width $D$"),
  ("from H·W to HW/P², which", r"from $H \cdot W$ to $HW/P^2$, which"),
  ("kernel and stride P, so", "kernel and stride $P$, so"),
  ("one P×P×C block", r"one $P \times P \times C$ block"),
  ("with the stride P making", "with the stride $P$ making"),
  ("The token count N grows 4×", "The token count $N$ grows 4×"),
  ('"Patch size P"', '"Patch size $P$"'),
  ('"Embedding width D"', '"Embedding width $D$"'),
  ("H and W must be divisible by P here", "$H$ and $W$ must be divisible by $P$ here"),
  ("(C, P, P versus P, P, C)", "($C, P, P$ versus $P, P, C$)"),
 ],
}

bad = 0
count = 0
for slug, edits in E.items():
    f = P / f"{slug}.yaml"
    text = f.read_text(encoding="utf-8")
    for old, new in edits:
        n = text.count(old)
        if n == 1:
            at = text.index(old)
            line = text[text.rfind("\n", 0, at) + 1:text.find("\n", at)]
            if line.lstrip().startswith(('- "', '"')) or ': "' in line:
                new = new.replace("\\", "\\\\")
            text = text.replace(old, new)
            count += new.count("$") // 2
        elif n == 0 and (new in text or new.replace("\\", "\\\\") in text):
            pass
        else:
            bad += 1
            print("MISSING/AMBIGUOUS", slug, n, old)
    f.write_text(text, encoding="utf-8", newline="")
print("g1 inline math done;", bad, "problems;", count, "formulas")
