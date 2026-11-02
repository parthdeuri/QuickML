"""Group 3: wrap formulas in prose fields with $...$ LaTeX (exact, single-hit edits)."""
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "problems"

E = {
 "autoencoder": [
  ("through a k-dimensional code", "through a $k$-dimensional code"),
  ("best k-dimensional", "best $k$-dimensional"),
  ("tanh: 1 − h²); identity is the constant 1", r"tanh: $1-h^2$); identity is the constant 1"),
  ("Bottleneck size k", "Bottleneck size $k$"),
  ("Activation function f (identity", "Activation function $f$ (identity"),
 ],
 "bias-variance-tradeoff": [
  ("around the mean prediction", "around the mean prediction"),
  ("E[ŷ]. The cross terms", r"$E[\hat{y}]$. The cross terms"),
  ("the deviation ŷ − E[ŷ] has zero mean, leaving exactly bias², variance",
   r"the deviation $\hat{y}-E[\hat{y}]$ has zero mean, leaving exactly $\mathrm{bias}^2$, variance"),
  ("Pick a known true function f and a test point x0 —", r"Pick a known true function $f$ and a test point $x_0$ —"),
  ("Fit the model on each and predict at x0 —", r"Fit the model on each and predict at $x_0$ —"),
  ("bias² from the mean prediction versus f(x0) —", r"$\mathrm{bias}^2$ from the mean prediction versus $f(x_0)$ —"),
  ("because the true f is unknown", "because the true $f$ is unknown"),
  ("Measured at one x0 only", "Measured at one $x_0$ only"),
  ("k in k-NN", "$k$ in k-NN"),
  ("The irreducible noise term σ² is", r"The irreducible noise term $\sigma^2$ is"),
 ],
 "cross-attention-encoder-decoder": [
  ("padded source columns to −∞ so", r"padded source columns to $-\infty$ so"),
  ("with Q·Kᵀ/√d_k —", r"with $QK^{\top}/\sqrt{d_k}$ —"),
  ("Mask padded source columns with −∞ —", r"Mask padded source columns with $-\infty$ —"),
  ("multiply by V —", "multiply by $V$ —"),
  ("Key/query dimension d_k (sets the scale √d_k)", r"Key/query dimension $d_k$ (sets the scale $\sqrt{d_k}$)"),
  ("Where do Q, K and V come from", "Where do $Q$, $K$ and $V$ come from"),
 ],
 "dpo-loss": [
  ("treats β·log(π/π_ref) as", r"treats $\beta\cdot\log(\pi/\pi_{\mathrm{ref}})$ as"),
  ("implicit reward up to β", r"implicit reward up to $\beta$"),
  ("difference and scale by β —", r"difference and scale by $\beta$ —"),
  ("Apply −log σ(margin) and", r"Apply $-\log\sigma(\mathrm{margin})$ and"),
  ("β (strength", r"$\beta$ (strength"),
  ("What does β control?", r"What does $\beta$ control?"),
  ("A larger β penalises", r"A larger $\beta$ penalises"),
  ("'It scales the log-ratio", "'It scales the log-ratio"),
 ],
 "feature-scaling": [
  ("z = (x − c) / s with", r"$z = (x-c)/s$ with"),
  ("Apply (x − centre) / scale to", r"Apply $(x-\mathrm{centre})/\mathrm{scale}$ to"),
  ("Invert with x = z·scale + centre when", r"Invert with $x = z\cdot\mathrm{scale}+\mathrm{centre}$ when"),
  ("such as 0 to 1 or −1 to 1", r"such as $0$ to $1$ or $-1$ to $1$"),
 ],
 "gaussian-process-regression": [
  ("Solve for α = K⁻¹y with", r"Solve for $\alpha = K^{-1}y$ with"),
  ("Solve L v = k* and subtract vᵀv from", r"Solve $Lv = k_*$ and subtract $v^{\top}v$ from"),
  ("to tune ℓ, s² and noise", r"to tune $\ell$, $s^2$ and noise"),
  ("Length scale ℓ —", r"Length scale $\ell$ —"),
  ("Signal variance s² —", r"Signal variance $s^2$ —"),
  ("Noise variance σₙ² —", r"Noise variance $\sigma_n^2$ —"),
  ("Skip it for large n, since", "Skip it for large $n$, since"),
  ("grows as n cubed", "grows as $n$ cubed"),
 ],
 "graph-attention-layer": [
  ("Project features H = XW —", "Project features $H = XW$ —"),
  ("Score every edge e_ij = LeakyReLU(a_iᵀ h_i + a_jᵀ h_j) — splitting a into",
   r"Score every edge $e_{ij} = \mathrm{LeakyReLU}(a_i^{\top} h_i + a_j^{\top} h_j)$ — splitting $a$ into"),
  ("Mask non-edges to −∞ (self-loops", r"Mask non-edges to $-\infty$ (self-loops"),
  ("attention weights α_ij that sum", r"attention weights $\alpha_{ij}$ that sum"),
  ("Output α H —", r"Output $\alpha H$ —"),
  ("Attention scores here are not symmetric: α_ij differs from α_ji", r"Attention scores here are not symmetric: $\alpha_{ij}$ differs from $\alpha_{ji}$"),
  ("Dense N×N scoring", r"Dense $N\times N$ scoring"),
  ("Setting non-edges to −∞ gives", r"Setting non-edges to $-\infty$ gives"),
  ("aᵀ[hᵢ‖hⱼ] = a_iᵀhᵢ + a_jᵀhⱼ, so scores", r"$a^{\top}[h_i\,\|\,h_j] = a_i^{\top}h_i + a_j^{\top}h_j$, so scores"),
  ("building N² concatenated", r"building $N^2$ concatenated"),
  ("(0.2 in the paper)", "($0.2$ in the paper)"),
 ],
 "k-means": [
  ("into k groups by", "into $k$ groups by"),
  ("You must also choose k in", "You must also choose $k$ in"),
  ("Pick k initial centres, for example k random points", "Pick $k$ initial centres, for example $k$ random points"),
  ("minimises J for fixed centres", "minimises $J$ for fixed centres"),
  ("minimises J for fixed assignments", "minimises $J$ for fixed assignments"),
  ("Number of clusters k (choose", "Number of clusters $k$ (choose"),
  ("with the lowest J)", "with the lowest $J$)"),
  ("because J never increases", "because $J$ never increases"),
  ("elbow in J versus k,", "elbow in $J$ versus $k$,"),
 ],
 "layer-normalization": [
  ("The learned γ and β restore", r"The learned $\gamma$ and $\beta$ restore"),
  ("divide by √(variance + ε) — ε prevents", r"divide by $\sqrt{\mathrm{variance}+\epsilon}$ — $\epsilon$ prevents"),
  ("Multiply by γ and add β —", r"Multiply by $\gamma$ and add $\beta$ —"),
  ("(divide by D)", "(divide by $D$)"),
  ("so ε controls", r"so $\epsilon$ controls"),
  ("Epsilon ε", r"Epsilon $\epsilon$"),
  ("Whether γ and β are learned", r"Whether $\gamma$ and $\beta$ are learned"),
 ],
 "logistic-regression": [
  ("(p − y) in place", r"$(p-y)$ in place"),
  ("Add a ones column to X —", "Add a ones column to $X$ —"),
  ("Start at w = 0 —", "Start at $w = 0$ —"),
  ("Compute p = σ(Xw) and the gradient Xᵀ(p − y) —", r"Compute $p = \sigma(Xw)$ and the gradient $X^{\top}(p-y)$ —"),
  ("Hessian with weights p(1 − p) —", r"Hessian with weights $p(1-p)$ —"),
  ("Solve H·Δ = g and update w —", r"Solve $H\cdot\Delta = g$ and update $w$ —"),
  ("Plain exp overflows for large negative z;", "Plain exp overflows for large negative $z$;"),
  ("Regularisation strength C (smaller C means", "Regularisation strength $C$ (smaller $C$ means"),
  ("and is convex in w;", "and is convex in $w$;"),
  ("Increasing that feature by one unit multiplies the odds by exp(coefficient)", r"Increasing that feature by one unit multiplies the odds by $\exp(\mathrm{coefficient})$"),
  ("The boundary σ(z) = 0.5 is wᵀx + b = 0, a", r"The boundary $\sigma(z) = 0.5$ is $w^{\top}x + b = 0$, a"),
 ],
 "mean-average-precision": [
  ("IoU ≥ threshold makes", r"$\mathrm{IoU}\geq\mathrm{threshold}$ makes"),
  ("Sum precision × recall increments", r"Sum precision $\times$ recall increments"),
  ("COCO mAP averages over IoU thresholds 0.50 to 0.95", r"COCO mAP averages over IoU thresholds $0.50$ to $0.95$"),
  ("single IoU threshold of 0.5;", r"single IoU threshold of $0.5$;"),
 ],
 "naive-bayes": [
  ("only needs the ranking of P(c | x), not", r"only needs the ranking of $P(c\,|\,x)$, not"),
  ("push the probabilities toward 0 or 1,", "push the probabilities toward $0$ or $1$,"),
  ("— the P(c) term", r"— the $P(c)$ term"),
  ("Compute log P(c) plus the sum", r"Compute $\log P(c)$ plus the sum"),
 ],
 "probability-calibration": [
  ("predictions of probability p, about a\n  fraction p are", "predictions of probability $p$, about a\n  fraction $p$ are"),
  ("E[y | p̂ = p] = p, so", r"$E[y\,|\,\hat{p}=p]=p$, so"),
  ("Cut [0, 1] into", "Cut $[0,1]$ into"),
  ("probability p, the observed positive rate is p;", "probability $p$, the observed positive rate is $p$;"),
 ],
 "residual-connections": [
  ("With\n  y = x + F(x), doing nothing just means F outputs zero,", r"With" + "\n  " + r"$y = x + F(x)$, doing nothing just means $F$ outputs zero,"),
  ("For y = x + F(x) the Jacobian is I + ∂F/∂x, so", r"For $y = x + F(x)$ the Jacobian is $I + \partial F/\partial x$, so"),
  ("zero-initialising F gives", "zero-initialising $F$ gives"),
  ("Compute the block's transformation F(x) with", "Compute the block's transformation $F(x)$ with"),
  ("Keep the original x untouched", "Keep the original $x$ untouched"),
  ("Return x + F(x) — the", "Return $x + F(x)$ — the"),
  ("for x + F(x); a broadcast", "for $x + F(x)$; a broadcast"),
 ],
 "rmsprop-adagrad": [
  ("Dividing by √v rescales", r"Dividing by $\sqrt{v}$ rescales"),
  ("Adagrad adds g² each step; RMSProp sets v = αv + (1 − α)g² —", r"Adagrad adds $g^2$ each step; RMSProp sets $v = \alpha v + (1-\alpha)g^2$ —"),
  ("and add ε — per-parameter scale, ε avoids", r"and add $\epsilon$ — per-parameter scale, $\epsilon$ avoids"),
  ("because v starts at zero", "because $v$ starts at zero"),
  ("ε placement", r"$\epsilon$ placement"),
  ("Learning rate η", r"Learning rate $\eta$"),
  ("Decay rate α for RMSProp (commonly 0.9 to 0.99)", r"Decay rate $\alpha$ for RMSProp (commonly $0.9$ to $0.99$)"),
  ("ε for numerical", r"$\epsilon$ for numerical"),
  ("effective step η/√G keeps", r"effective step $\eta/\sqrt{G}$ keeps"),
  ("after the square root, √v + ε; putting it inside the root, √(v + ε), is", r"after the square root, $\sqrt{v}+\epsilon$; putting it inside the root, $\sqrt{v+\epsilon}$, is"),
  ("which matters when v is tiny", "which matters when $v$ is tiny"),
 ],
 "svd": [
  ("as A = U S Vᵀ with orthonormal U and V and", r"as $A = USV^{\top}$ with orthonormal $U$ and $V$ and"),
  ("non-negative singular values in S.", "non-negative singular values in $S$."),
  ("approximation of A.", "approximation of $A$."),
  ("The SVD names the\n  ellipsoid's axes: V gives", "The SVD names the\n  ellipsoid's axes: $V$ gives"),
  ("and U gives the output", "and $U$ gives the output"),
  ("AᵀA = V S² Vᵀ, so the right singular vectors are eigenvectors of AᵀA\n  and", r"$A^{\top}A = VS^2V^{\top}$, so the right singular vectors are eigenvectors of $A^{\top}A$" + "\n  " + "and"),
  ("no rank-k matrix is closer to A than", "no rank-$k$ matrix is closer to $A$ than"),
  ("keeps\n  the top k singular", "keeps\n  the top $k$ singular"),
  ("Form AᵀA —", r"Form $A^{\top}A$ —"),
  ("Compute U = A V / s column", r"Compute $U = AV/s$ column"),
  ("Truncate to the top k triplets", "Truncate to the top $k$ triplets"),
  ("This code assumes m ≥ n and", r"This code assumes $m\geq n$ and"),
  ("zero columns in U", "zero columns in $U$"),
  ("Forming AᵀA squares", r"Forming $A^{\top}A$ squares"),
  ("Rank k —", "Rank $k$ —"),
  ("eigenvectors of AᵀA and the left ones those of AAᵀ, with", r"eigenvectors of $A^{\top}A$ and the left ones those of $AA^{\top}$, with"),
  ("Truncating the SVD to rank k gives the closest rank-k matrix to A in", "Truncating the SVD to rank $k$ gives the closest rank-$k$ matrix to $A$ in"),
  ("squared singular values over n − 1 are", r"squared singular values over $n-1$ are"),
  ("It works with A directly, avoiding the squared condition number of AᵀA, and", r"It works with $A$ directly, avoiding the squared condition number of $A^{\top}A$, and"),
 ],
 "transposed-convolution": [
  ("of size (H − 1)·s + k —", r"of size $(H-1)\cdot s+k$ —"),
  ("one k×k stamp", r"one $k\times k$ stamp"),
  ("at offset (s·i, s·j) —", r"at offset $(s\cdot i,\, s\cdot j)$ —"),
  ("Crop p rows", "Crop $p$ rows"),
  ("(H − 1)·s − 2p + k along", r"$(H-1)\cdot s-2p+k$ along"),
 ],
}

bad = 0
count = 0
for slug, edits in E.items():
    f = P / f"{slug}.yaml"
    text = f.read_text(encoding="utf-8")
    for old, new in edits:
        if old == new:
            continue
        n = text.count(old)
        if n == 1:
            # a double-quoted YAML scalar needs every backslash doubled
            at = text.index(old)
            line = text[text.rfind("\n", 0, at) + 1:text.find("\n", at)]
            if line.lstrip().startswith(('- "', '"')) or ': "' in line:
                new = new.replace("\\", "\\\\")
            text = text.replace(old, new)
            count += new.count("$") // 2
        elif n == 0 and (new in text or new.replace("\\", "\\\\") in text):
            pass  # already applied
        else:
            bad += 1
            print("MISSING/AMBIGUOUS", slug, n, old)
    f.write_text(text, encoding="utf-8", newline="")
print("group 3 inline math done;", count, "formulas;", bad, "problems")
