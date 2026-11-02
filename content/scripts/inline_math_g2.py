"""Group 2: wrap formulas in prose fields with $...$ LaTeX (exact, single-hit edits)."""
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "problems"

E = {
 "autocorrelation-acf-pacf": [
  ("AR(p) process shows a PACF that cuts off after lag p while", r"AR($p$) process shows a PACF that cuts off after lag $p$ while"),
  ("The PACF at lag k is the last", "The PACF at lag $k$ is the last"),
  ("coefficient φₖₖ of the best order-k autoregression, and the", r"coefficient $\phi_{kk}$ of the best order-$k$ autoregression, and the"),
  ("builds order k from order k − 1 in O(k)", r"builds order $k$ from order $k-1$ in O(k)"),
  ("divisor of n at every lag", "divisor of $n$ at every lag"),
  ("scale free in [−1, 1]", r"scale free in $[-1, 1]$"),
  ("about two over root n —", "about two over root $n$ —"),
  ("max lag K — a common rule is a fraction of n, such as n/4", "max lag $K$ — a common rule is a fraction of $n$, such as $n/4$"),
  ("Divisor n versus n − k — n keeps", r"Divisor $n$ versus $n-k$ — $n$ keeps"),
  ("An AR(p) has a PACF that cuts off after lag p and an ACF that decays; an MA(q) has an ACF that cuts off after lag q and",
   "An AR($p$) has a PACF that cuts off after lag $p$ and an ACF that decays; an MA($q$) has an ACF that cuts off after lag $q$ and"),
  ("What does the PACF at lag k measure?", "What does the PACF at lag $k$ measure?"),
  ("between yₜ and yₜ₋ₖ after linearly", r"between $y_t$ and $y_{t-k}$ after linearly"),
  ("last coefficient of an order-k autoregression.", "last coefficient of an order-$k$ autoregression."),
  ("Why divide by n and not n − k in the autocovariance?", r"Why divide by $n$ and not $n-k$ in the autocovariance?"),
  ("Dividing by n guarantees", "Dividing by $n$ guarantees"),
  ("needs; n − k is less biased", r"needs; $n-k$ is less biased"),
 ],
 "beam-search": [
  ("keeping the B highest-scoring", "keeping the $B$ highest-scoring"),
  ("Keeping the top B prefixes is", "Keeping the top $B$ prefixes is"),
  ("exact whenever B is at least", "exact whenever $B$ is at least"),
  ("For smaller B it is only", "For smaller $B$ it is only"),
  ("Sort all B·|V| candidates and keep the best B —", r"Sort all $B\cdot|V|$ candidates and keep the best $B$ —"),
  ("Beam width B (larger", "Beam width $B$ (larger"),
  ("It only keeps B prefixes", "It only keeps $B$ prefixes"),
  ("only when B covers", "only when $B$ covers"),
 ],
 "concept-drift": [
  ("changes in P(y | x) with stable", r"changes in $P(y\,|\,x)$ with stable"),
  ("outer edges to ±∞ —", r"outer edges to $\pm\infty$ —"),
  ("change in P(x) and", "change in $P(x)$ and"),
  ("change in P(y | x) and needs", r"change in $P(y\,|\,x)$ and needs"),
 ],
 "diffusion-forward-process": [
  ("jump straight to any step t with", "jump straight to any step $t$ with"),
  ("pick a random t and corrupt x₀ in", r"pick a random $t$ and corrupt $x_0$ in"),
  ("simulating t steps", "simulating $t$ steps"),
  ("Choose a noise schedule β₁ … β_T —", r"Choose a noise schedule $\beta_1 \ldots \beta_T$ —"),
  ("Form ᾱₜ as the running product of (1 − βₜ) — the fraction of signal variance that survives to step t",
   r"Form $\bar{\alpha}_t$ as the running product of $(1-\beta_t)$ — the fraction of signal variance that survives to step $t$"),
  ("Draw ε ~ N(0, I) —", r"Draw $\epsilon \sim \mathcal{N}(0, I)$ —"),
  ("Output √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε — a sample of q(xₜ | x₀) in one step",
   r"Output $\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$ — a sample of $q(x_t\,|\,x_0)$ in one step"),
  ("Optionally iterate √(1−βₜ) x + √βₜ ε — the same",
   r"Optionally iterate $\sqrt{1-\beta_t}\,x + \sqrt{\beta_t}\,\epsilon$ — the same"),
  ("Each step maps x to a Gaussian with mean √(1−β)·x and variance β. A",
   r"Each step maps $x$ to a Gaussian with mean $\sqrt{1-\beta}\,x$ and variance $\beta$. A"),
  ("so the mean shrinks by √(1−βₜ) each step and the", r"so the mean shrinks by $\sqrt{1-\beta_t}$ each step and the"),
  ("variance obeys v ← (1−βₜ)·v + βₜ. Solving", r"variance obeys $v \leftarrow (1-\beta_t)\,v + \beta_t$. Solving"),
  ("√ᾱₜ x₀ and variance 1 − ᾱₜ, and variance one", r"$\sqrt{\bar{\alpha}_t}\,x_0$ and variance $1-\bar{\alpha}_t$, and variance one"),
  ("Why can we sample xₜ directly from x₀?", r"Why can we sample $x_t$ directly from $x_0$?"),
  ("composing t of them", "composing $t$ of them"),
  ("mean √ᾱₜ x₀ and variance 1 − ᾱₜ.", r"mean $\sqrt{\bar{\alpha}_t}\,x_0$ and variance $1-\bar{\alpha}_t$."),
  ("Why scale the signal by √(1−βₜ) instead", r"Why scale the signal by $\sqrt{1-\beta_t}$ instead"),
  ("predicts the noise ε from xₜ and t, which is equivalent to estimating x₀ using",
   r"predicts the noise $\epsilon$ from $x_t$ and $t$, which is equivalent to estimating $x_0$ using"),
  ("Number of steps T", "Number of steps $T$"),
  ("Schedule endpoints β₁ and β_T, or", r"Schedule endpoints $\beta_1$ and $\beta_T$, or"),
  ("usually to [−1, 1]", r"usually to $[-1, 1]$"),
  ("array position t here is step t+1 of the maths", "array position $t$ here is step $t+1$ of the maths"),
  ("The final ᾱ_T must be near zero", r"The final $\bar{\alpha}_T$ must be near zero"),
  ("for the same ε", r"for the same $\epsilon$"),
 ],
 "exponential-smoothing": [
  ("Unrolling the recursion ℓ = α y + (1 − α) ℓ shows", r"Unrolling the recursion $\ell = \alpha y + (1-\alpha)\ell$ shows"),
  ("weights α(1 − α)ᵏ that", r"weights $\alpha(1-\alpha)^k$ that"),
  ("Choose a start level ℓ₀ (and slope b₀ for Holt) —", r"Choose a start level $\ell_0$ (and slope $b_0$ for Holt) —"),
  ("— α sets how fast", r"— $\alpha$ sets how fast"),
  ("— β sets how quickly", r"— $\beta$ sets how quickly"),
  ("α level smoothing in (0, 1) —", r"$\alpha$ level smoothing in $(0, 1)$ —"),
  ("β trend smoothing in (0, 1) —", r"$\beta$ trend smoothing in $(0, 1)$ —"),
  ("Damping φ —", r"Damping $\phi$ —"),
  ("A high α on a noisy series chases noise; the fitted values lag turning points when α is small",
   r"A high $\alpha$ on a noisy series chases noise; the fitted values lag turning points when $\alpha$ is small"),
  ("Pick α and β by one-step", r"Pick $\alpha$ and $\beta$ by one-step"),
  ("What do α near 0 and α near 1 mean?", r"What do $\alpha$ near $0$ and $\alpha$ near $1$ mean?"),
  ("shrinks by a constant factor (1 − α) per step back", r"shrinks by a constant factor $(1-\alpha)$ per step back"),
  ("α near 1 trusts the latest value and nearly copies it forward, a naive forecast; α near 0 keeps",
   r"$\alpha$ near $1$ trusts the latest value and nearly copies it forward, a naive forecast; $\alpha$ near $0$ keeps"),
 ],
 "gaussian-mixture-em": [
  ("Compute log πₖ N(x; μₖ, Σₖ) per point", r"Compute $\log \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$ per point"),
  ("Add a small ε to each covariance", r"Add a small $\epsilon$ to each covariance"),
  ("n_components K — choose", "n_components $K$ — choose"),
  ("reg_covar ε — floor", r"reg_covar $\epsilon$ — floor"),
  ("without the ε floor", r"without the $\epsilon$ floor"),
  ("Fit several K and compare BIC", "Fit several $K$ and compare BIC"),
  ("always improves with K.", "always improves with $K$."),
 ],
 "gradient-descent": [
  ("If the step is below 2/L, where", "If the step is below $2/L$, where"),
  ("L bounds the curvature", "$L$ bounds the curvature"),
  ("Step w ← w − η·∇L(w) — moves downhill by an amount set by η",
   r"Step $w \leftarrow w - \eta\,\nabla L(w)$ — moves downhill by an amount set by $\eta$"),
  ("Scale features and tune η —", r"Scale features and tune $\eta$ —"),
  ("Learning rate η (the most", r"Learning rate $\eta$ (the most"),
  ("Momentum β (0 for plain", r"Momentum $\beta$ (0 for plain"),
 ],
 "isotonic-regression-pava": [
  ("Predictions at new x need", "Predictions at new $x$ need"),
  ("Tied x values must", "Tied $x$ values must"),
 ],
 "label-smoothing": [
  ("(1 − ε) one-hot plus ε uniform.", r"$(1-\epsilon)$ one-hot plus $\epsilon$ uniform."),
  ("Build the target: ε/K everywhere, then add 1 − ε at the true class",
   r"Build the target: $\epsilon/K$ everywhere, then add $1-\epsilon$ at the true class"),
  ('"Smoothing ε"', r'"Smoothing $\epsilon$"'),
  ("Number of classes K (sets", "Number of classes $K$ (sets"),
  ("Too large ε underfits", r"Too large $\epsilon$ underfits"),
  ("spread ε over K classes, others over K − 1;", r"spread $\epsilon$ over $K$ classes, others over $K-1$;"),
  ("(1 − ε) on the true class mixed with a uniform ε/K over all classes.",
   r"$(1-\epsilon)$ on the true class mixed with a uniform $\epsilon/K$ over all classes."),
 ],
 "local-outlier-factor": [
  ("its k nearest neighbours. A score near one", "its $k$ nearest neighbours. A score near one"),
  ("Take the k nearest neighbours of each point and the distance to the k-th — the k-distance defines",
   "Take the $k$ nearest neighbours of each point and the distance to the $k$-th — the $k$-distance defines"),
  ("floor at the neighbour's k-distance — smooths", "floor at the neighbour's $k$-distance — smooths"),
  ("own k-distance, so", "own $k$-distance, so"),
  ("n_neighbors k — larger", "n_neighbors $k$ — larger"),
  ("Ties at the k-th distance", "Ties at the $k$-th distance"),
  ("Scores are relative to k;", "Scores are relative to $k$;"),
  ("neighbour's k-distance, which stops", "neighbour's $k$-distance, which stops"),
  ("How do you choose k?", "How do you choose $k$?"),
  ("larger k smooths the scores, and ranking stability across a range of k is",
   "larger $k$ smooths the scores, and ranking stability across a range of $k$ is"),
 ],
 "max-pooling": [
  ("Choose a window size k and stride", "Choose a window size $k$ and stride"),
  ("Compute the output size as ⌊(H − k) / stride⌋ + 1 —", r"Compute the output size as $\lfloor (H-k)/\mathrm{stride} \rfloor + 1$ —"),
  ("slice the matching k × k window", r"slice the matching $k \times k$ window"),
  ('"Window size k"', '"Window size $k$"'),
 ],
 "multiclass-metrics": [
  ("Reduce a K-class confusion", "Reduce a $K$-class confusion"),
  ("Divide total true positives by n for micro", "Divide total true positives by $n$ for micro"),
  ("integers 0 to K−1 here", "integers 0 to $K-1$ here"),
 ],
 "precision-recall-f1": [
  ("β in F-beta", r"$\beta$ in F-beta"),
 ],
 "reinforce-policy-gradient": [
  ("Form the gradient of Gₜ·log π(aₜ|sₜ) summed over steps", r"Form the gradient of $G_t \cdot \log \pi(a_t|s_t)$ summed over steps"),
  ("Discount γ", r"Discount $\gamma$"),
  ("omits the γᵗ factor", r"omits the $\gamma^t$ factor"),
  ("∇ p = p ∇ log p, which", r"$\nabla p = p\,\nabla \log p$, which"),
 ],
 "ridge-lasso-regularization": [
  ("Ridge adds λI to XᵀX, which", r"Ridge adds $\lambda I$ to $X^{\top}X$, which"),
  ("below λ stays at zero", r"below $\lambda$ stays at zero"),
  ("Centre X and y —", "Centre $X$ and $y$ —"),
  ("Ridge: solve (XᵀX + λI)w = Xᵀy — the added λI makes it",
   r"Ridge: solve $(X^{\top}X + \lambda I)w = X^{\top}y$ — the added $\lambda I$ makes it"),
  ("removing feature j from", "removing feature $j$ from"),
  ("λ (alpha) — penalty", r"$\lambda$ (alpha) — penalty"),
  ("Ridge uses λ‖w‖² on the summed error but Lasso uses 1/(2n);", r"Ridge uses $\lambda\|w\|^2$ on the summed error but Lasso uses $1/(2n)$;"),
  ("How do you choose λ?", r"How do you choose $\lambda$?"),
 ],
 "stratified-split": [
  ("Take the first round(f · n_c) shuffled", r"Take the first $\mathrm{round}(f \cdot n_c)$ shuffled"),
  ("differ slightly from f·n;", r"differ slightly from $f \cdot n$;"),
 ],
 "transformer-block": [
  ("residual path x + f(norm(x)) lets", r"residual path $x + f(\mathrm{norm}(x))$ lets"),
  ("Normalise x with LayerNorm", "Normalise $x$ with LayerNorm"),
  ("Project to Q, K, V and split", "Project to $Q$, $K$, $V$ and split"),
  ("project with Wo and add to x —", "project with $W_o$ and add to $x$ —"),
  ("which is linear in V.", "which is linear in $V$."),
  ("usually a few times d, so", "usually a few times $d$, so"),
  ("Model width d and number of heads h", "Model width $d$ and number of heads $h$"),
  ("small multiple of d)", "small multiple of $d$)"),
 ],
 "weight-initialization": [
  ("For y = Σ wᵢxᵢ with", r"For $y = \sum_i w_i x_i$ with"),
 ],
}

bad = 0
cnt = 0
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
            cnt += new.count("$") // 2
        elif n == 0 and new in text:
            pass  # already applied
        else:
            bad += 1
            print("MISSING/AMBIGUOUS", slug, n, old)
    f.write_text(text, encoding="utf-8", newline="")
print("g2 inline math done;", cnt, "formulas;", bad, "problems")
