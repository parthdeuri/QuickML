"""Group 0: wrap formulas in prose fields with $...$ LaTeX (exact, single-hit edits)."""
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "problems"

E = {
 "ab-test-significance": [
  ("gives a z-score that", r"gives a $z$-score that"),
  ("to get z — how many", r"to get $z$ — how many"),
  ("Convert z to a two-sided p-value with", r"Convert $z$ to a two-sided $p$-value with"),
  ("Significance level α —", r"Significance level $\alpha$ —"),
  ("Power 1−β —", r"Power $1-\beta$ —"),
  ("Minimum detectable effect δ —", r"Minimum detectable effect $\delta$ —"),
  ("Baseline rate p —", r"Baseline rate $p$ —"),
  ("when p dips below α inflates", r"when $p$ dips below $\alpha$ inflates"),
  ("for α=0.05 two-sided", r"for $\alpha=0.05$ two-sided"),
  ("What does a p-value mean", r"What does a $p$-value mean"),
  ("rises well above α;", r"rises well above $\alpha$;"),
 ],
 "bahdanau-additive-attention": [
  ("decoder state with W_s and every encoder state with W_h —", r"decoder state with $W_s$ and every encoder state with $W_h$ —"),
  ("Dot with a learned vector v —", r"Dot with a learned vector $v$ —"),
  ("One weight per source position (T,) and", r"One weight per source position $(T,)$ and"),
  ("encoder width d_h.", r"encoder width $d_h$."),
  ("Attention hidden width d_a", r"Attention hidden width $d_a$"),
 ],
 "class-imbalance": [
  ("With weight wᶜ on class c, the", r"With weight $w^c$ on class $c$, the"),
  ("Compute balanced weights n / (K·nᶜ) —", r"Compute balanced weights $\frac{n}{K\,n^c}$ —"),
 ],
 "elastic-net-coordinate-descent": [
  ("p larger than", r"$p$ larger than"),
  ("n, automatic feature", r"$n$, automatic feature"),
  ("Start at w = 0 with residual r = y —", r"Start at $w = 0$ with residual $r = y$ —"),
  ("For coordinate j compute", r"For coordinate $j$ compute"),
  ("target for wⱼ alone", r"target for $w_j$ alone"),
  ("Soft-threshold by αρ and divide by zⱼ + α(1 − ρ) —", r"Soft-threshold by $\alpha\rho$ and divide by $z_j + \alpha(1-\rho)$ —"),
  ("change in wⱼ — keeps", r"change in $w_j$ — keeps"),
  ("instead of recomputing Xw", r"instead of recomputing $Xw$"),
  ("α — overall penalty", r"$\alpha$ — overall penalty"),
  ("l1_ratio ρ in [0, 1] —", r"l1_ratio $\rho$ in $[0, 1]$ —"),
  ("so centre y and X or", r"so centre $y$ and $X$ or"),
  ("features and small α", r"features and small $\alpha$"),
  ("keep more than n features when p is larger than n.", r"keep more than $n$ features when $p$ is larger than $n$."),
 ],
 "gae-advantage": [
  ("single parameter λ.", r"single parameter $\lambda$."),
  ("each TD error δ looks", r"each TD error $\delta$ looks"),
  ("with γλ controlling how", r"with $\gamma\lambda$ controlling how"),
  ("weights (γλ)^l is", r"weights $(\gamma\lambda)^l$ is"),
  ("λ = 0 recovers the", r"$\lambda = 0$ recovers the"),
  ("and λ = 1 recovers", r"and $\lambda = 1$ recovers"),
  ("satisfies A_t = δ_t + γλ·A_{t+1}, so", r"satisfies $A_t = \delta_t + \gamma\lambda\,A_{t+1}$, so"),
  ("Compute the TD error δ_t for every step, bootstrapping from V(s_{t+1}) unless",
   r"Compute the TD error $\delta_t$ for every step, bootstrapping from $V(s_{t+1})$ unless"),
  ("add δ_t to γλ times the next", r"add $\delta_t$ to $\gamma\lambda$ times the next"),
  ("Return advantages and A + V as", r"Return advantages and $A + V$ as"),
  ("Discount γ", r"Discount $\gamma$"),
  ("GAE parameter λ (low", r"GAE parameter $\lambda$ (low"),
  ("What do λ = 0 and λ = 1 give?", r"What do $\lambda = 0$ and $\lambda = 1$ give?"),
  ("λ = 0 gives the one-step", r"$\lambda = 0$ gives the one-step"),
  ("λ = 1 gives the discounted", r"$\lambda = 1$ gives the discounted"),
  ("plus γλ times the next advantage, so", r"plus $\gamma\lambda$ times the next advantage, so"),
 ],
 "gradient-boosting": [
  ("The residual y − F(x)", r"The residual $y - F(x)$"),
  ("moves F downhill", r"moves $F$ downhill"),
  ("Start from the mean of y —", r"Start from the mean of $y$ —"),
  ("Compute residuals y − F —", r"Compute residuals $y - F$ —"),
  ("Add η times the stump to F —", r"Add $\eta$ times the stump to $F$ —"),
  ("Repeat for M rounds", r"Repeat for $M$ rounds"),
  ("n_estimators M —", r"n_estimators $M$ —"),
  ("learning_rate η —", r"learning_rate $\eta$ —"),
 ],
 "hidden-markov-forward": [
  ("The forward variable αₜ(j) is", r"The forward variable $\alpha_t(j)$ is"),
  ("state j. Rescaling αₜ to sum", r"state $j$. Rescaling $\alpha_t$ to sum"),
  ("Start with α = π ⊙ B[:, o₁] —", r"Start with $\alpha = \pi \odot B[:, o_1]$ —"),
  ("Normalise α by its sum c and store log c —", r"Normalise $\alpha$ by its sum $c$ and store $\log c$ —"),
  ("Propagate with α @ A —", r"Propagate with $\alpha$ @ $A$ —"),
  ("Rescale again and add log c to", r"Rescale again and add $\log c$ to"),
  ("Number of hidden states N —", r"Number of hidden states $N$ —"),
  ("Initial distribution π, transition matrix A, emission matrix B —",
   r"Initial distribution $\pi$, transition matrix $A$, emission matrix $B$ —"),
  ("Zero entries in A or B make", r"Zero entries in $A$ or $B$ make"),
  ("each step then costs N² work instead of multiplying the number of paths by N.",
   r"each step then costs $N^2$ work instead of multiplying the number of paths by $N$."),
 ],
 "knn": [
  ("among the k closest", r"among the $k$ closest"),
  ("k sets the smoothness: k = 1 follows", r"$k$ sets the smoothness: $k = 1$ follows"),
  ("while a large k averages", r"while a large $k$ averages"),
  ("As n grows and", r"As $n$ grows and"),
  ("k grows more slowly than n,", r"$k$ grows more slowly than $n$,"),
  ("even k = 1 has error", r"even $k = 1$ has error"),
  ("Skip it for large n (each", r"Skip it for large $n$ (each"),
  ("via ‖a‖² + ‖b‖² − 2ab —", r"via $\|a\|^2 + \|b\|^2 - 2\,a^{\top}b$ —"),
  ("keep the first k indices", r"keep the first $k$ indices"),
  ("k — small is low-bias", r"$k$ — small is low-bias"),
  ("Use an odd k for two", r"Use an odd $k$ for two"),
  ("How do you choose k?", r"How do you choose $k$?"),
  ("small k overfits noise, large k underfits", r"small $k$ overfits noise, large $k$ underfits"),
 ],
 "linear-discriminant-analysis": [
  ("linear in x, so", r"linear in $x$, so"),
  ("eigenproblem of S_w⁻¹ S_b and keep the top k vectors —", r"eigenproblem of $S_w^{-1} S_b$ and keep the top $k$ vectors —"),
  ("Rescale each vector so wᵀ S_w w = 1 —", r"Rescale each vector so $w^{\top} S_w w = 1$ —"),
  ("n_components k — at most min(C−1, d)", r"n_components $k$ — at most $\min(C-1, d)$"),
  ("regularises S_w when", r"regularises $S_w$ when"),
  ("S_w is singular when features outnumber", r"$S_w$ is singular when features outnumber"),
  ("Only C−1 meaningful", r"Only $C-1$ meaningful"),
  ("when S_w is singular?", r"when $S_w$ is singular?"),
 ],
 "lstm-cell": [
  ("affine map of x and h, then", r"affine map of $x$ and $h$, then"),
  ("Output h = output gate · tanh(c) —", r"Output $h$ = output gate · $\tanh(c)$ —"),
  ("The cell state update c = f·c + i·g has", r"The cell state update $c = f \cdot c + i \cdot g$ has"),
  ("Hidden size H", r"Hidden size $H$"),
 ],
 "mlp-forward-pass": [
  ("with std √(2 / fan_in) and", r"with std $\sqrt{2 / \mathrm{fan\_in}}$ and"),
  ("Set a = X with one row", r"Set $a = X$ with one row"),
  ("Compute z = aW + b for", r"Compute $z = aW + b$ for"),
 ],
 "positional-encoding": [
  ("Compute angle = pos / 10000^(i/d) —", r"Compute $\mathrm{angle} = \mathrm{pos} / 10000^{i/d}$ —"),
  ("Put sin(angle) in the even columns and cos(angle) in the odd", r"Put $\sin(\mathrm{angle})$ in the even columns and $\cos(\mathrm{angle})$ in the odd"),
  ("at frequency ω is a point", r"at frequency $\omega$ is a point"),
  ("and moving k", r"and moving $k$"),
  ("by angle ωk, which", r"by angle $\omega k$, which"),
  ("assumes an even dimension d.", r"assumes an even dimension $d$."),
  ("Model dimension d", r"Model dimension $d$"),
  ("Maximum sequence length T", r"Maximum sequence length $T$"),
  ("The code assumes d is even; odd d needs", r"The code assumes $d$ is even; odd $d$ needs"),
 ],
 "random-forest": [
  ("average of B models with variance σ² and pairwise", r"average of $B$ models with variance $\sigma^2$ and pairwise"),
  ("correlation ρ is ρσ² + (1 − ρ)σ²/B. More", r"correlation $\rho$ is $\rho\sigma^2 + (1-\rho)\sigma^2/B$. More"),
  ("shrink ρ, the part", r"shrink $\rho$, the part"),
  ("Draw n rows with replacement", r"Draw $n$ rows with replacement"),
  ("pick m random features", r"pick $m$ random features"),
  ("Repeat for B trees —", r"Repeat for $B$ trees —"),
  ("n_estimators B —", r"n_estimators $B$ —"),
  ("max_features m —", r"max_features $m$ —"),
  ("so keep m below d", r"so keep $m$ below $d$"),
  ("only shrink the (1 − ρ)σ²/B term", r"only shrink the $(1-\rho)\sigma^2/B$ term"),
 ],
 "retrieval-metrics-recall-at-k": [
  ("appear in the top k, how", r"appear in the top $k$, how"),
  ("items made the top k, MRR", r"items made the top $k$, MRR"),
  ("hits in the first k divided", r"hits in the first $k$ divided"),
  ("Cut-off k (match", r"Cut-off $k$ (match"),
  ("here; 2^rel − 1 is also common", r"here; $2^{\mathrm{rel}} - 1$ is also common"),
  ("libraries use 2^rel − 1", r"libraries use $2^{\mathrm{rel}} - 1$"),
  ("needs k no larger", r"needs $k$ no larger"),
  ("missing from the top k can never", r"missing from the top $k$ can never"),
  ("Order within the top k matters", r"Order within the top $k$ matters"),
 ],
 "rotary-position-embedding": [
  ("rotate a query by m·θ and a key by n·θ,", r"rotate a query by $m\theta$ and a key by $n\theta$,"),
  ("angle difference (m−n)·θ, because", r"angle difference $(m-n)\theta$, because"),
  ("Pair up coordinates (x₀, x₁), (x₂, x₃), … —", r"Pair up coordinates $(x_0, x_1)$, $(x_2, x_3)$, … —"),
  ("Give pair i a frequency θᵢ = base^(−2i/d) —", r"Give pair $i$ a frequency $\theta_i = \mathrm{base}^{-2i/d}$ —"),
  ("Multiply position m by each θᵢ to get", r"Multiply position $m$ by each $\theta_i$ to get"),
  ("only the score q·k needs", r"only the score $q \cdot k$ needs"),
  ("rotating by angle φ is", r"rotating by angle $\phi$ is"),
  ("multiplication by e^{iφ}.", r"multiplication by $e^{i\phi}$."),
  ("angles combine into e^{i(m−n)θ}. The", r"angles combine into $e^{i(m-n)\theta}$. The"),
  ("the offset m−n alone", r"the offset $m-n$ alone"),
  ("Head width d (must be even)", r"Head width $d$ (must be even)"),
  ("the rotation by m−n and not on m and n separately", r"the rotation by $m-n$ and not on $m$ and $n$ separately"),
  ("enter the attention score q·k.", r"enter the attention score $q \cdot k$."),
  ("multiplies q and k by rotations", r"multiplies $q$ and $k$ by rotations"),
  ("to q and k after the projection", r"to $q$ and $k$ after the projection"),
 ],
 "tf-idf": [
  ("occurs — this is df", r"occurs — this is $\mathrm{df}$"),
  ("(use 1 + ln(tf) so repeats", r"(use $1 + \ln(\mathrm{tf})$ so repeats"),
  ("damps the ratio N/df so that", r"damps the ratio $N/\mathrm{df}$ so that"),
 ],
 "value-iteration": [
  ("discount γ < 1, so by", r"discount $\gamma < 1$, so by"),
  ("when γ is close to one", r"when $\gamma$ is close to one"),
  ("Initialise V to zeros", r"Initialise $V$ to zeros"),
  ("Compute Q = r + γ V[next] for every", r"Compute $Q = r + \gamma V[\mathrm{next}]$ for every"),
  ("Recompute Q from the final V and take", r"Recompute $Q$ from the final $V$ and take"),
  ("Discount γ — closer", r"Discount $\gamma$ — closer"),
  ("Rows of Q for terminal", r"Rows of $Q$ for terminal"),
  ("is a γ-contraction in the max norm", r"is a $\gamma$-contraction in the max norm"),
  ("get a policy from V*?", r"get a policy from $V^*$?"),
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
print("g0 inline math done;", bad, "problems")
