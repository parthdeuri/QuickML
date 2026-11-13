#!/usr/bin/env python3
"""Writes content/patterns/*.yaml (technique families used by Quiz/Compare).

Run to regenerate; the YAML files are the source of truth afterwards.
Members whose card file does not exist yet are skipped.
"""
from pathlib import Path
import yaml

OUT = Path(__file__).resolve().parent.parent / "patterns"


class Lit(str):
    pass


yaml.add_representer(
    Lit, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", str(v), style="|"))

F = [
 dict(id="linear-models", name="Linear models",
  summary="Predict with a weighted sum of features, then fit the weights by minimising a convex loss.",
  when_to_use=["You need a fast, interpretable baseline.",
               "Features are roughly linearly related to the target or a link of it.",
               "Data is small or wide and a simple model should not overfit."],
  key_idea="The model is one dot product, so the loss surface is a smooth bowl and the weights can be read as feature effects. Changing the loss (squared, log, hinge) changes the model but not the shape of the idea.",
  template="""import numpy as np

def predict(w, b, X):
    return X @ w + b            # regression score

def step(w, b, X, y, lr, grad_of_loss):
    g = grad_of_loss(predict(w, b, X), y)   # dL/d(score)
    return w - lr * X.T @ g / len(X), b - lr * g.mean()
""",
  complexity="Fit O(n·d) per gradient step or O(n·d² + d³) closed form; predict O(d).",
  pitfalls=["Collinear features make the weights unstable.", "Unscaled features slow gradient descent and skew regularisation."],
  aliases=["Linear models"]),
 dict(id="regularisation", name="Regularisation",
  summary="Constrain or perturb a model during training so it cannot memorise the training set.",
  when_to_use=["Training error is far below validation error.",
               "You have many features or parameters relative to the data.",
               "You want sparse weights (L1) or small stable weights (L2)."],
  key_idea="Every regulariser trades a little training fit for lower variance: penalties shrink weights, dropout and noise stop co-adaptation, early stopping limits how far optimisation goes. Pick the one whose bias you can live with.",
  template="""import numpy as np

def l2_grad(w, lam):
    return 2 * lam * w                 # adds to the loss gradient

def l1_grad(w, lam):
    return lam * np.sign(w)            # pushes small weights to zero

def dropout(h, p, rng):
    mask = rng.random(h.shape) >= p
    return h * mask / (1 - p)          # inverted dropout
""",
  complexity="Negligible extra cost per step; the tuning cost is choosing the strength.",
  pitfalls=["Too much regularisation underfits.", "Penalties assume scaled features; scale first."],
  aliases=["Regularisation"]),
 dict(id="gradient-optimisation", name="Gradient-based optimisation",
  summary="Improve parameters by following the gradient of a loss, with tricks for speed and stability.",
  when_to_use=["The loss is differentiable in the parameters.",
               "There is no closed form or it is too expensive.",
               "You train a neural network end to end."],
  key_idea="The gradient says which way is downhill; the learning rate says how far to step. Momentum, adaptive scaling and schedules are all ways of choosing a better step from the same gradient history.",
  template="""import numpy as np

def sgd(w, g, lr):
    return w - lr * g

def adam(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
    return w - lr * mh / (np.sqrt(vh) + eps), m, v
""",
  complexity="O(parameters) per step; memory doubles for Adam's two moments.",
  pitfalls=["A learning rate that is too high diverges.", "Adaptive methods can generalise slightly worse than tuned SGD."],
  aliases=["Gradient-based optimisation"]),
 dict(id="tree-ensembles", name="Tree ensembles",
  summary="Combine many shallow or randomised decision trees by averaging (bagging) or by fitting residuals (boosting).",
  when_to_use=["Tabular data with mixed feature types and interactions.",
               "You want strong accuracy with little preprocessing.",
               "Features are on different scales or have outliers."],
  key_idea="One tree is high variance, so average many decorrelated trees (random forest) or add small trees that each correct the previous errors (boosting). Bagging cuts variance, boosting cuts bias.",
  template="""import numpy as np

def bagging_predict(trees, X):
    return np.mean([t(X) for t in trees], axis=0)

def boosting_fit(X, y, fit_tree, rounds, lr):
    pred, trees = np.zeros(len(y)), []
    for _ in range(rounds):
        tree = fit_tree(X, y - pred)       # fit the residual
        pred += lr * tree(X)
        trees.append(tree)
    return trees
""",
  complexity="Fit O(T·n·d·log n); predict O(T·depth).",
  pitfalls=["Boosting overfits noisy labels without early stopping.", "Trees cannot extrapolate beyond the training range."],
  aliases=["Tree ensembles"]),
 dict(id="probabilistic-models", name="Probabilistic models",
  summary="Model the data distribution explicitly and reason with Bayes rule and likelihoods.",
  when_to_use=["You need calibrated uncertainty, not only a label.",
               "Data is scarce and a sensible prior helps.",
               "A latent structure (clusters, classes) generates the data."],
  key_idea="Write down how the data could have been generated, then fit parameters by maximising likelihood or computing a posterior. Naive Bayes, Gaussian mixtures and logistic regression are all this idea with different assumptions.",
  template="""import numpy as np

def gaussian_logpdf(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)

def posterior(log_prior, log_lik):
    s = log_prior + log_lik
    s -= s.max()                      # stabilise
    p = np.exp(s)
    return p / p.sum()
""",
  complexity="Usually O(n·d·k) for k components or classes.",
  pitfalls=["Wrong independence assumptions give overconfident probabilities.", "Work in log space to avoid underflow."],
  aliases=["Probabilistic models"]),
 dict(id="distance-and-clustering", name="Distance and clustering",
  summary="Group or classify points by how close they are in feature space.",
  when_to_use=["You have no labels and want structure (clustering).",
               "Similar examples should share a label (nearest neighbours).",
               "A meaningful distance between examples exists."],
  key_idea="Everything depends on the distance and the scale of the features. K-means uses centres, DBSCAN uses density, kNN uses the nearest labelled points; all fail if distances are meaningless.",
  template="""import numpy as np

def pairwise_dist(A, B):
    return np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))

def assign(X, centres):
    return pairwise_dist(X, centres).argmin(axis=1)
""",
  complexity="O(n·k·d) per assignment pass; kNN prediction O(n·d) without an index.",
  pitfalls=["Distances break down in very high dimensions.", "Unscaled features let one column dominate."],
  aliases=["Distance and clustering"]),
 dict(id="classification-metrics", name="Classification metrics",
  summary="Summarise a classifier's errors with numbers chosen for the cost of each mistake.",
  when_to_use=["Classes are imbalanced and accuracy hides failures.",
               "Missing a positive costs more than a false alarm, or the reverse.",
               "You must compare models independent of a threshold."],
  key_idea="Accuracy counts all errors equally. Precision, recall and F1 expose the trade-off at one threshold, ROC-AUC summarises all thresholds, and calibration asks whether the probabilities mean what they say.",
  template="""import numpy as np

def prf(y, yhat):
    tp = np.sum((y == 1) & (yhat == 1))
    fp = np.sum((y == 0) & (yhat == 1))
    fn = np.sum((y == 1) & (yhat == 0))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)
""",
  complexity="O(n) per threshold; AUC needs a sort, O(n log n).",
  pitfalls=["ROC-AUC looks optimistic on heavily imbalanced data.", "Picking the threshold on the test set leaks information."],
  aliases=["Classification metrics"]),
 dict(id="validation-hygiene", name="Validation hygiene",
  summary="Evaluate a model so the number you report is an honest estimate of future performance.",
  when_to_use=["You tune hyperparameters or choose between models.",
               "The dataset is small and one split is noisy.",
               "Preprocessing uses statistics of the data."],
  key_idea="Anything that sees the test data, even through a scaler or feature selection, inflates the score. Fit every preprocessing step inside the training fold only, and keep one untouched final test set.",
  template="""import numpy as np

def kfold(n, k, rng):
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate(folds[:i] + folds[i + 1:])
        yield train, val
""",
  complexity="k times the cost of one fit.",
  pitfalls=["Time series need forward-chaining splits, not shuffled folds.", "Repeated tuning on one validation set overfits it."],
  aliases=["Validation hygiene"]),
 dict(id="normalisation", name="Normalisation",
  summary="Rescale features or activations to a stable range so optimisation and distances behave.",
  when_to_use=["Features have very different scales.",
               "Deep network training is unstable or slow.",
               "Distance-based or penalised models are used."],
  key_idea="Subtract a mean and divide by a spread. Feature scaling does it once on the inputs; batch norm and layer norm do it on hidden activations, differing only in which axis the statistics are taken over.",
  template="""import numpy as np

def standardise(X, mean, std, eps=1e-8):
    return (X - mean) / (std + eps)

def layer_norm(h, gamma, beta, eps=1e-5):
    mu = h.mean(-1, keepdims=True)
    var = h.var(-1, keepdims=True)
    return gamma * (h - mu) / np.sqrt(var + eps) + beta
""",
  complexity="O(n·d); batch norm also stores running statistics.",
  pitfalls=["Compute scaling statistics on training data only.", "Batch norm behaves differently in train and eval mode."],
  aliases=["Normalisation"]),
 dict(id="attention-and-sequences", name="Attention and sequences",
  summary="Model ordered data by carrying state through time or by letting every position look at every other.",
  when_to_use=["Inputs are sequences such as text, audio or time series.",
               "Long-range dependencies matter.",
               "You want parallel training over positions."],
  key_idea="Recurrent cells compress the past into one state, which is sequential and forgets. Attention scores every pair of positions directly, so it is parallel and long-range, at the price of quadratic cost and the need for positional information.",
  template="""import numpy as np

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    return softmax(scores) @ V
""",
  complexity="Attention O(n²·d) in sequence length n; an RNN is O(n·d²) but sequential.",
  pitfalls=["Without positional encoding attention ignores order.", "Plain RNNs suffer vanishing gradients on long sequences."],
  aliases=["Attention and sequences"]),
 dict(id="dimensionality-reduction", name="Dimensionality reduction",
  summary="Compress features into fewer coordinates that keep most of the structure.",
  when_to_use=["There are many correlated features.",
               "You want to visualise or denoise data.",
               "Downstream models are slow or overfit in high dimensions."],
  key_idea="Find the directions that carry the most variance (PCA, via SVD) or learn a compressed code that reconstructs the input (autoencoder). A linear autoencoder spans the same subspace as PCA.",
  template="""import numpy as np

def pca(X, k):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, Vt[:k]       # scores, components
""",
  complexity="SVD O(min(n,d)·n·d).",
  pitfalls=["Variance is not the same as usefulness for the label.", "Scale features before PCA."],
  aliases=["Dimensionality reduction"]),
 dict(id="information-theoretic-losses", name="Information-theoretic losses",
  summary="Train by minimising the gap between the model's distribution and the data's, measured with entropy.",
  when_to_use=["The model outputs a probability distribution.",
               "You classify with softmax or sigmoid outputs.",
               "You compare or regularise distributions (VAEs, distillation)."],
  key_idea="Cross-entropy equals entropy plus KL divergence, and the entropy of the data is fixed, so minimising cross-entropy minimises KL and maximises likelihood. Those three views are one idea.",
  template="""import numpy as np

def cross_entropy(logits, y):
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return -logp[np.arange(len(y)), y].mean()

def kl(p, q):
    return np.sum(p * (np.log(p) - np.log(q)))
""",
  complexity="O(n·classes).",
  pitfalls=["Compute log-softmax with log-sum-exp, not log(softmax).", "KL is asymmetric; the direction changes the behaviour."],
  aliases=["Information-theoretic losses"]),
]

# Family -> card slugs (cards that do not exist yet are skipped).
MEMBERS = {
 "linear-models": ["linear-regression", "logistic-regression", "svm", "ridge-lasso-regularization"],
 "regularisation": ["ridge-lasso-regularization", "dropout", "early-stopping", "bias-variance-tradeoff"],
 "gradient-optimisation": ["gradient-descent", "adam-optimizer", "learning-rate-schedules", "backpropagation", "weight-initialization"],
 "tree-ensembles": ["decision-tree", "random-forest", "gradient-boosting"],
 "probabilistic-models": ["naive-bayes", "bayes-theorem", "maximum-likelihood", "gaussian-mixture-em"],
 "distance-and-clustering": ["knn", "k-means", "dbscan"],
 "classification-metrics": ["precision-recall-f1", "roc-auc", "probability-calibration", "class-imbalance"],
 "validation-hygiene": ["cross-validation", "data-leakage", "hyperparameter-search", "concept-drift"],
 "normalisation": ["batch-normalization", "layer-normalization", "feature-scaling"],
 "attention-and-sequences": ["self-attention", "multi-head-attention", "positional-encoding", "rnn-cell", "lstm-cell"],
 "dimensionality-reduction": ["pca", "svd", "autoencoder"],
 "information-theoretic-losses": ["entropy-kl-divergence", "softmax-cross-entropy", "maximum-likelihood", "vae-reparameterization"],
}

F += [
 dict(id="network-building-blocks", name="Network building blocks",
  summary="The small differentiable pieces every neural network is assembled from.",
  when_to_use=["You are designing or debugging a feed-forward network.",
               "Gradients vanish or explode as depth grows.",
               "You need non-linearity between linear layers."],
  key_idea="A network is linear maps separated by non-linearities. Activations decide what gradients survive, and residual connections give the gradient a shortcut so depth stops hurting.",
  template="""import numpy as np

def relu(x):
    return np.maximum(0, x)

def block(x, W, b):
    return x + relu(x @ W + b)          # residual connection
""",
  complexity="O(width²) per dense layer.",
  pitfalls=["Without a non-linearity stacked layers collapse to one linear map.", "Saturating activations such as sigmoid kill gradients in deep stacks."],
  aliases=["Network building blocks"]),
 dict(id="convolution-and-vision", name="Convolution and vision",
  summary="Process images with weights shared across positions, then pool and score regions.",
  when_to_use=["Inputs have a grid structure such as images.",
               "The same pattern can appear anywhere in the input.",
               "You detect objects and must merge overlapping boxes."],
  key_idea="Sliding one small kernel over the image reuses its weights everywhere, which gives translation equivariance with few parameters. Pooling then shrinks the map, and IoU with non-maximum suppression cleans up detections.",
  template="""import numpy as np

def conv2d(x, k):
    h, w = k.shape
    H, W = x.shape[0] - h + 1, x.shape[1] - w + 1
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = (x[i:i + h, j:j + w] * k).sum()
    return out
""",
  complexity="O(H·W·kh·kw) per channel pair.",
  pitfalls=["Strided pooling throws away fine position information.", "Plain NMS removes overlapping true objects of different classes."],
  aliases=["Convolution and vision"]),
 dict(id="text-processing", name="Text processing",
  summary="Turn raw text into numbers: tokens, weights, embeddings and search over outputs.",
  when_to_use=["Inputs are words or sentences.",
               "You need a vocabulary that handles rare and unseen words.",
               "You generate sequences and must choose among many continuations."],
  key_idea="Models need numbers, so text is split into tokens. Tokens are then weighted by how informative they are (TF-IDF), mapped to vectors (word2vec), and generated by searching for the most likely sequence (beam search).",
  template="""import numpy as np

def tfidf(counts):
    df = (counts > 0).sum(0)
    idf = np.log((1 + len(counts)) / (1 + df)) + 1
    x = counts * idf
    return x / np.linalg.norm(x, axis=1, keepdims=True)
""",
  complexity="TF-IDF O(documents·vocabulary); beam search O(width·vocabulary·length).",
  pitfalls=["Word-level vocabularies fail on unseen words; subword tokenisation helps.", "Greedy decoding misses sequences whose early tokens look unlikely."],
  aliases=["Text processing"]),
 dict(id="reinforcement-learning", name="Reinforcement learning",
  summary="Learn to act by trial and error from rewards, by estimating values or improving a policy directly.",
  when_to_use=["There is no labelled answer, only a delayed reward.",
               "Actions change the situation you see next.",
               "You can simulate the environment cheaply."],
  key_idea="Value methods such as Q-learning learn how good each action is and act greedily on it. Policy-gradient methods such as REINFORCE nudge the policy toward actions that led to high return. Both trade exploration against exploitation.",
  template="""import numpy as np

def q_update(Q, s, a, r, s2, alpha=0.1, gamma=0.99):
    target = r + gamma * Q[s2].max()
    Q[s, a] += alpha * (target - Q[s, a])
    return Q
""",
  complexity="O(1) per Q-learning update; table size O(states·actions).",
  pitfalls=["Too little exploration locks in a poor policy.", "Policy-gradient estimates have high variance; use a baseline."],
  aliases=["Reinforcement learning"]),
]
MEMBERS.update({
 "network-building-blocks": ["activation-functions", "mlp-forward-pass", "residual-connections"],
 "convolution-and-vision": ["convolution-2d", "max-pooling", "iou-and-nms"],
 "text-processing": ["tf-idf", "word2vec-skipgram", "bpe-tokenization", "beam-search"],
 "reinforcement-learning": ["q-learning", "reinforce-policy-gradient"],
})

F += [
 dict(id="generative-models", name="Generative models",
  summary="Learn the data distribution well enough to sample new examples from it.",
  when_to_use=["You need to create new images, text or other data.",
               "You want a compact latent representation you can sample from.",
               "Labels are scarce but raw data is plentiful."],
  key_idea="Every generative model turns simple noise into data. VAEs do it with an encoder and a decoder trained on a likelihood bound, GANs with a generator trained against a critic, and diffusion models by learning to undo gradual noising.",
  template="""import numpy as np

def reparameterise(mu, logvar, rng):
    eps = rng.standard_normal(mu.shape)
    return mu + np.exp(0.5 * logvar) * eps     # z ~ N(mu, sigma^2)

def noise_step(x0, abar_t, rng):
    eps = rng.standard_normal(x0.shape)
    return np.sqrt(abar_t) * x0 + np.sqrt(1 - abar_t) * eps
""",
  complexity="Training is usually the expensive part; diffusion sampling needs many network passes.",
  pitfalls=["GAN training can oscillate or collapse to few modes.", "VAE samples tend to be blurry compared with GANs and diffusion."],
  aliases=["Generative models"]),
 dict(id="transfer-and-adaptation", name="Transfer and adaptation",
  summary="Reuse a trained model by fine-tuning it cheaply or compressing it into a smaller one.",
  when_to_use=["You have little data for the target task.",
               "A large pretrained model exists for a related domain.",
               "You must fit a model into a small memory or latency budget."],
  key_idea="Pretrained weights already encode useful features, so train only a small part (a head or low-rank update) or let a small student imitate the big teacher's outputs. You keep most of the accuracy for a fraction of the cost.",
  template="""import numpy as np

def lora_forward(x, W, A, B, alpha, r):
    return x @ W + (alpha / r) * (x @ A) @ B   # W frozen

def distill_loss(student_logits, teacher_logits, T=2.0):
    def sm(z):
        e = np.exp((z - z.max(-1, keepdims=True)) / T)
        return e / e.sum(-1, keepdims=True)
    p, q = sm(teacher_logits), sm(student_logits)
    return np.sum(p * (np.log(p) - np.log(q)), -1).mean() * T * T
""",
  complexity="LoRA trains O(r·(d_in + d_out)) parameters per layer instead of O(d_in·d_out).",
  pitfalls=["Fine-tuning with too large a learning rate destroys pretrained features.", "A student cannot exceed what the teacher's outputs reveal."],
  aliases=["Transfer and adaptation"]),
]
MEMBERS.update({
 "generative-models": ["autoencoder", "vae-reparameterization", "gan-minimax-loss", "diffusion-forward-process"],
 "transfer-and-adaptation": ["transfer-learning-fine-tuning", "lora-low-rank-adaptation", "knowledge-distillation"],
})
EXTRA = {
 "convolution-and-vision": ["receptive-field", "depthwise-separable-convolution", "transposed-convolution", "data-augmentation"],
 "regularisation": ["data-augmentation", "label-smoothing"],
 "gradient-optimisation": ["gradient-clipping", "momentum-sgd"],
 "attention-and-sequences": ["causal-masking", "transformer-block", "rotary-position-embedding", "kv-cache", "hidden-markov-forward"],
 "text-processing": ["top-k-top-p-sampling", "perplexity", "cosine-similarity-embeddings"],
 "information-theoretic-losses": ["focal-loss", "label-smoothing"],
 "tree-ensembles": ["adaboost"],
 "probabilistic-models": ["linear-discriminant-analysis", "hidden-markov-forward", "metropolis-hastings"],
 "linear-models": ["kernel-trick-rbf"],
 "dimensionality-reduction": ["matrix-factorization-recommender"],
 "classification-metrics": ["multiclass-metrics"],
 "validation-hygiene": ["ab-test-significance", "stratified-split"],
 "reinforcement-learning": ["value-iteration"],
}
for _k, _v in EXTRA.items():
    MEMBERS[_k] = MEMBERS[_k] + [x for x in _v if x not in MEMBERS[_k]]

F += [
 dict(id="time-series", name="Time series",
  summary="Forecast ordered data by smoothing the past, measuring how it correlates with itself, and modelling it autoregressively.",
  when_to_use=["Observations are ordered in time and neighbours are related.",
               "You need a quick baseline forecast before anything complex.",
               "You want to know how many past steps still carry information."],
  key_idea="Time series break the independence assumption, so we model the memory directly. Exponential smoothing keeps a decaying average, autocorrelation measures how far back the memory reaches, and ARIMA turns that memory into a regression on past values after differencing away trends.",
  template="""import numpy as np

def ses(x, alpha):
    level, out = x[0], []
    for v in x:
        out.append(level)             # forecast before seeing v
        level = alpha * v + (1 - alpha) * level
    return np.array(out), level

def acf(x, k):
    x = x - x.mean()
    return (x[:-k] * x[k:]).sum() / (x * x).sum()
""",
  complexity="O(n) per pass for smoothing and per lag for autocorrelation.",
  pitfalls=["Shuffled train and test splits leak the future; split by time.", "Trends and seasonality must be removed or modelled before assuming stationarity."],
  aliases=["Time series"]),
 dict(id="graph-neural-networks", name="Graph neural networks",
  summary="Learn on graphs by letting every node repeatedly combine its neighbours' features.",
  when_to_use=["Data is a graph such as molecules, social networks or citations.",
               "A node's label depends on its neighbourhood.",
               "The number of neighbours varies from node to node."],
  key_idea="One layer aggregates each node's neighbours (a normalised sum, or an attention-weighted sum) and then transforms the result with shared weights. Stacking k layers lets information travel k hops, but too many layers blur all nodes together.",
  template="""import numpy as np

def gcn_layer(A, X, W):
    A = A + np.eye(len(A))               # add self-loops
    d = A.sum(1) ** -0.5
    A_hat = d[:, None] * A * d[None]       # D^-1/2 A D^-1/2
    return np.maximum(A_hat @ X @ W, 0)    # aggregate, then transform
""",
  complexity="O(edges·d) with a sparse adjacency; the dense version here is O(n²·d).",
  pitfalls=["Deep stacks over-smooth: node features become nearly identical.", "A dense adjacency does not scale to large graphs."],
  aliases=["Graph neural networks"]),
 dict(id="preference-alignment", name="Preference alignment and policy gradients",
  summary="Steer a model with comparisons or rewards: learn a reward from preferences, or optimise the policy directly.",
  when_to_use=["Quality is easier to compare than to label.",
               "You fine-tune a language model toward human preferences.",
               "An agent must improve a stochastic policy from rewards."],
  key_idea="A pairwise Bradley-Terry loss turns preferences into a reward score. DPO folds that reward into the policy's own log-ratios, so no separate reward model or sampling loop is needed. PPO and advantage estimation do the reinforcement-learning version, clipping how far each update can move the policy.",
  template="""import numpy as np

def log_sigmoid(z):
    return -np.logaddexp(0, -z)

def bradley_terry_loss(r_chosen, r_rejected):
    return -log_sigmoid(r_chosen - r_rejected).mean()

def dpo_loss(pc, pr, rc, rr, beta=0.1):
    margin = (pc - rc) - (pr - rr)      # vs reference
    return -log_sigmoid(beta * margin).mean()
""",
  complexity="O(batch) once log-probabilities are computed; the cost is in running the language model.",
  pitfalls=["Preference data is noisy and can encode annotator bias.", "Pushing the policy far from the reference can hurt fluency."],
  aliases=["Preference alignment and policy gradients"]),
 dict(id="text-evaluation-retrieval", name="Text evaluation and retrieval",
  summary="Score generated text against references and measure how well search puts relevant documents first.",
  when_to_use=["You compare machine translation or summaries with references.",
               "You build search or retrieval-augmented generation.",
               "You need to rank documents and report a single quality number."],
  key_idea="Overlap metrics such as BLEU and ROUGE count shared n-grams, so they are cheap but blind to meaning. Retrieval metrics judge the ranked list: recall at k asks whether the answer was found, MRR rewards finding it early, and nDCG also values graded relevance.",
  template="""import numpy as np

def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)

def mrr(ranked, relevant):
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1 / i
    return 0.0
""",
  complexity="O(k) per query for recall and O(n) for MRR.",
  pitfalls=["High n-gram overlap does not guarantee a correct or fluent answer.", "Retrieval scores depend on the quality of the relevance labels."],
  aliases=["Text evaluation and retrieval"]),
]
MEMBERS.update({
 "time-series": ["exponential-smoothing", "autocorrelation-acf-pacf", "arima-model"],
 "graph-neural-networks": ["gnn-message-passing", "graph-attention-layer"],
 "preference-alignment": ["dpo-loss", "reward-model-bradley-terry", "ppo-clipped-objective", "gae-advantage"],
 "text-evaluation-retrieval": ["bleu-score", "rouge-score", "retrieval-metrics-recall-at-k", "rag-dense-retrieval"],
})
EXTRA2 = {
 "text-processing": ["bert-masked-lm"],
 "attention-and-sequences": ["bahdanau-additive-attention", "cross-attention-encoder-decoder", "gru-cell", "bidirectional-rnn"],
 "convolution-and-vision": ["resnet-basic-block", "vision-transformer-patch-embed", "mean-average-precision"],
 "gradient-optimisation": ["rmsprop-adagrad"],
 "information-theoretic-losses": ["clip-contrastive-loss"],
 "distance-and-clustering": ["hierarchical-clustering", "local-outlier-factor", "silhouette-clustering-metrics"],
 "dimensionality-reduction": ["nmf-multiplicative-updates"],
 "linear-models": ["elastic-net-coordinate-descent"],
 "regularisation": ["elastic-net-coordinate-descent"],
 "probabilistic-models": ["gaussian-process-regression"],
 "classification-metrics": ["isotonic-regression-pava"],
 "reinforcement-learning": ["ppo-clipped-objective", "gae-advantage"],
}
for _k, _v in EXTRA2.items():
    MEMBERS[_k] = MEMBERS[_k] + [x for x in _v if x not in MEMBERS[_k]]

EXTRA3 = {
 "text-processing": ["text-preprocessing-stemming", "bag-of-words-ngram-lm", "glove-embeddings",
                     "wordpiece", "sentencepiece-unigram", "contextual-embeddings"],
 "attention-and-sequences": ["decoder-only-block", "contextual-embeddings", "autoregressive-decoding"],
 "network-building-blocks": ["feed-forward-gelu-swiglu"],
 "information-theoretic-losses": ["next-token-prediction-loss"],
 "text-evaluation-retrieval": ["hallucination-detection"],
}
for _k, _v in EXTRA3.items():
    MEMBERS[_k] = MEMBERS[_k] + [x for x in _v if x not in MEMBERS[_k]]

EXTRA4 = {
 "attention-and-sequences": ["gpt-decoder-only", "llama-rmsnorm-gqa", "sliding-window-attention"],
 "normalisation": ["llama-rmsnorm-gqa"],
 "network-building-blocks": ["mixture-of-experts-routing"],
 "text-processing": ["t5-span-corruption"],
}
for _k, _v in EXTRA4.items():
    MEMBERS[_k] = MEMBERS[_k] + [x for x in _v if x not in MEMBERS[_k]]

EXTRA5 = {
 "transfer-and-adaptation": ["adapter-layers", "qlora-nf4-quantization", "sft-loss-masking",
                             "instruction-tuning-chat-templates"],
 "preference-alignment": ["rlhf-kl-penalty"],
 "text-evaluation-retrieval": ["rag-evaluation-faithfulness", "hybrid-search-rrf", "reranking-mmr",
                               "query-transformation-hyde", "document-chunking", "ann-vector-index"],
 "validation-hygiene": ["finetune-dataset-prep"],
}
for _k, _v in EXTRA5.items():
    MEMBERS[_k] = MEMBERS[_k] + [x for x in _v if x not in MEMBERS[_k]]

KEYS = ["id", "name", "order", "summary", "when_to_use", "key_idea",
        "template", "complexity", "pitfalls", "problems", "aliases"]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    have = {p.stem for p in (OUT.parent / "problems").glob("*.yaml")}
    for order, f in enumerate(F, 1):
        d = dict(f)
        d["order"] = order
        d["template"] = Lit(d["template"])
        d["problems"] = [s for s in MEMBERS[d["id"]] if s in have]
        d = {k: d[k] for k in KEYS}
        (OUT / f"{d['id']}.yaml").write_text(
            yaml.dump(d, sort_keys=False, allow_unicode=True, width=78),
            encoding="utf-8")
    print("wrote", len(F), "families")
