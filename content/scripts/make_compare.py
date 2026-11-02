#!/usr/bin/env python3
"""Writes the "Compare: A vs B" cards into content/patterns/.

They reuse the family-card format (shown in the app's Families mode) and
use aliases no concept card carries, so they never appear in the quiz.
Orders start at 100, after the families.
"""
from pathlib import Path
import yaml

OUT = Path(__file__).resolve().parent.parent / "patterns"


class Lit(str):
    pass


yaml.add_representer(
    Lit, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", str(v), style="|"))

C = [
 dict(id="compare-ridge-vs-lasso", name="Compare: Ridge vs Lasso",
  summary="Both shrink weights to fight overfitting; ridge shrinks smoothly, lasso can set weights exactly to zero.",
  when_to_use=["Pick ridge when most features carry some signal or are correlated.",
               "Pick lasso when you expect only a few features to matter and want a sparse model.",
               "Use elastic net when features are correlated and you still want sparsity."],
  key_idea="The L2 penalty has a gradient that shrinks as a weight shrinks, so weights get small but rarely reach zero. The L1 penalty pulls with constant strength, so small weights are driven all the way to zero. That is why lasso selects features and ridge does not.",
  template="""import numpy as np

def ridge_grad(w, lam):
    return 2 * lam * w          # shrinks in proportion to w

def lasso_grad(w, lam):
    return lam * np.sign(w)     # constant pull toward zero

def ridge_fit(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
""",
  complexity="Ridge has a closed form, O(n·d² + d³); lasso needs an iterative solver such as coordinate descent.",
  pitfalls=["Both need scaled features, or the penalty treats columns unequally.",
            "Lasso picks one of a group of correlated features more or less arbitrarily."],
  problems=["ridge-lasso-regularization", "linear-regression"]),
 dict(id="compare-forest-vs-boosting", name="Compare: Random forest vs Gradient boosting",
  summary="Both combine many trees; a forest averages independent deep trees, boosting adds shallow trees one after another.",
  when_to_use=["Pick a random forest for a strong, low-tuning baseline that is hard to overfit.",
               "Pick gradient boosting when you can tune carefully and want the best accuracy on tabular data.",
               "Prefer the forest when training must parallelise easily, because boosting is sequential."],
  key_idea="A forest reduces variance by averaging trees trained on different samples. Boosting reduces bias by letting each new tree fix what the ensemble still gets wrong. So forests use deep trees and boosting uses shallow ones.",
  template="""import numpy as np

def forest_predict(trees, X):
    return np.mean([t(X) for t in trees], axis=0)    # average

def boosting_predict(trees, X, lr):
    return lr * np.sum([t(X) for t in trees], axis=0)  # sum
""",
  complexity="A forest trains its trees in parallel; boosting trains them in sequence.",
  pitfalls=["Boosting overfits noisy labels without early stopping.",
            "A forest with too few trees gives noisy predictions."],
  problems=["random-forest", "gradient-boosting", "decision-tree"]),
 dict(id="compare-batchnorm-vs-layernorm", name="Compare: Batch norm vs Layer norm",
  summary="Both normalise activations; batch norm uses statistics across the batch, layer norm across the features of one example.",
  when_to_use=["Pick batch norm for convolutional networks with reasonably large batches.",
               "Pick layer norm for transformers and recurrent models, or when batches are tiny or vary.",
               "Pick layer norm when training and evaluation behaviour must be identical."],
  key_idea="The only difference is the axis the mean and variance are taken over. Batch norm needs the other examples in the batch, so it keeps running averages for evaluation. Layer norm looks at a single example, so it behaves the same in training and evaluation.",
  template="""import numpy as np

def batch_norm(x, eps=1e-5):          # x: (batch, features)
    mu, var = x.mean(0), x.var(0)       # over the batch
    return (x - mu) / np.sqrt(var + eps)

def layer_norm(x, eps=1e-5):
    mu = x.mean(1, keepdims=True)       # over the features
    var = x.var(1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)
""",
  complexity="Both are O(n·d); batch norm also stores running statistics.",
  pitfalls=["Batch norm degrades with very small batches.",
            "In evaluation mode batch norm must use running statistics, not the batch's."],
  problems=["batch-normalization", "layer-normalization"]),
 dict(id="compare-pr-vs-roc", name="Compare: Precision-recall vs ROC",
  summary="Both summarise a classifier across thresholds; precision-recall focuses on the positive class, ROC compares true and false positive rates.",
  when_to_use=["Pick precision-recall when positives are rare and false alarms matter.",
               "Pick ROC-AUC when classes are fairly balanced or both error types matter equally.",
               "Report both when stakeholders care about different parts of the trade-off."],
  key_idea="ROC uses the false positive rate, whose denominator is the large pool of negatives, so many false alarms barely move it. Precision divides by the predicted positives, so the same false alarms hurt it a lot. That is why ROC can look optimistic on imbalanced data.",
  template="""import numpy as np

def tpr_fpr(y, score, t):
    pred = score >= t
    tpr = (pred & (y == 1)).sum() / (y == 1).sum()
    fpr = (pred & (y == 0)).sum() / (y == 0).sum()
    return tpr, fpr

def precision_recall(y, score, t):
    pred = score >= t
    tp = (pred & (y == 1)).sum()
    return tp / max(pred.sum(), 1), tp / (y == 1).sum()
""",
  complexity="O(n) per threshold; the full curves need a sort, O(n log n).",
  pitfalls=["A random classifier has AUC 0.5 in ROC, but its precision-recall baseline equals the positive rate.",
            "Both curves depend on the ordering of scores, not on whether probabilities are calibrated."],
  problems=["precision-recall-f1", "roc-auc"]),
 dict(id="compare-momentum-vs-adam", name="Compare: SGD with momentum vs Adam",
  summary="Both speed up gradient descent; momentum smooths the direction, Adam also rescales each parameter by its gradient history.",
  when_to_use=["Pick SGD with momentum for well-tuned vision training where the best final accuracy matters.",
               "Pick Adam for fast progress with little tuning, especially for transformers and sparse gradients.",
               "Pick Adam when parameters have very different gradient scales."],
  key_idea="Momentum averages past gradients to keep moving through noise and narrow valleys. Adam adds a second average of squared gradients and divides by it, giving every parameter its own step size. That makes it robust to scale but a little less predictable in how well the result generalises.",
  template="""import numpy as np

def momentum_step(w, g, v, lr=0.1, mu=0.9):
    v = mu * v + g
    return w - lr * v, v

def adam_step(w, g, m, s, t, lr=1e-3, b1=0.9, b2=0.999):
    m = b1 * m + (1 - b1) * g
    s = b2 * s + (1 - b2) * g * g
    mh, sh = m / (1 - b1 ** t), s / (1 - b2 ** t)
    return w - lr * mh / (np.sqrt(sh) + 1e-8), m, s
""",
  complexity="Momentum stores one extra vector per parameter; Adam stores two.",
  pitfalls=["Adam needs about twice the optimiser memory of plain SGD.",
            "Weight decay behaves differently in Adam; use the decoupled form (AdamW)."],
  problems=["momentum-sgd", "adam-optimizer", "gradient-descent"]),
 dict(id="compare-pca-vs-autoencoder", name="Compare: PCA vs Autoencoder",
  summary="Both compress data into fewer dimensions; PCA is linear and has a closed form, an autoencoder can be non-linear and is trained.",
  when_to_use=["Pick PCA for a fast, deterministic, interpretable linear projection.",
               "Pick an autoencoder when the structure is curved and you have enough data to train one.",
               "Start with PCA as a baseline; an autoencoder must beat it to be worth the effort."],
  key_idea="A linear autoencoder trained with squared error learns the same subspace as PCA. Adding non-linear layers lets the encoder bend to curved structure, at the cost of training, tuning and a result that is not unique.",
  template="""import numpy as np

def pca_encode(X, k):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T                 # closed form

def ae_encode(X, W1, b1, W2, b2):
    return np.tanh(X @ W1 + b1) @ W2 + b2   # learned by training
""",
  complexity="PCA is one SVD; an autoencoder needs many gradient steps.",
  pitfalls=["PCA directions follow variance, which is not always what matters for the task.",
            "An autoencoder with a large hidden layer can simply copy its input."],
  problems=["pca", "autoencoder", "svd"]),
 dict(id="compare-kmeans-vs-gmm", name="Compare: k-means vs Gaussian mixture",
  summary="Both group points around centres; k-means makes hard round clusters, a Gaussian mixture gives soft, shaped clusters.",
  when_to_use=["Pick k-means for fast, simple clustering of roughly round, equal-sized groups.",
               "Pick a Gaussian mixture when clusters differ in size or are elliptical.",
               "Pick the mixture when you need membership probabilities."],
  key_idea="K-means is the limit of a Gaussian mixture where every cluster is a sphere of equal size and each point belongs fully to one. The mixture keeps covariances and weights and shares points between clusters through responsibilities. Both are fit by alternating two steps and can end in local optima.",
  template="""import numpy as np

def kmeans_assign(X, C):
    d = ((X[:, None] - C[None]) ** 2).sum(-1)
    return d.argmin(1)                      # hard labels

def gmm_responsibilities(logp):             # logp: (n, k)
    z = logp - logp.max(1, keepdims=True)
    r = np.exp(z)
    return r / r.sum(1, keepdims=True)      # soft labels
""",
  complexity="Both cost O(n·k·d) per iteration; the mixture adds covariance work.",
  pitfalls=["Both depend on the starting centres, so restart several times.",
            "You must choose the number of clusters in advance."],
  problems=["k-means", "gaussian-mixture-em"]),
 dict(id="compare-kmeans-vs-dbscan", name="Compare: k-means vs DBSCAN",
  summary="K-means needs the number of clusters and finds round groups; DBSCAN finds dense regions of any shape and marks outliers as noise.",
  when_to_use=["Pick k-means when you know roughly how many clusters there are and they are compact.",
               "Pick DBSCAN when clusters have odd shapes or the data has outliers.",
               "Avoid DBSCAN when the density varies a lot between clusters."],
  key_idea="K-means assigns every point to the nearest centre, so it must place outliers somewhere and cuts curved shapes into round pieces. DBSCAN grows clusters from points that have enough neighbours within a radius, so shape is free and sparse points become noise. The radius and neighbour count replace the cluster count as the settings to tune.",
  template="""import numpy as np

def kmeans_labels(X, C):
    return ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)

def is_core(X, eps, min_pts):
    d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    return (d <= eps).sum(1) >= min_pts     # dense enough?
""",
  complexity="K-means is O(n·k·d) per pass; plain DBSCAN is O(n²) without an index.",
  pitfalls=["DBSCAN's radius is hard to choose when features are not scaled.",
            "K-means is distorted by outliers because centres are means."],
  problems=["k-means", "dbscan"]),
 dict(id="compare-lstm-vs-attention", name="Compare: LSTM vs Attention",
  summary="Both model sequences; a recurrent cell passes one state step by step, attention lets every position read every other directly.",
  when_to_use=["Pick attention for long-range dependencies and parallel training on modern hardware.",
               "Pick a recurrent cell for streaming input with small, constant memory per step.",
               "Pick attention-based models when you have enough data and compute."],
  key_idea="A recurrent cell must squeeze everything it has seen into a fixed-size state and process steps in order. Attention compares each position with all the others, so information travels in one hop, but the cost grows with the square of the length. Attention also needs positional information because it ignores order by itself.",
  template="""import numpy as np

def rnn_step(h, x, Wh, Wx, b):
    return np.tanh(h @ Wh + x @ Wx + b)     # one state, in order

def attend(Q, K, V):
    s = Q @ K.T / np.sqrt(K.shape[-1])      # all pairs at once
    e = np.exp(s - s.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)) @ V
""",
  complexity="A recurrent pass is O(n·d²) but sequential; attention is O(n²·d) but parallel.",
  pitfalls=["Plain RNNs lose gradients over long sequences, and LSTMs only soften this.",
            "Attention memory grows with the square of the sequence length."],
  problems=["lstm-cell", "rnn-cell", "self-attention"]),
 dict(id="compare-vae-vs-gan", name="Compare: VAE vs GAN",
  summary="Both generate data from noise; a VAE trains with a likelihood bound, a GAN trains a generator against a discriminator.",
  when_to_use=["Pick a VAE when you need a smooth latent space and stable training.",
               "Pick a GAN when sharp, realistic samples matter more than stability.",
               "Pick a VAE when you also need to encode real data into latents."],
  key_idea="A VAE is trained by reconstruction plus a KL term, so it has a clear loss to minimise but tends to give blurry samples. A GAN has no likelihood; the generator improves only by fooling a discriminator, which gives sharp samples but an unstable two-player game.",
  template="""import numpy as np

def vae_loss(x, xhat, mu, logvar):
    rec = ((x - xhat) ** 2).sum()
    kl = -0.5 * (1 + logvar - mu ** 2 - np.exp(logvar)).sum()
    return rec + kl

def gan_d_loss(d_real, d_fake):      # probabilities in (0, 1)
    return -(np.log(d_real) + np.log(1 - d_fake)).mean()
""",
  complexity="Both need many training steps; GANs also need careful balancing of two networks.",
  pitfalls=["GANs can collapse to a few modes.",
            "VAE samples blur when the decoder noise model is too simple."],
  problems=["vae-reparameterization", "gan-minimax-loss", "autoencoder"]),
 dict(id="compare-naive-bayes-vs-logistic", name="Compare: Naive Bayes vs Logistic regression",
  summary="Both give a simple classifier; naive Bayes models how each class generates the features, logistic regression models the class probability directly.",
  when_to_use=["Pick naive Bayes when data is tiny or very high-dimensional, such as text.",
               "Pick logistic regression when you have enough data and correlated features.",
               "Pick naive Bayes for a very fast baseline that trains in one pass."],
  key_idea="Naive Bayes is generative: it learns each feature's distribution per class and assumes the features are independent given the class. Logistic regression is discriminative: it fits the decision boundary directly, so it copes with correlated features. With lots of data the discriminative model usually wins; with little data the generative one can win.",
  template="""import numpy as np

def nb_scores(log_prior, loglik):            # (k,), (n, k)
    return log_prior + loglik                # generative

def lr_probs(w, b, X):
    return 1 / (1 + np.exp(-(X @ w + b)))    # discriminative
""",
  complexity="Naive Bayes fits in one pass over the data; logistic regression needs an iterative solver.",
  pitfalls=["Naive Bayes probabilities are often overconfident.",
            "Logistic regression needs scaling and regularisation for stable fits."],
  problems=["naive-bayes", "logistic-regression"]),
 dict(id="compare-mse-vs-cross-entropy", name="Compare: Squared error vs Cross-entropy",
  summary="Both are losses; squared error suits real-valued targets, cross-entropy suits class probabilities.",
  when_to_use=["Pick squared error for regression with roughly Gaussian noise.",
               "Pick cross-entropy for classification with softmax or sigmoid outputs.",
               "Avoid squared error on probabilities, because its gradient vanishes when the output saturates."],
  key_idea="Squared error is the negative log-likelihood of Gaussian noise, and cross-entropy is the negative log-likelihood of a categorical label. With a sigmoid or softmax output, the gradient of cross-entropy with respect to the logits is simply prediction minus target, while squared error's gradient is damped by the saturating output.",
  template="""import numpy as np

def mse(pred, y):
    return ((pred - y) ** 2).mean()

def cross_entropy(logits, y):
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return -logp[np.arange(len(y)), y].mean()
""",
  complexity="Both are O(n) in the number of examples (times classes for cross-entropy).",
  pitfalls=["Compute log-softmax with log-sum-exp, not log(softmax).",
            "Class imbalance may call for class weights or focal loss on top of cross-entropy."],
  problems=["softmax-cross-entropy", "linear-regression", "maximum-likelihood"]),
]

KEYS = ["id", "name", "order", "summary", "when_to_use", "key_idea",
        "template", "complexity", "pitfalls", "problems", "aliases"]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    have = {p.stem for p in (OUT.parent / "problems").glob("*.yaml")}
    for order, c in enumerate(C, 100):
        d = dict(c)
        missing = [s for s in d["problems"] if s not in have]
        assert not missing, (d["id"], missing)
        d["order"] = order
        d["aliases"] = [d["name"]]
        d["template"] = Lit(d["template"])
        d = {k: d[k] for k in KEYS}
        (OUT / f"{d['id']}.yaml").write_text(
            yaml.dump(d, sort_keys=False, allow_unicode=True, width=78),
            encoding="utf-8")
    print("wrote", len(C), "compare cards")
