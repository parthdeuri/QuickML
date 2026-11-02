#!/usr/bin/env python3
"""One-off assembly: sheets + family tags for the 56-card content set."""
from pathlib import Path
import re
import sys

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_families as mf  # noqa: E402

C = HERE.parent
PROBLEMS = C / "problems"

SHEETS = {
 "ml-core": ("ML Core", "Core classical machine learning, evaluation and the math behind them.", [
  ("Foundations", ["gradient-descent", "bayes-theorem", "maximum-likelihood", "entropy-kl-divergence",
                   "metropolis-hastings"]),
  ("Regression", ["linear-regression", "ridge-lasso-regularization"]),
  ("Classification", ["logistic-regression", "naive-bayes", "knn", "svm", "decision-tree",
                      "linear-discriminant-analysis", "kernel-trick-rbf"]),
  ("Ensembles", ["random-forest", "gradient-boosting", "adaboost"]),
  ("Sequences and recommenders", ["hidden-markov-forward", "matrix-factorization-recommender"]),
  ("Unsupervised", ["k-means", "gaussian-mixture-em", "dbscan", "pca"]),
  ("Preprocessing and linear algebra", ["feature-scaling", "svd"]),
  ("Evaluation and generalisation", ["bias-variance-tradeoff", "cross-validation", "precision-recall-f1",
                                     "roc-auc", "class-imbalance", "probability-calibration",
                                     "hyperparameter-search", "data-leakage", "multiclass-metrics",
                                     "stratified-split", "ab-test-significance"]),
  ("Production monitoring", ["concept-drift"]),
 ]),
 "dl-core": ("Deep Learning Core", "Neural network foundations, training tricks and the first architectures.", [
  ("Neural network foundations", ["softmax-cross-entropy", "mlp-forward-pass", "activation-functions",
                                  "backpropagation"]),
  ("Training setup", ["weight-initialization", "momentum-sgd", "adam-optimizer",
                      "learning-rate-schedules", "gradient-clipping"]),
  ("Normalisation and regularisation", ["batch-normalization", "layer-normalization", "dropout",
                                        "early-stopping", "residual-connections", "data-augmentation",
                                        "label-smoothing"]),
  ("Loss design and compression", ["focal-loss", "knowledge-distillation"]),
  ("Transfer and adaptation", ["transfer-learning-fine-tuning", "lora-low-rank-adaptation"]),
  ("Sequence models", ["rnn-cell", "lstm-cell"]),
  ("Architectures", ["convolution-2d", "max-pooling", "self-attention", "multi-head-attention",
                     "positional-encoding"]),
  ("Generative models", ["autoencoder", "vae-reparameterization", "gan-minimax-loss",
                         "diffusion-forward-process"]),
 ]),
 "nlp-vision-rl": ("NLP, Vision and RL", "Text, image and reinforcement-learning building blocks.", [
  ("NLP", ["tf-idf", "word2vec-skipgram", "bpe-tokenization", "positional-encoding", "self-attention",
           "multi-head-attention", "beam-search", "causal-masking", "transformer-block",
           "rotary-position-embedding", "kv-cache", "top-k-top-p-sampling", "perplexity",
           "cosine-similarity-embeddings"]),
  ("Vision", ["convolution-2d", "max-pooling", "iou-and-nms", "receptive-field",
             "depthwise-separable-convolution", "transposed-convolution", "data-augmentation"]),
  ("Reinforcement learning", ["q-learning", "reinforce-policy-gradient", "value-iteration"]),
 ]),
}


def main():
    have = {p.stem for p in PROBLEMS.glob("*.yaml")}
    used = set()
    counts = {}
    for sid, (name, desc, sections) in SHEETS.items():
        secs = []
        n = 0
        for sec, slugs in sections:
            missing = [s for s in slugs if s not in have]
            if missing:
                sys.exit(f"{sid}/{sec}: missing cards {missing}")
            secs.append({"name": sec, "problems": slugs})
            n += len(slugs)
            used.update(slugs)
        counts[sid] = n
        doc = {"id": sid, "name": name, "description": desc, "sections": secs}
        (C / "sheets" / f"{sid}.yaml").write_text(
            yaml.dump(doc, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8")
    orphans = sorted(have - used)
    if orphans:
        sys.exit(f"cards in no sheet: {orphans}")

    # family tags
    fam_name = {f["id"]: f["name"] for f in mf.F}
    tags = {}
    for fid, slugs in mf.MEMBERS.items():
        for s in slugs:
            if s in have:
                tags.setdefault(s, []).append(fam_name[fid])
    for slug, names in tags.items():
        path = PROBLEMS / f"{slug}.yaml"
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^patterns:.*\n?", "", text, flags=re.M)
        if not text.endswith("\n"):
            text += "\n"
        text += "patterns: [" + ", ".join(f'"{n}"' for n in names) + "]\n"
        path.write_text(text, encoding="utf-8")
    print("sheet counts:", counts)
    print("tagged cards:", len(tags), "untagged:", sorted(have - set(tags)))


if __name__ == "__main__":
    main()
