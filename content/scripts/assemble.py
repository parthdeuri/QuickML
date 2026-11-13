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


def _extend():
    def add(sid, sec, slugs):
        for name, lst in SHEETS[sid][2]:
            if name == sec:
                lst.extend(x for x in slugs if x not in lst)
                return
        SHEETS[sid][2].append((sec, list(slugs)))

    add("ml-core", "Regression", ["gaussian-process-regression", "elastic-net-coordinate-descent"])
    add("ml-core", "Unsupervised", ["hierarchical-clustering", "local-outlier-factor",
                                    "nmf-multiplicative-updates"])
    add("ml-core", "Evaluation and generalisation", ["silhouette-clustering-metrics",
                                                     "isotonic-regression-pava"])
    add("ml-core", "Time series", ["exponential-smoothing", "autocorrelation-acf-pacf", "arima-model"])
    add("dl-core", "Training setup", ["rmsprop-adagrad"])
    add("dl-core", "Loss design and compression", ["clip-contrastive-loss"])
    add("dl-core", "Sequence models", ["gru-cell", "bidirectional-rnn"])
    add("dl-core", "Architectures", ["resnet-basic-block", "vision-transformer-patch-embed"])
    add("dl-core", "Graph neural networks", ["gnn-message-passing", "graph-attention-layer"])
    add("nlp-vision-rl", "NLP", ["bahdanau-additive-attention", "cross-attention-encoder-decoder",
                                 "bert-masked-lm"])
    add("nlp-vision-rl", "LLM alignment", ["reward-model-bradley-terry", "dpo-loss"])
    add("nlp-vision-rl", "Text evaluation and retrieval", ["bleu-score", "rouge-score",
                                                           "retrieval-metrics-recall-at-k",
                                                           "rag-dense-retrieval"])
    add("nlp-vision-rl", "Vision", ["resnet-basic-block", "vision-transformer-patch-embed",
                                    "mean-average-precision"])
    add("nlp-vision-rl", "Reinforcement learning", ["ppo-clipped-objective", "gae-advantage"])


_extend()


SHEETS["llm-engineer"] = ("LLM Engineer", "The LLM engineer roadmap: language, transformers, how LLMs work and tokenisation.", [
  ("5. Natural language processing", ["text-preprocessing-stemming", "bag-of-words-ngram-lm", "tf-idf",
                                      "word2vec-skipgram", "glove-embeddings", "rnn-cell", "lstm-cell",
                                      "gru-cell", "perplexity"]),
  ("6. Transformer architecture", ["self-attention", "multi-head-attention", "positional-encoding",
                                   "feed-forward-gelu-swiglu", "layer-normalization", "residual-connections",
                                   "causal-masking", "transformer-block", "decoder-only-block",
                                   "cross-attention-encoder-decoder"]),
  ("7. Understanding large language models", ["next-token-prediction-loss", "autoregressive-decoding",
                                              "temperature-sampling", "top-k-top-p-sampling", "kv-cache",
                                              "scaling-laws", "parameter-count-flops",
                                              "context-window-limits", "hallucination-detection"]),
  ("8. Tokenisation and embeddings", ["bpe-tokenization", "wordpiece", "sentencepiece-unigram",
                                      "cosine-similarity-embeddings", "contextual-embeddings"]),
])

SHEETS["llm-engineer"][2].extend([
  ("9. Models and architectures", ["bert-masked-lm", "gpt-decoder-only", "t5-span-corruption",
                                   "llama-rmsnorm-gqa", "rotary-position-embedding",
                                   "sliding-window-attention", "mixture-of-experts-routing",
                                   "model-landscape"]),
  ("10. Prompt engineering", ["few-shot-prompting", "chain-of-thought-self-consistency",
                              "role-system-prompts", "context-engineering",
                              "structured-outputs-json-schema", "prompt-templates",
                              "prompt-optimization", "prompt-injection-defense"]),
])

SHEETS["llm-engineer"][2].extend([
  ("12. Fine-tuning and adaptation", ["transfer-learning-fine-tuning", "sft-loss-masking",
                                      "instruction-tuning-chat-templates", "finetune-dataset-prep",
                                      "lora-low-rank-adaptation", "qlora-nf4-quantization",
                                      "adapter-layers", "int8-quantization", "knowledge-distillation",
                                      "reward-model-bradley-terry", "rlhf-kl-penalty", "dpo-loss",
                                      "finetune-evaluation"]),
  ("13. Retrieval-augmented generation", ["rag-dense-retrieval", "document-loading-extraction",
                                          "document-chunking", "ann-vector-index", "metadata-filtering",
                                          "hybrid-search-rrf", "reranking-mmr",
                                          "query-transformation-hyde", "retrieval-metrics-recall-at-k",
                                          "rag-evaluation-faithfulness"]),
])

SHEETS["llm-engineer"][2].extend([
  ("16. Evaluation and observability", ["bleu-score", "rouge-score", "llm-as-judge",
                                        "elo-pairwise-ranking", "benchmark-contamination",
                                        "tracing-and-logging",
                                        "prompt-versioning", "concept-drift"]),
  ("17. Inference and optimisation", ["magnitude-pruning", "gpu-memory-estimation",
                                      "continuous-batching", "speculative-decoding",
                                      "flash-attention-online-softmax", "tensor-parallel-matmul",
                                      "pipeline-parallel-schedule"]),
  ("18. Security and responsible AI", ["jailbreak-and-red-teaming",
                                       "pii-redaction", "output-validation-guardrails",
                                       "content-safety-classifiers", "fairness-metrics"]),
])

SHEETS["llm-engineer"][2][0:0] = [
  ("1. Programming foundations", ["numpy-broadcasting-vectorisation", "pandas-groupby-merge-window",
                                  "python-generators-functional", "python-oop-protocols-dataclasses",
                                  "json-api-requests-retries", "git-object-model",
                                  "big-o-empirical-complexity", "linux-command-line",
                                  "data-structures-algorithms"]),
  ("2. Mathematics for LLMs", ["vectors-norms-distances", "dot-product-matrix-multiplication",
                               "eigen-decomposition-power-iteration", "svd",
                               "derivatives-chain-rule-gradient-check", "gradient-descent",
                               "probability-distributions-sampling", "bayes-theorem",
                               "maximum-likelihood", "entropy-kl-divergence",
                               "statistics-hypothesis-testing", "convex-optimisation-lagrange"]),
  ("3. Machine learning fundamentals", ["linear-regression", "logistic-regression", "k-means", "pca",
                                        "feature-engineering-encoding", "stratified-split",
                                        "cross-validation", "bias-variance-tradeoff",
                                        "loss-functions-regression-classification", "precision-recall-f1",
                                        "roc-auc", "sklearn-pipeline-column-transformer"]),
  ("4. Deep learning foundations", ["perceptron-learning-rule", "activation-functions", "mlp-forward-pass",
                                    "softmax-cross-entropy", "backpropagation", "micrograd-autograd-engine",
                                    "weight-initialization", "momentum-sgd", "adam-optimizer",
                                    "adamw-decoupled-weight-decay", "dropout", "batch-normalization",
                                    "pytorch-training-loop-anatomy", "gpu-computing-roofline"]),
]

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
        print("warning: cards in no sheet:", orphans)

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
