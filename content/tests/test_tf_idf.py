import numpy as np, yaml
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "tf-idf.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
docs = ["the cat sat on the mat", "the dog sat on the log",
        "cats and dogs are pets", "the mat is on the log and the cat"]
X, vocab = ns["tfidf"](docs)
ref = TfidfVectorizer(smooth_idf=True, norm="l2", sublinear_tf=False)
R = ref.fit_transform(docs).toarray()
assert vocab == list(ref.get_feature_names_out())
assert np.allclose(X, R)
assert np.allclose(np.linalg.norm(X, axis=1), 1)
print("tf-idf OK")
