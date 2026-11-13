import yaml
from pathlib import Path
from nltk.stem import PorterStemmer

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "text-preprocessing-stemming.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# words covered by the simplified rules must agree with the real Porter stemmer
porter = PorterStemmer()
words = ("caresses ponies cats caress plastered motoring sing hopping "
         "falling hissing sized happy sky running jumped tanned quickly").split()
for w in words:
    assert ns["stem"](w) == porter.stem(w), (w, ns["stem"](w), porter.stem(w))

# pipeline on a fixed sentence
s = "The cats are running quickly, and the ponies jumped over walls!"
assert ns["normalise"](s)[:3] == ["the", "cats", "are"]
out = ns["pipeline"](s)
assert out == ["cat", "run", "quickli", "poni", "jump", "wall"], out
assert out == [porter.stem(t) for t in ns["normalise"](s) if t not in ns["STOP"]]

# lemmatiser handles irregular forms the stemmer cannot
assert ns["stem"]("ran") == "ran" and ns["lemmatise"]("ran") == "run"
assert ns["pipeline"]("The mice ran", use_stem=False) == ["mouse", "run"]
print("text-preprocessing-stemming OK")
