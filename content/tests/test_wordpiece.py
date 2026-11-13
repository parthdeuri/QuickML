import tempfile, os, yaml
from pathlib import Path
from transformers import BertTokenizer

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "wordpiece.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

vocab_list = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "un", "##able", "##believ", "##believable",
              "play", "##ing", "##ed", "##s", "a", "##b", "##l", "##e", "h", "##u", "##g", "hug", "##ug"]
vocab = set(vocab_list)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "vocab.txt")
    Path(p).write_text("\n".join(vocab_list) + "\n", encoding="utf-8")
    bert = BertTokenizer(p, do_lower_case=False)
    words = ["unable", "playing", "played", "plays", "hugs", "hug", "unbelievable", "xyz", "unplayable", "ab", "able", "hugging"]
    for w in words:
        mine = ns["wordpiece_tokenize"](w, vocab)
        assert mine == bert.tokenize(w), (w, mine, bert.tokenize(w))
assert ns["wordpiece_tokenize"]("playing", vocab) == ["play", "##ing"]
assert ns["wordpiece_tokenize"]("unbelievable", vocab) == ["un", "##believable"]
assert ns["wordpiece_tokenize"]("xyz", vocab) == ["[UNK]"]
assert ns["wordpiece_tokenize"]("hugging", vocab) == ["hug", "##g", "##ing"]
assert ns["wordpiece_tokenize"]("hugx", vocab) == ["[UNK]"]  # whole word is [UNK]

# score picks a different pair than raw frequency
freqs = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}
splits = {w: ns["split_word"](w) for w in freqs}
assert splits["hug"] == ["h", "##u", "##g"]
by_freq = ns["best_pair"](splits, freqs, use_score=False)
by_score = ns["best_pair"](splits, freqs, use_score=True)
assert by_freq == ("##u", "##g"), by_freq
assert by_score != by_freq
one, two = ns["pair_stats"](splits, freqs)
assert two[by_freq] == 20 and one["##u"] == 36
# hand check of the score
a, b = by_score
assert abs(two[by_score] / (one[a] * one[b]) - max(two[p] / (one[p[0]] * one[p[1]]) for p in two)) < 1e-15
# merging
m = ns["merge"](splits, ("h", "##u"))
assert m["hug"] == ["hu", "##g"] and m["pun"] == ["p", "##u", "##n"]
m2 = ns["merge"](splits, ("##u", "##g"))
assert m2["hugs"] == ["h", "##ug", "##s"]
print("wordpiece OK")
