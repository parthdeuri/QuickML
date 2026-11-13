import yaml
d = yaml.safe_load(open("content/patterns/compare-decoder-vs-encoder-vs-encoder-decoder.yaml"))
ns = {}
exec(d["template"], ns)
assert list(ns["seen_dec"]) == [1, 2, 3, 4]
assert list(ns["seen_enc"]) == [4, 4, 4, 4]
assert ns["causal_mask"](4)[1, 2] == 0 and ns["bidirectional_mask"](4)[1, 2] == 1
assert ns["cross_mask"](3, 5).shape == (3, 5) and ns["cross_mask"](3, 5).all()
assert ns["decoder_loss_positions"](5) == [0, 1, 2, 3]
assert ns["encoder_loss_positions"]({3, 1}) == [1, 3]
print("pattern compare-decoder-vs-encoder-vs-encoder-decoder OK")
