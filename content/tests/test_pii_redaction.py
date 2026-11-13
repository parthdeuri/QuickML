import yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "pii-redaction.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
luhn, detect, redact, pr = (ns[k] for k in ("luhn_valid", "detect", "redact", "precision_recall"))

# Luhn: published test numbers pass, altered ones fail
for good in ["4111111111111111", "4111 1111 1111 1111", "5500-0000-0000-0004", "378282246310005"]:
    assert luhn(good), good
for bad in ["4111111111111112", "4111 1111 1111 1110", "1234567812345678", "411"]:
    assert not luhn(bad), bad

text = ("Mail user@example.com or USER@example.com, call (555) 010-0100. "
        "Card 4111 1111 1111 1111, typo 4111 1111 1111 1112, "
        "host 192.168.0.1, bad 999.1.1.1, cc other@example.com.")
exp = ("Mail <EMAIL_1> or <EMAIL_1>, call <PHONE_1>. "
       "Card <CARD_1>, typo 4111 1111 1111 1112, "
       "host <IP_1>, bad 999.1.1.1, cc <EMAIL_2>.")
assert redact(text) == exp, redact(text)
# same value -> same placeholder, different values -> different
out = redact("a@example.com b@example.com a@example.com")
assert out == "<EMAIL_1> <EMAIL_2> <EMAIL_1>"

# spans do not overlap, are sorted, and point at the right text
sp = detect(text)
assert [k for k, _ in sp] == ["EMAIL", "EMAIL", "PHONE", "CARD", "IP", "EMAIL"]
for (_, (a, b)), (_, (c, d)) in zip(sp, sp[1:]):
    assert a < b <= c < d
assert text[sp[3][1][0]:sp[3][1][1]] == "4111 1111 1111 1111"
assert text[sp[2][1][0]:sp[2][1][1]] == "(555) 010-0100"
# clean text is unchanged
assert redact("nothing personal here, order 42") == "nothing personal here, order 42"

# precision / recall by hand: tp=2, |pred|=3, |true|=4
pred = [("EMAIL", (0, 5)), ("IP", (10, 20)), ("CARD", (30, 40))]
true = [("EMAIL", (0, 5)), ("IP", (10, 20)), ("PHONE", (50, 60)), ("EMAIL", (70, 80))]
p, r = pr(pred, true)
assert abs(p - 2 / 3) < 1e-12 and abs(r - 0.5) < 1e-12
# detector on the synthetic text: perfect against its own labels
assert pr(sp, sp) == (1.0, 1.0)
print("pii-redaction OK")
