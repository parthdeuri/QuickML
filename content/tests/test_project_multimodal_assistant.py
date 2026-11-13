import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "project-multimodal-assistant.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
Assistant = ns["Assistant"]

def img(tl, tr, bl, br, n=32):
    a = np.zeros((n, n)); h = n // 2
    a[:h, :h], a[:h, h:], a[h:, :h], a[h:, h:] = tl, tr, bl, br
    return a

gallery = {"sea": img(1, 1, 0, 0), "night": img(0, 0, 0, 0), "lamp": img(1, 0, 1, 0)}
boot = Assistant([])
items = [(k, "image", boot.embed_image(a)) for k, a in gallery.items()]
items.append(("note1", "note", boot.embed_text("quarterly budget meeting notes")))
A = Assistant(items, budget=64)

# routing
assert A.route({"text": "hi"}) == "text"
assert A.route({"image": gallery["sea"]}) == "image"
assert A.route({"text": "hi", "image": gallery["sea"]}) == "both"
assert A.route({}) == "empty"
# a text query retrieves the right image, and notes by text
for q, want in [("sky over sea", "sea"), ("night", "night"), ("lamp", "lamp")]:
    assert A.retrieve({"text": q}) == [want], q
assert A.retrieve({"text": "budget notes"}) == ["note1"]
assert A.retrieve({"image": gallery["night"]}) == ["night"]
assert A.retrieve({"text": "lamp", "image": gallery["lamp"]}) == ["lamp"]
# token accounting: 32x32, P=4 -> 64 patches; budget 64 forces pooling k=2
p = A.build_prompt({"text": "what is this", "image": gallery["sea"]})
assert p["ok"] and p["text_tokens"] == 3 and p["pool"] == 2
assert p["image_tokens"] == (32 // 4) ** 2 // 2 ** 2 == 16
assert p["total"] == 3 + 16 <= 64
# 64x64 -> 256 patches, needs k=4 -> 16 tokens
p = A.build_prompt({"text": "x", "image": np.zeros((64, 64))})
assert p["pool"] == 4 and p["image_tokens"] == 16 and p["total"] == 17
# text only has no image tokens
p = A.build_prompt({"text": "a b c d"})
assert p["image_tokens"] == 0 and p["total"] == 4
# oversize: 128x128 -> 1024 patches, 64 at k=4, plus 1 text token > 64
p = A.build_prompt({"text": "x", "image": np.zeros((128, 128))})
assert not p["ok"] and "1024 patches" in p["reason"]
assert A.answer({"text": "x", "image": np.zeros((128, 128))}) == p["reason"]
# never over budget for any size
for n in (16, 32, 64, 128):
    p = A.build_prompt({"text": "a b", "image": np.zeros((n, n))})
    assert (not p["ok"]) or p["total"] <= 64
# deterministic and cites what it used
r = {"text": "sky sea", "image": gallery["sea"]}
assert A.answer(r) == A.answer(r) == "used sea"
print("project-multimodal-assistant OK")
