import yaml, numpy as np
d = yaml.safe_load(open("content/patterns/multimodal-and-speech.yaml", encoding="utf-8"))
assert d["id"] == "multimodal-and-speech"
ns = {}
exec(d["template"], ns)
img = np.arange(3 * 4 * 6, dtype=float).reshape(3, 4, 6)
p = ns["patchify"](img, 2)
assert p.shape == (ns["image_tokens"](4, 6, 2), 3 * 2 * 2) == (6, 12)
assert np.array_equal(p[0].reshape(3, 2, 2), img[:, :2, :2])
assert np.array_equal(p[1].reshape(3, 2, 2), img[:, :2, 2:4])
f = ns["sample_frames"](100, 5)
assert len(f) == 5 and f[0] == 0 and f[-1] == 99 and (np.diff(f) > 0).all()
assert ns["video_tokens"](100, 8, 224, 224, 16) == 8 * 196
assert ns["audio_frames"](16000, 400, 160) == 98
assert ns["audio_frames"](400, 400, 160) == 1
print("pattern multimodal-and-speech OK")
