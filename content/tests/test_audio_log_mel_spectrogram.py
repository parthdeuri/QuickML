import numpy as np, yaml, librosa
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "audio-log-mel-spectrogram.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

sr, n_fft, hop, n_mels = 16000, 512, 160, 40
rng = np.random.default_rng(0)
x = rng.normal(size=5000)

# frame count formula (no padding)
P = ns["stft_power"](x, n_fft, hop)
assert P.shape == (1 + (len(x) - n_fft) // hop, n_fft // 2 + 1)

# filterbank equals librosa's (HTK mel, no area normalisation)
fb = ns["mel_filterbank"](sr, n_fft, n_mels)
ref_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, htk=True, norm=None, dtype=np.float64)
assert np.allclose(fb, ref_fb, rtol=1e-6, atol=1e-12), np.abs(fb - ref_fb).max()

# power mel spectrogram equals librosa (periodic hann, center=False)
mel = P @ fb.T
ref = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop, center=False, power=2.0, window="hann", htk=True, norm=None, n_mels=n_mels, dtype=np.float64)
assert ref.shape == mel.T.shape
assert np.allclose(mel.T, ref, rtol=1e-6, atol=1e-9), np.abs(mel.T - ref).max()
L = ns["log_mel"](x, sr, n_fft, hop, n_mels)
assert np.allclose(L, np.log(ref.T + 1e-10), rtol=1e-6, atol=1e-6)

# hz <-> mel round trip and the 1000 Hz anchor of the HTK formula
f = np.array([0.0, 100.0, 1000.0, 7999.0])
assert np.allclose(ns["mel_to_hz"](ns["hz_to_mel"](f)), f)
assert abs(ns["hz_to_mel"](1000.0) - 1000.0) < 0.1

# a pure tone concentrates its energy in the mel band containing it
t = np.arange(8000) / sr
for f0 in (300.0, 1000.0, 3000.0):
    M = ns["stft_power"](np.sin(2 * np.pi * f0 * t), n_fft, hop) @ fb.T
    band = M.mean(0).argmax()
    centres = fb.argmax(1) * sr / n_fft
    lo = ns["mel_to_hz"](np.linspace(0, ns["hz_to_mel"](sr / 2), n_mels + 2))
    assert lo[band] <= f0 <= lo[band + 2], (f0, band)
    assert M.mean(0)[band] > 0.3 * M.mean(0).sum()
print("audio-log-mel-spectrogram OK")
