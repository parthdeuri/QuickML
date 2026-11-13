import numpy as np, yaml, librosa
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "griffin-lim-waveform-reconstruction.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
stft, istft, gl = ns["stft"], ns["istft"], ns["griffin_lim"]

n_fft, hop, sr = 256, 64, 8000
t = np.arange(4096) / sr
x = np.sin(2 * np.pi * (200 * t + 300 * t ** 2))  # seeded-free chirp

# STFT equals librosa (center=False, periodic hann, no scaling)
X = stft(x, n_fft, hop)
ref = librosa.stft(x, n_fft=n_fft, hop_length=hop, center=False, window="hann").T
assert X.shape == ref.shape == (1 + (len(x) - n_fft) // hop, n_fft // 2 + 1)
assert np.allclose(X, ref, rtol=0, atol=1e-10), np.abs(X - ref).max()

# perfect reconstruction in the interior; edges (first/last window
# samples, where the window is ~0) are only covered by few frames
r = istft(X, n_fft, hop)
assert len(r) == len(x)
inner = slice(n_fft, len(x) - n_fft)
assert np.abs(r - x)[inner].max() < 1e-10, np.abs(r - x)[inner].max()

# Griffin-Lim: monotone spectral convergence on the chirp
S = np.abs(X)
y, errs = gl(S, 100, n_fft, hop, np.random.default_rng(0))
errs = np.array(errs)
assert np.all(np.diff(errs) <= 1e-12), np.diff(errs).max()
assert errs[-1] < 0.1 * errs[0], (errs[0], errs[-1])
final = np.linalg.norm(np.abs(stft(y, n_fft, hop)) - S) / np.linalg.norm(S)
assert final <= errs[-1] + 1e-12 and final < 0.1

# phase ambiguity: waveform need not equal the original
wave_err = np.linalg.norm(y[inner] - x[inner]) / np.linalg.norm(x[inner])
print("waveform relative error", round(wave_err, 3), "spectral", round(final, 4))

# same order of magnitude as librosa.griffinlim
yl = librosa.griffinlim(S.T, n_iter=100, hop_length=hop, win_length=n_fft, n_fft=n_fft, center=False, random_state=0, length=len(x))
lib = np.linalg.norm(np.abs(stft(yl, n_fft, hop)) - S) / np.linalg.norm(S)
assert final < 10 * lib + 1e-3 and lib < 10 * final + 1e-3, (final, lib)
print("griffin-lim-waveform-reconstruction OK")
