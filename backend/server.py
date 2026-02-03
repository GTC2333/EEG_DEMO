import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass

import websockets


BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


@dataclass
class EmotionState:
    label: str
    valence: float
    arousal: float
    confidence: float


def classify_emotion(
    valence: float,
    arousal: float,
    t_low: float = 0.4,
    t_high: float = 0.6,
    v_eps: float = 0.15,
) -> str:
    if arousal > t_high:
        return "Excited" if valence > 0 else "Stressed"
    if arousal < t_low:
        return "Calm" if valence > 0 else "Fatigued"
    if abs(valence) <= v_eps:
        return "Focused"
    return "Focused"


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _gauss(rng: random.Random, mu: float = 0.0, sigma: float = 1.0) -> float:
    return rng.gauss(mu, sigma)


class _AR2:
    def __init__(self, rng: random.Random, a1: float, a2: float, sigma: float):
        self.rng = rng
        self.a1 = a1
        self.a2 = a2
        self.sigma = sigma
        self.x1 = 0.0
        self.x2 = 0.0

    def step(self) -> float:
        x = self.a1 * self.x1 + self.a2 * self.x2 + _gauss(self.rng, 0.0, self.sigma)
        self.x2 = self.x1
        self.x1 = x
        return x


class MockEEG:
    def __init__(self, channels: list[str], fs: float = 128.0, seed: int = 7):
        self.channels = channels
        self.fs = fs
        self.rng = random.Random(seed)
        self.t = 0.0

        self._delta_f = [self.rng.uniform(1.2, 3.0) for _ in channels]
        self._theta_f = [self.rng.uniform(4.0, 7.0) for _ in channels]
        self._alpha_f = [self.rng.uniform(8.0, 12.0) for _ in channels]
        self._beta_f = [self.rng.uniform(14.0, 22.0) for _ in channels]
        self._gamma_f = [self.rng.uniform(32.0, 42.0) for _ in channels]

        self._delta_phi = [self.rng.uniform(0, 2 * math.pi) for _ in channels]
        self._theta_phi = [self.rng.uniform(0, 2 * math.pi) for _ in channels]
        self._alpha_phi = [self.rng.uniform(0, 2 * math.pi) for _ in channels]
        self._beta_phi = [self.rng.uniform(0, 2 * math.pi) for _ in channels]
        self._gamma_phi = [self.rng.uniform(0, 2 * math.pi) for _ in channels]

        # simple symmetric mixing to introduce cross-channel correlation
        n = len(channels)
        self._mix_main = 0.78
        self._mix_nei = 0.18
        self._mix_far = 0.04
        self._mix_buf = [0.0] * n

        self._noise = [_AR2(self.rng, a1=1.60, a2=-0.64, sigma=0.7) for _ in channels]

        # low-freq global drift (electrode movement / skin potential)
        self._drift_phi = self.rng.uniform(0, 2 * math.pi)

        # artifacts
        self._blink_timer = self.rng.uniform(1.0, 3.0)
        self._blink_amp = 0.0
        self._blink_decay = 0.0

        self._emg_timer = self.rng.uniform(0.8, 2.5)
        self._emg_amp = 0.0
        self._emg_decay = 0.0

        # channel weights for artifacts
        self._fp_weight = {"Fp1": 1.0, "Fp2": 1.0, "F7": 0.6, "F8": 0.6, "F3": 0.45, "F4": 0.45}
        self._temporal_weight = {"T7": 1.0, "T8": 1.0, "P7": 0.4, "P8": 0.4}

    def _update_artifacts(self, dt: float):
        self._blink_timer -= dt
        if self._blink_timer <= 0:
            self._blink_timer = self.rng.uniform(2.0, 5.0)
            self._blink_amp = self.rng.uniform(35.0, 70.0)
            self._blink_decay = self.rng.uniform(0.15, 0.30)

        if self._blink_amp > 0:
            self._blink_amp *= math.exp(-dt / max(1e-6, self._blink_decay))

        self._emg_timer -= dt
        if self._emg_timer <= 0:
            self._emg_timer = self.rng.uniform(1.2, 4.0)
            self._emg_amp = self.rng.uniform(8.0, 20.0)
            self._emg_decay = self.rng.uniform(0.25, 0.60)

        if self._emg_amp > 0:
            self._emg_amp *= math.exp(-dt / max(1e-6, self._emg_decay))

    def _state_envelopes(self) -> tuple[float, float]:
        # Slowly varying "state" envelopes (0..1) that drive alpha/beta balance
        # valence driven by F4-F3 alpha asymmetry
        s = 0.5 + 0.5 * math.sin(2 * math.pi * 0.03 * self.t + 0.9)
        alpha_env = _clamp(0.55 + 0.35 * (1 - s), 0.15, 0.95)
        beta_env = _clamp(0.35 + 0.55 * s, 0.10, 1.00)
        return alpha_env, beta_env

    def sample_window(self, window_s: float = 1.0) -> list[list[float]]:
        n = max(1, int(self.fs * window_s))
        out = []
        for _ in range(n):
            out.append(self._sample_one())
        return out

    def _sample_one(self) -> list[float]:
        dt = 1.0 / self.fs
        self.t += dt
        self._update_artifacts(dt)
        alpha_env, beta_env = self._state_envelopes()

        drift = 6.0 * math.sin(2 * math.pi * 0.12 * self.t + self._drift_phi) + 3.0 * math.sin(2 * math.pi * 0.05 * self.t)

        # frontal alpha asymmetry: modulate F3/F4 alpha a bit differently over time
        aas = 0.18 * math.sin(2 * math.pi * 0.015 * self.t + 1.7)

        row = []
        for i, ch in enumerate(self.channels):
            # add explicit delta/theta/alpha/beta/gamma components
            delta = 18.0 * (0.45 + 0.25 * math.sin(2 * math.pi * 0.010 * self.t + 0.3 * i)) * math.sin(2 * math.pi * self._delta_f[i] * self.t + self._delta_phi[i])

            theta = 10.0 * (0.55 + 0.25 * math.sin(2 * math.pi * 0.02 * self.t + i)) * math.sin(2 * math.pi * self._theta_f[i] * self.t + self._theta_phi[i])

            a_scale = 22.0 * alpha_env
            if ch == "F3":
                a_scale *= (1 - aas)
            elif ch == "F4":
                a_scale *= (1 + aas)
            alpha = a_scale * math.sin(2 * math.pi * self._alpha_f[i] * self.t + self._alpha_phi[i])

            beta = (10.0 * beta_env) * math.sin(2 * math.pi * self._beta_f[i] * self.t + self._beta_phi[i])

            gamma = (2.8 + 1.8 * beta_env) * math.sin(2 * math.pi * self._gamma_f[i] * self.t + self._gamma_phi[i])

            noise = 2.5 * self._noise[i].step()  # colored noise

            # blink: sharp positive transient mostly on frontal channels
            blink_w = self._fp_weight.get(ch, 0.12)
            blink = blink_w * self._blink_amp

            # emg: high-frequency "buzz" on temporal channels
            emg_w = self._temporal_weight.get(ch, 0.08)
            emg = emg_w * self._emg_amp * math.sin(2 * math.pi * (55.0 + 15.0 * math.sin(2 * math.pi * 0.2 * self.t)) * self.t + 0.3 * i)

            v = 0.45 * alpha + 0.24 * beta + 0.16 * theta + 0.22 * delta + 0.08 * gamma + drift + noise + blink + emg
            row.append(float(v))

        # mix to make channels correlated (volume conduction like)
        n = len(row)
        for i in range(n):
            prev = row[i - 1] if i - 1 >= 0 else row[i]
            nxt = row[i + 1] if i + 1 < n else row[i]
            self._mix_buf[i] = self._mix_main * row[i] + self._mix_nei * (prev + nxt) / 2 + self._mix_far * (drift)
        return [float(x) for x in self._mix_buf]


class MockBandPower:
    def __init__(self, channels: list[str], fs: float = 128.0):
        self.channels = channels
        self.fs = fs

    def _band_energy(self, samples: list[float], f_lo: float, f_hi: float) -> float:
        # naive DFT band energy; window size kept small (<= 128) so it's fast enough
        n = len(samples)
        if n < 4:
            return 0.0
        # Hann window
        win = [0.5 * (1 - math.cos((2 * math.pi * i) / (n - 1))) for i in range(n)]
        x = [samples[i] * win[i] for i in range(n)]

        k_lo = int(math.floor(f_lo * n / self.fs))
        k_hi = int(math.ceil(f_hi * n / self.fs))
        k_lo = max(1, min(k_lo, n // 2))
        k_hi = max(k_lo + 1, min(k_hi, n // 2))

        e = 0.0
        for k in range(k_lo, k_hi):
            re = 0.0
            im = 0.0
            ang0 = (2 * math.pi * k) / n
            for i in range(n):
                ang = ang0 * i
                re += x[i] * math.cos(ang)
                im -= x[i] * math.sin(ang)
            e += re * re + im * im
        return e / max(1.0, (k_hi - k_lo))

    def from_window(self, window: list[list[float]]) -> dict:
        # window: [n_samples][n_channels]
        if not window:
            n_ch = len(self.channels)
            return {
                "schema": "band_power.v1",
                "channels": self.channels,
                "bands": BANDS,
                "values": {b: [0.0] * n_ch for b in BANDS},
            }

        n_ch = len(window[0])
        by_ch = list(zip(*window))  # [n_channels][n_samples]

        bands = {b: [] for b in BANDS}
        for ch_sig in by_ch:
            delta = self._band_energy(ch_sig, 1.0, 4.0)
            theta = self._band_energy(ch_sig, 4.0, 8.0)
            alpha = self._band_energy(ch_sig, 8.0, 13.0)
            beta = self._band_energy(ch_sig, 13.0, 30.0)
            gamma = self._band_energy(ch_sig, 30.0, 45.0)

            # keep strictly positive
            bands["delta"].append(float(max(1e-6, delta)))
            bands["theta"].append(float(max(1e-6, theta)))
            bands["alpha"].append(float(max(1e-6, alpha)))
            bands["beta"].append(float(max(1e-6, beta)))
            bands["gamma"].append(float(max(1e-6, gamma)))

        return {
            "schema": "band_power.v1",
            "channels": self.channels,
            "bands": BANDS,
            "values": bands,
        }


_LATENT_STATE = {
    "t": 0.0,
    "vx": 0.0,
    "ax": 0.0,
}


def estimate_from_bandpower(bandpower: dict) -> EmotionState:
    chs = bandpower["channels"]
    delta = bandpower["values"]["delta"]
    theta = bandpower["values"]["theta"]
    alpha = bandpower["values"]["alpha"]
    beta = bandpower["values"]["beta"]
    gamma = bandpower["values"]["gamma"]

    def _idx(name: str) -> int:
        try:
            return chs.index(name)
        except ValueError:
            return 0

    li = _idx("F3")
    ri = _idx("F4")

    alpha_left = alpha[li]
    alpha_right = alpha[ri]

    # Observable features (normalized)
    denom_lr = max(1e-6, (alpha_left + alpha_right) / 2)
    alpha_asym = (alpha_right - alpha_left) / denom_lr  # ~[-1,1]

    alpha_mean = sum(alpha) / len(alpha)
    beta_mean = sum(beta) / len(beta)
    gamma_mean = sum(gamma) / len(gamma)
    theta_mean = sum(theta) / len(theta)
    delta_mean = sum(delta) / len(delta)

    # arousal proxy: high-freq / alpha, with some theta contribution
    hf = beta_mean + 0.6 * gamma_mean
    arousal_proxy = hf / max(1e-6, alpha_mean)

    # fatigue proxy: (delta+theta) / beta
    slow_proxy = (delta_mean + 0.8 * theta_mean) / max(1e-6, beta_mean)

    # Add a slow 2D latent state so VA doesn't collapse to a line.
    # Simple OU-like update driven by bandpower features.
    t = time.time()
    dt = 0.5 if _LATENT_STATE["t"] == 0.0 else max(0.05, min(1.0, t - _LATENT_STATE["t"]))
    _LATENT_STATE["t"] = t

    # independent noises
    nx = random.gauss(0.0, 0.08)
    ny = random.gauss(0.0, 0.08)

    # push latent by independent drivers
    drive_v = 0.55 * _clamp(alpha_asym, -1.0, 1.0) + 0.25 * math.tanh(0.6 * (1.0 - slow_proxy))
    drive_a = 0.55 * math.tanh(0.8 * (arousal_proxy - 1.0)) - 0.20 * math.tanh(0.7 * (slow_proxy - 1.0))

    # mean reversion + driven + noise
    vx = 0.92 * _LATENT_STATE["vx"] + 0.20 * drive_v + nx
    ax = 0.90 * _LATENT_STATE["ax"] + 0.24 * drive_a + ny
    _LATENT_STATE["vx"] = vx
    _LATENT_STATE["ax"] = ax

    # map to valence/arousal in [0,1]
    valence = 0.5 + 0.5 * math.tanh(vx)
    arousal = 0.5 + 0.5 * math.tanh(ax)

    label = classify_emotion(valence - 0.5, arousal)

    # confidence: lower when very noisy or very ambiguous
    confidence = float(_clamp(0.65 + 0.25 * (1 - abs((valence - 0.5) * 2)) + 0.10 * (1 - abs((arousal - 0.5) * 2)), 0.55, 0.95))

    return EmotionState(label=label, valence=float(valence), arousal=float(arousal), confidence=confidence)


async def producer(ws, *, channels: list[str], eeg_hz: float = 10.0, emotion_hz: float = 2.0):
    eeg_period = 1.0 / eeg_hz
    emotion_period = 1.0 / emotion_hz

    fs = 128.0
    mock_eeg = MockEEG(channels, fs=fs, seed=7)
    mock_bp = MockBandPower(channels, fs=fs)

    last_emotion_t = 0.0
    emotion = EmotionState(label="Calm", valence=0.0, arousal=0.5, confidence=0.75)
    last_bandpower = {
        "schema": "band_power.v1",
        "channels": channels,
        "bands": BANDS,
        "values": {b: [0.0] * len(channels) for b in BANDS},
    }

    while True:
        now = time.time()

        window = mock_eeg.sample_window(window_s=1.0)
        eeg = {"schema": "eeg_waveform.v1", "channels": channels, "waveform": [window[-1]]}

        if now - last_emotion_t >= emotion_period:
            last_bandpower = mock_bp.from_window(window)
            emotion = estimate_from_bandpower(last_bandpower)
            last_emotion_t = now

        payload = {
            "timestamp": now,
            "source": {"mode": "mock", "schema": "emotiv_compat.v1"},
            "emotion": asdict(emotion),
            "band_power": last_bandpower,
            "eeg": eeg,
        }
        await ws.send(json.dumps(payload))
        await asyncio.sleep(eeg_period)


async def handler(ws):
    channels = [
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T7",
        "T8",
        "P7",
        "P8",
        "Fp1",
        "Fp2",
    ]
    await ws.send(json.dumps({"type": "hello", "channels": channels, "bands": BANDS}))
    await producer(ws, channels=channels)


async def main():
    host = "0.0.0.0"
    port = int(__import__("os").environ.get("EEG_DEMO_WS_PORT", "3012"))

    async def _handler(ws):
        try:
            await handler(ws)
        except websockets.ConnectionClosed:
            return

    print(f"WS server listening on ws://{host}:{port}")
    async with websockets.serve(_handler, host, port):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
