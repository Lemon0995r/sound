"""Обработка голоса в реальном времени: фильтр, шумоподавление, гейт, смена тона, робот."""
import numpy as np
from scipy.signal import butter, lfilter

SR = 48000          # частота дискретизации
HOP = 480           # размер блока (10 мс)
WIN = HOP * 2       # окно STFT (50% перекрытие)


class VoiceProcessor:
    def __init__(self):
        # --- параметры (меняются из GUI) ---
        self.denoise = True
        self.strength = 1.5      # сила шумоподавления
        self.gate_on = True
        self.gate_db = -50.0     # порог гейта, дБ
        self.hpf = True          # срез низких частот
        self.semitones = 0.0     # смена тона
        self.robot = False
        self.volume = 1.0
        # --- состояние для индикаторов ---
        self.level_db = -90.0
        self.gate_open = True

        # STFT-шумодав
        self._window = np.sin(np.pi * (np.arange(WIN) + 0.5) / WIN)
        self._prev = np.zeros(HOP)
        self._tail = np.zeros(HOP)
        self._noise = None
        self._gain = np.ones(HOP + 1)
        self._calib = 0

        # фильтр высоких частот 90 Гц
        self._b, self._a = butter(2, 90 / (SR / 2), "high")
        self._zi = np.zeros(2)

        # гейт
        self._hold = 0
        self._gate_g = 1.0

        # питч-шифтер (две перекрывающиеся линии задержки)
        self._buf = np.zeros(1 << 16)
        self._w = 0
        self._phase = 0.0
        self._pwin = int(SR * 0.05)
        self._idx = np.arange(HOP)

        # робот (кольцевая модуляция)
        self._rph = 0.0

    # ---------- публичное ----------
    def calibrate_noise(self):
        """Запомнить фоновый шум (~1 с тишины)."""
        self._noise = None
        self._calib = 100

    def process(self, block):
        x = np.asarray(block, dtype=np.float64)
        if self.hpf:
            x, self._zi = lfilter(self._b, self._a, x, zi=self._zi)
        if self.denoise:
            y = self._denoise(x)
        else:
            self._prev = x.copy()
            self._tail[:] = 0
            y = x
        y = self._gate(y)
        if abs(self.semitones) >= 0.05:
            y = self._pitch(y)
        if self.robot:
            f = 55.0
            ph = self._rph + 2 * np.pi * f * self._idx / SR
            y = y * np.sin(ph)
            self._rph = (self._rph + 2 * np.pi * f * HOP / SR) % (2 * np.pi)
        y = np.tanh(y * self.volume)
        return y.astype(np.float32)

    # ---------- этапы ----------
    def _denoise(self, x):
        frame = np.concatenate((self._prev, x))
        self._prev = x.copy()
        spec = np.fft.rfft(frame * self._window)
        mag = np.abs(spec)
        if self._noise is None:
            self._noise = mag.copy()
        if self._calib > 0:
            self._noise = 0.9 * self._noise + 0.1 * mag
            self._calib -= 1
        else:
            # адаптивная оценка: быстро вниз, медленно вверх
            self._noise = np.where(mag < self._noise,
                                   0.9 * self._noise + 0.1 * mag,
                                   self._noise * 1.002)
        g = np.maximum(1.0 - self.strength * self._noise / (mag + 1e-9), 0.08)
        self._gain = 0.5 * self._gain + 0.5 * g
        out = np.fft.irfft(spec * self._gain, WIN) * self._window
        y = self._tail + out[:HOP]
        self._tail = out[HOP:].copy()
        return y

    def _gate(self, y):
        rms = np.sqrt(np.mean(y * y) + 1e-12)
        db = 20 * np.log10(rms + 1e-9)
        self.level_db = db
        if db > self.gate_db:
            self._hold = 15            # держим открытым ~150 мс
        elif self._hold > 0:
            self._hold -= 1
        is_open = self._hold > 0 or not self.gate_on
        self.gate_open = is_open
        target = 1.0 if is_open else 0.0
        g0 = self._gate_g
        g1 = g0 + float(np.clip(target - g0, -0.15, 0.6))
        self._gate_g = g1
        return y * np.linspace(g0, g1, HOP, endpoint=False)

    def _pitch(self, x):
        ratio = 2.0 ** (self.semitones / 12.0)
        win = self._pwin
        mask = len(self._buf) - 1
        n = len(x)
        w = self._w
        self._buf[(w + self._idx) & mask] = x
        ph1 = (self._phase + self._idx * (1.0 - ratio) / win) % 1.0
        ph2 = (ph1 + 0.5) % 1.0
        out = np.zeros(n)
        for ph in (ph1, ph2):
            pos = (w + self._idx) - (ph * win + 2.0)
            i0 = np.floor(pos).astype(np.int64)
            f = pos - i0
            v = self._buf[i0 & mask] * (1 - f) + self._buf[(i0 + 1) & mask] * f
            out += v * (0.5 - 0.5 * np.cos(2 * np.pi * ph))
        self._w = (w + n) & mask
        self._phase = (self._phase + n * (1.0 - ratio) / win) % 1.0
        return out
