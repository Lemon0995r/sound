"""Голосовой процессор: микрофон -> шумоподавление/смена голоса -> виртуальный микрофон."""
import sys
import tkinter as tk
from collections import deque
from tkinter import messagebox, ttk

import numpy as np
import sounddevice as sd

from dsp import HOP, SR, VoiceProcessor

PRESETS = {
    "Обычный": (0.0, False),
    "Низкий": (-5.0, False),
    "Высокий": (5.0, False),
    "Монстр": (-9.0, False),
    "Бурундук": (9.0, False),
    "Робот": (-2.0, True),
}


def list_devices():
    hostapis = sd.query_hostapis()
    ins, outs = [], []
    for i, d in enumerate(sd.query_devices()):
        api = hostapis[d["hostapi"]]["name"]
        if sys.platform == "win32" and "WASAPI" not in api:
            continue  # на Windows оставляем только WASAPI, без дублей
        if d["max_input_channels"] > 0:
            ins.append((i, d["name"]))
        if d["max_output_channels"] > 0:
            outs.append((i, d["name"]))
    return ins, outs


class Engine:
    def __init__(self, proc):
        self.proc = proc
        self.q_out = deque()
        self.q_mon = deque()
        self.streams = []
        self.monitor_on = False
        self.running = False

    @staticmethod
    def _push(q, y):
        q.append(y)
        while len(q) > 6:
            q.popleft()

    @staticmethod
    def _extra(dev):
        info = sd.query_devices(dev)
        api = sd.query_hostapis(info["hostapi"])["name"]
        return sd.WasapiSettings(auto_convert=True) if "WASAPI" in api else None

    def start(self, in_dev, out_dev, mon_dev):
        self.stop()
        self.q_out.clear()
        self.q_mon.clear()
        self.monitor_on = mon_dev is not None

        def in_cb(indata, frames, t, status):
            y = self.proc.process(indata[:, 0])
            self._push(self.q_out, y)
            if self.monitor_on:
                self._push(self.q_mon, y)

        def make_out_cb(q):
            def cb(outdata, frames, t, status):
                try:
                    b = q.popleft()
                except IndexError:
                    outdata.fill(0)
                    return
                outdata[:] = b[:, None]
            return cb

        def out_stream(dev, q):
            ch = min(2, sd.query_devices(dev)["max_output_channels"])
            return sd.OutputStream(device=dev, samplerate=SR, blocksize=HOP, channels=ch,
                                   dtype="float32", latency="low",
                                   callback=make_out_cb(q), extra_settings=self._extra(dev))

        try:
            self.streams.append(sd.InputStream(
                device=in_dev, samplerate=SR, blocksize=HOP, channels=1, dtype="float32",
                latency="low", callback=in_cb, extra_settings=self._extra(in_dev)))
            self.streams.append(out_stream(out_dev, self.q_out))
            if mon_dev is not None:
                self.streams.append(out_stream(mon_dev, self.q_mon))
            for s in self.streams:
                s.start()
            self.running = True
        except Exception:
            self.stop()
            raise

    def stop(self):
        for s in self.streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self.streams = []
        self.running = False


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Голосовой процессор")
        self.geometry("520x720")
        self.minsize(480, 680)
        self.proc = VoiceProcessor()
        self.engine = Engine(self.proc)
        self.ins, self.outs = list_devices()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(50, self._tick)

    # ---------- интерфейс ----------
    def _build(self):
        pad = {"padx": 12, "pady": 6}

        dev = ttk.LabelFrame(self, text="Устройства")
        dev.pack(fill="x", **pad)
        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.mon_var = tk.StringVar(value="Выкл (не слушать себя)")
        in_names = [n for _, n in self.ins]
        out_names = [n for _, n in self.outs]

        ttk.Label(dev, text="Микрофон (вход):").pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Combobox(dev, textvariable=self.in_var, values=in_names, state="readonly").pack(fill="x", padx=8)
        ttk.Label(dev, text="Виртуальный микрофон (выход, напр. «CABLE Input»):").pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Combobox(dev, textvariable=self.out_var, values=out_names, state="readonly").pack(fill="x", padx=8)
        ttk.Label(dev, text="Слушать себя в наушниках (по желанию):").pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Combobox(dev, textvariable=self.mon_var, values=["Выкл (не слушать себя)"] + out_names,
                     state="readonly").pack(fill="x", padx=8, pady=(0, 8))

        self._default_devices(in_names, out_names)

        self.btn = ttk.Button(self, text="Запустить", command=self._toggle)
        self.btn.pack(fill="x", padx=12, pady=6, ipady=6)
        self.status = ttk.Label(self, text="Остановлено", foreground="#888")
        self.status.pack(anchor="w", padx=14)

        lv = ttk.LabelFrame(self, text="Уровень (после шумоподавления)")
        lv.pack(fill="x", **pad)
        self.meter = ttk.Progressbar(lv, maximum=100)
        self.meter.pack(fill="x", padx=8, pady=(8, 2))
        self.gate_lbl = ttk.Label(lv, text="Гейт: —")
        self.gate_lbl.pack(anchor="w", padx=8, pady=(0, 6))

        ns = ttk.LabelFrame(self, text="Шумоподавление")
        ns.pack(fill="x", **pad)
        self._check(ns, "Шумоподавление", True, lambda v: setattr(self.proc, "denoise", v))
        self._check(ns, "Срез низких частот (гул, стук)", True, lambda v: setattr(self.proc, "hpf", v))
        self._check(ns, "Шумовой гейт (глушит паузы)", True, lambda v: setattr(self.proc, "gate_on", v))
        self._slider(ns, "Сила шумоподавления", 0, 4, 1.5, lambda v: f"{v:.1f}",
                     lambda v: setattr(self.proc, "strength", v))
        self._slider(ns, "Порог гейта", -80, -20, -50, lambda v: f"{v:.0f} дБ",
                     lambda v: setattr(self.proc, "gate_db", v))
        ttk.Button(ns, text="Откалибровать шум (1 сек тишины)",
                   command=self._calibrate).pack(fill="x", padx=8, pady=(2, 8))

        vc = ttk.LabelFrame(self, text="Голос")
        vc.pack(fill="x", **pad)
        row = ttk.Frame(vc)
        row.pack(fill="x", padx=8, pady=6)
        for i, (name, (semi, robot)) in enumerate(PRESETS.items()):
            ttk.Button(row, text=name, command=lambda s=semi, r=robot: self._preset(s, r)
                       ).grid(row=i // 3, column=i % 3, sticky="ew", padx=2, pady=2)
        for c in range(3):
            row.columnconfigure(c, weight=1)
        self.pitch_var = self._slider(vc, "Высота тона", -12, 12, 0, lambda v: f"{v:+.1f} пт",
                                      lambda v: setattr(self.proc, "semitones", v))
        self.robot_var = self._check(vc, "Эффект робота", False, lambda v: setattr(self.proc, "robot", v))
        self._slider(vc, "Громкость", 0, 2, 1, lambda v: f"{int(v * 100)}%",
                     lambda v: setattr(self.proc, "volume", v))

    def _default_devices(self, in_names, out_names):
        try:
            din = sd.query_devices(kind="input")["name"]
            dout = sd.query_devices(kind="output")["name"]
        except Exception:
            din = dout = ""
        pick_in = next((n for n in in_names if n == din or n.startswith(din[:25])), in_names[0] if in_names else "")
        # не берём виртуальный кабель в качестве микрофона
        if "cable" in pick_in.lower():
            pick_in = next((n for n in in_names if "cable" not in n.lower()), pick_in)
        cable = next((n for n in out_names if "cable input" in n.lower()), None)
        pick_out = cable or next((n for n in out_names if n == dout), out_names[0] if out_names else "")
        self.in_var.set(pick_in)
        self.out_var.set(pick_out)
        self.cable_found = cable is not None

    def _check(self, parent, text, init, cb):
        var = tk.BooleanVar(value=init)
        var.trace_add("write", lambda *_: cb(var.get()))
        ttk.Checkbutton(parent, text=text, variable=var).pack(anchor="w", padx=8, pady=2)
        cb(init)
        return var

    def _slider(self, parent, text, lo, hi, init, fmt, cb):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ttk.Label(row, text=text, width=20).pack(side="left")
        lab = ttk.Label(row, width=9, anchor="e")
        lab.pack(side="right")
        var = tk.DoubleVar(value=init)

        def upd(*_):
            v = var.get()
            lab.config(text=fmt(v))
            cb(v)

        var.trace_add("write", upd)
        ttk.Scale(row, from_=lo, to=hi, variable=var).pack(side="left", fill="x", expand=True, padx=6)
        upd()
        return var

    # ---------- действия ----------
    def _preset(self, semi, robot):
        self.pitch_var.set(semi)
        self.robot_var.set(robot)

    def _calibrate(self):
        self.proc.calibrate_noise()
        self.status.config(text="Калибровка шума… помолчи 1 секунду", foreground="#d98c00")

    def _id(self, items, name):
        return next((i for i, n in items if n == name), None)

    def _toggle(self):
        if self.engine.running:
            self.engine.stop()
            self._set_state(False)
            return
        in_id = self._id(self.ins, self.in_var.get())
        out_id = self._id(self.outs, self.out_var.get())
        mon_id = self._id(self.outs, self.mon_var.get())
        if in_id is None or out_id is None:
            messagebox.showerror("Ошибка", "Выбери микрофон и выход.")
            return
        if "cable input" not in self.out_var.get().lower() and mon_id is None:
            if not messagebox.askyesno(
                    "Выход не виртуальный",
                    "Выбранный выход — не виртуальный кабель, другие программы не получат обработанный звук.\n"
                    "Продолжить (ты будешь слышать результат на этом выходе)?"):
                return
        try:
            self.engine.start(in_id, out_id, mon_id)
        except Exception as e:
            messagebox.showerror("Не удалось запустить звук", str(e))
            return
        self._set_state(True)

    def _set_state(self, on):
        self.btn.config(text="Остановить" if on else "Запустить")
        self.status.config(text="Работает" if on else "Остановлено",
                           foreground="#1a9b5b" if on else "#888")

    def _tick(self):
        db = self.proc.level_db if self.engine.running else -90
        self.meter["value"] = max(0, min(100, (db + 80) / 80 * 100))
        if self.engine.running:
            self.gate_lbl.config(text="Гейт: открыт" if self.proc.gate_open else "Гейт: закрыт")
            if self.proc._calib == 0 and self.status.cget("text").startswith("Калибровка"):
                self._set_state(True)
        else:
            self.gate_lbl.config(text="Гейт: —")
        self.after(50, self._tick)

    def _close(self):
        self.engine.stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
