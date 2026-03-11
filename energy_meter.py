"""
energy_meter.py
---------------
GPU energy measurement via pynvml.
Samples power draw at a fixed interval using a background thread
and computes total Joules via the trapezoidal rule.

Usage:
    from energy_meter import EnergyMeter

    meter = EnergyMeter(device_idx=0, sample_interval=0.1)
    meter.start()
    # ... run inference ...
    meter.stop()
    print(f"Total energy: {meter.joules:.2f} J")
    print(f"Avg power:    {meter.avg_power_watts:.1f} W")
"""

import threading
import time
import os

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class EnergyMeter:
    """
    Samples GPU power draw, GPU utilization, and CPU utilization.
    Integrates power samples with the trapezoidal rule to get Joules.
    """

    def __init__(self, device_idx: int = 0, sample_interval: float = 0.1):
        self.device_idx = device_idx
        self.sample_interval = sample_interval
        self._samples: list = []   # list of (timestamp, watts, gpu_util, cpu_util)
        self._running = False
        self._thread = None
        self._handle = None

        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                gpu_name = pynvml.nvmlDeviceGetName(self._handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode()
                print(f"[EnergyMeter] Monitoring GPU {device_idx}: {gpu_name}")
            except Exception as e:
                print(f"[EnergyMeter] WARNING: pynvml init failed: {e}. Energy will be 0.")
                self._handle = None
        
        if not _PSUTIL_AVAILABLE:
            print("[EnergyMeter] WARNING: psutil not installed. CPU util will be 0.")

    def start(self):
        self._samples = []
        self._running = True
        # Pre-sample CPU to initialize psutil's internal counter
        if _PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval * 10)

    def _sample_loop(self):
        while self._running:
            ts = time.perf_counter()
            watts = self._read_power_watts()
            gpu_util = self._read_gpu_util()
            cpu_util = self._read_cpu_util()
            
            self._samples.append((ts, watts, gpu_util, cpu_util))
            time.sleep(self.sample_interval)

    def _read_power_watts(self) -> float:
        if self._handle is None:
            return 0.0
        try:
            mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return mw / 1000.0   # milliwatts → watts
        except Exception:
            return 0.0

    def _read_gpu_util(self) -> float:
        if self._handle is None:
            return 0.0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(util.gpu)
        except Exception:
            return 0.0

    def _read_cpu_util(self) -> float:
        if not _PSUTIL_AVAILABLE:
            return 0.0
        try:
            # interval=None returns usage since last call (non-blocking)
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0

    @property
    def joules(self) -> float:
        """Total energy consumed (Joules) via trapezoidal integration."""
        if len(self._samples) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self._samples)):
            dt = self._samples[i][0] - self._samples[i - 1][0]
            avg_w = (self._samples[i][1] + self._samples[i - 1][1]) / 2.0
            total += avg_w * dt
        return total

    @property
    def avg_power_watts(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s[1] for s in self._samples) / len(self._samples)

    @property
    def avg_gpu_util(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s[2] for s in self._samples) / len(self._samples)

    @property
    def avg_cpu_util(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s[3] for s in self._samples) / len(self._samples)

    @property
    def peak_power_watts(self) -> float:
        if not self._samples:
            return 0.0
        return max(s[1] for s in self._samples)

    def summary(self) -> dict:
        return {
            "total_joules": round(self.joules, 3),
            "avg_power_watts": round(self.avg_power_watts, 2),
            "avg_gpu_util_percent": round(self.avg_gpu_util, 1),
            "avg_cpu_util_percent": round(self.avg_cpu_util, 1),
            "peak_power_watts": round(self.peak_power_watts, 2),
            "num_power_samples": len(self._samples),
            "pynvml_available": _PYNVML_AVAILABLE and self._handle is not None,
            "psutil_available": _PSUTIL_AVAILABLE,
        }
