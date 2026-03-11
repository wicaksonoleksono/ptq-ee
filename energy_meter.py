"""
energy_meter.py
---------------
Updated to capture both Compute Util (%) and Memory Usage (MB).
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
    def __init__(self, device_idx: int = 0, sample_interval: float = 0.1):
        self.device_idx = device_idx
        self.sample_interval = sample_interval
        # samples: (timestamp, watts, gpu_compute_util, cpu_util, gpu_mem_used_mb)
        self._samples: list = []   
        self._running = False
        self._thread = None
        self._handle = None
        self.gpu_mem_total = 0.0

        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.gpu_mem_total = mem_info.total / 1024**2 # MB
                
                gpu_name = pynvml.nvmlDeviceGetName(self._handle)
                if isinstance(gpu_name, bytes): gpu_name = gpu_name.decode()
                print(f"[EnergyMeter] Monitoring {gpu_name} ({self.gpu_mem_total:.0f}MB total)")
            except Exception as e:
                print(f"[EnergyMeter] WARNING: pynvml failed: {e}")
                self._handle = None

    def start(self):
        self._samples = []
        self._running = True
        if _PSUTIL_AVAILABLE: psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None: self._thread.join(timeout=1.0)

    def _sample_loop(self):
        while self._running:
            ts = time.perf_counter()
            watts = self._read_power_watts()
            gpu_util = self._read_gpu_util()
            cpu_util = self._read_cpu_util()
            gpu_mem = self._read_gpu_mem_mb()
            
            self._samples.append((ts, watts, gpu_util, cpu_util, gpu_mem))
            time.sleep(self.sample_interval)

    def _read_power_watts(self):
        if not self._handle: return 0.0
        try: return pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except: return 0.0

    def _read_gpu_util(self):
        if not self._handle: return 0.0
        try: return float(pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
        except: return 0.0

    def _read_gpu_mem_mb(self):
        if not self._handle: return 0.0
        try: return pynvml.nvmlDeviceGetMemoryInfo(self._handle).used / 1024**2
        except: return 0.0

    def _read_cpu_util(self):
        if not _PSUTIL_AVAILABLE: return 0.0
        return psutil.cpu_percent(interval=None)

    @property
    def joules(self):
        if len(self._samples) < 2: return 0.0
        total = 0.0
        for i in range(1, len(self._samples)):
            dt = self._samples[i][0] - self._samples[i-1][0]
            avg_w = (self._samples[i][1] + self._samples[i-1][1]) / 2.0
            total += avg_w * dt
        return total

    @property
    def avg_gpu_util(self): return np.mean([s[2] for s in self._samples]) if self._samples else 0.0
    
    @property
    def avg_gpu_mem_mb(self): return np.mean([s[4] for s in self._samples]) if self._samples else 0.0

    @property
    def peak_gpu_mem_mb(self): return max([s[4] for s in self._samples]) if self._samples else 0.0
    
    @property
    def avg_cpu_util(self): return np.mean([s[3] for s in self._samples]) if self._samples else 0.0

    @property
    def avg_power_watts(self): return np.mean([s[1] for s in self._samples]) if self._samples else 0.0

import numpy as np
