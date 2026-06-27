"""Utilities for models / mpnn / gpu_monitor.py in the DCG benchmark codebase."""

# gpu_monitor.py
from __future__ import annotations
import time, threading

try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

def bytes_to_gb(b: int) -> float:
    """
    Implement the bytes to gb step for models / mpnn / gpu_monitor.py.

    Args:
        b: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    return round(float(b) / (1024**3), 3)

class GPUMonitor:
    """
    Lightweight NVML sampler. Records GPU utilization (%), memory used (bytes),
    power draw (W), and clocks every ~0.25s (configurable). Safe to use on
    single-GPU runs; does nothing if NVML is unavailable.

    Role:
        GPUMonitor groups state and methods for this repository component.
    """
    def __init__(self, device_index=0, interval_sec=0.25):
        """
        Initialize the GPUMonitor instance and store constructor configuration.

        Args:
            device_index: Caller-supplied value used by this routine.
            interval_sec: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.device_index = device_index
        self.interval = interval_sec
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.nvml_ok = False
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                name = pynvml.nvmlDeviceGetName(self._handle)
                self.name = name.decode() if isinstance(name, bytes) else name
                self.total_mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle).total
                drv = pynvml.nvmlSystemGetDriverVersion()
                self.driver = drv.decode() if isinstance(drv, bytes) else drv
                self.nvml_ok = True
            except Exception:
                self.nvml_ok = False

    def start(self):
        """
        Implement the start step for models / mpnn / gpu_monitor.py.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        if not self.nvml_ok:
            print("[INFO] GPUMonitor: NVML unavailable – utilization/power sampling disabled "
                  "(install 'nvidia-ml-py3' if you want these).")
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Implement the stop step for models / mpnn / gpu_monitor.py.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        if not self.nvml_ok:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _loop(self):
        """
        Implement the loop step for models / mpnn / gpu_monitor.py.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # W
                except Exception:
                    power = None
                try:
                    sm = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
                    memclk = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    sm = memclk = None
                self.samples.append((t, util, mem.used, mem.total, power, sm, memclk))
            except Exception:
                pass
            time.sleep(self.interval)

    def summary(self) -> dict:
        """
        Implement the summary step for models / mpnn / gpu_monitor.py.

        Returns:
            Computed value used by the caller.
        """
        if not self.nvml_ok or not self.samples:
            return {'nvml_available': False}
        t0, t1 = self.samples[0][0], self.samples[-1][0]
        duration = max(0.0, t1 - t0)
        utils   = [s[1] for s in self.samples if s[1] is not None]
        memused = [s[2] for s in self.samples if s[2] is not None]
        power   = [s[4] for s in self.samples if s[4] is not None]
        smclk   = [s[5] for s in self.samples if s[5] is not None]
        memclk  = [s[6] for s in self.samples if s[6] is not None]

        # Trapezoidal integration: W*s = J
        energy_j = 0.0
        for i in range(1, len(self.samples)):
            p0, p1 = self.samples[i-1][4], self.samples[i][4]
            dt = self.samples[i][0] - self.samples[i-1][0]
            if p0 is not None and p1 is not None:
                energy_j += 0.5 * (p0 + p1) * dt

        return {
            'nvml_available': True,
            'device_index': self.device_index,
            'gpu_name': self.name,
            'driver_version': self.driver,
            'total_mem_bytes': int(self.total_mem),
            'duration_sec': float(duration),
            'avg_util_percent': float(sum(utils) / len(utils)) if utils else None,
            'peak_util_percent': float(max(utils)) if utils else None,
            'avg_power_w': float(sum(power) / len(power)) if power else None,
            'peak_power_w': float(max(power)) if power else None,
            'energy_wh': float(energy_j / 3600.0) if energy_j else None,
            'peak_mem_used_bytes_nvml': int(max(memused)) if memused else None,
            'avg_sm_clock_mhz': float(sum(smclk)/len(smclk)) if smclk else None,
            'avg_mem_clock_mhz': float(sum(memclk)/len(memclk)) if memclk else None,
        }
