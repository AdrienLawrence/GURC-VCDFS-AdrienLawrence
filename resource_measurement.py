"""Process-memory sampling shared by scientific experiment runners."""

import ctypes
import platform
from pathlib import Path
from threading import Event, Thread


def current_rss_bytes():
    """Return current resident memory, or ``None`` when unsupported."""
    if platform.system() == "Windows":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set", ctypes.c_size_t),
                ("working_set", ctypes.c_size_t),
                ("quota_peak_paged", ctypes.c_size_t),
                ("quota_paged", ctypes.c_size_t),
                ("quota_peak_nonpaged", ctypes.c_size_t),
                ("quota_nonpaged", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t),
                ("peak_pagefile", ctypes.c_size_t),
                ("private_usage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(Counters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.working_set)

    statm = Path("/proc/self/statm")
    if statm.exists():
        import os

        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    return None


class RssSampler:
    """Sample whole-process resident memory in a background thread."""

    def __init__(self, interval=0.002):
        self.interval = interval
        self.start = current_rss_bytes()
        self.peak = self.start
        self._stop = Event()
        self._thread = Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.wait(self.interval):
            value = current_rss_bytes()
            if value is not None and (self.peak is None or value > self.peak):
                self.peak = value

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        value = current_rss_bytes()
        if value is not None and (self.peak is None or value > self.peak):
            self.peak = value
