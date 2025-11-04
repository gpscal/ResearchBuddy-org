import threading
import time
import psutil
import gc


class ResourceMonitor(threading.Thread):
    """Monitor system resources and apply mitigations when thresholds are exceeded (CPU-only)."""

    def __init__(self, ram_threshold: float = 0.85, check_interval: int = 5):
        super().__init__(daemon=True)
        self.ram_threshold = ram_threshold
        self.check_interval = check_interval
        self._running = True

    def run(self) -> None:
        while self._running:
            try:
                # RAM usage monitoring (CPU-only)
                ram_usage = psutil.virtual_memory().percent / 100.0
                if ram_usage > self.ram_threshold:
                    self._clear_caches()
            finally:
                time.sleep(self.check_interval)

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _clear_caches() -> None:
        """Clear Python caches to free up memory (CPU-only)."""
        gc.collect()


