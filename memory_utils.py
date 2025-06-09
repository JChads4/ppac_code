import os
import psutil
import signal
from contextlib import contextmanager

class MemoryLimitExceeded(Exception):
    """Exception raised when memory usage exceeds the specified limit."""
    pass

def get_memory_usage():
    """Return the current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def check_critical_memory():
    """Check if system memory is critically low."""
    mem = psutil.virtual_memory()
    return mem.available / mem.total < 0.15

@contextmanager
def memory_limit(max_mb, safety_margin_mb=500):
    """Context manager enforcing a memory limit with safety margin."""
    adjusted_limit = max(100, max_mb - safety_margin_mb)

    def memory_check():
        used_mb = get_memory_usage()
        if used_mb > adjusted_limit:
            raise MemoryLimitExceeded(
                f"Memory limit approaching: {used_mb:.1f}MB used, limit is {adjusted_limit}MB"
            )
        if check_critical_memory():
            raise MemoryLimitExceeded("System memory critically low")

    memory_check()

    original_handler = None

    def sigalrm_handler(signum, frame):
        memory_check()
        signal.alarm(1)

    try:
        original_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
        signal.alarm(1)
        yield
    finally:
        signal.alarm(0)
        if original_handler:
            signal.signal(signal.SIGALRM, original_handler)
