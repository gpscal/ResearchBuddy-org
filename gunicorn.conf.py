# Simplified Gunicorn configuration for researchsite
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:80"
backlog = 2048

# Worker processes - optimized for low-resource systems (1GB RAM)
# Use 1 worker with minimal threads to minimize memory usage
workers = 1
worker_class = "gthread"
threads = 1  # Reduced to 1 thread for 1GB RAM system
worker_connections = 200  # Further reduced for memory-constrained system
timeout = 120  # Reduced timeout for faster failover
keepalive = 2

# Restart workers more frequently to prevent memory leaks
max_requests = 100  # Very frequent restarts for 1GB RAM system
max_requests_jitter = 10

# Logging
accesslog = "/var/log/researchsite/access.log"
errorlog = "/var/log/researchsite/error.log"
loglevel = "info"

# Process naming
proc_name = "researchsite"

# Server mechanics
daemon = False
pidfile = "/var/run/researchsite/researchsite.pid"

# Environment variables
raw_env = [
    "PORT=80",
    "PYTHONPATH=/home/ubuntu/researchsite",
]

# IMPORTANT: Disable preload to avoid CUDA forking issues
# CUDA cannot be re-initialized in forked subprocesses
preload_app = False

# Graceful timeout
graceful_timeout = 60

# Memory management - reduced for low-resource systems
# For 1GB RAM system: allow 256MB per worker (OS + nginx + overhead need ~300-400MB)
max_worker_memory = 256  # MB - optimized for 1GB total RAM

# Worker lifecycle hooks
def post_fork(server, worker):
    """Called after a worker has been forked."""
    # No CUDA initialization needed for Anthropic-only setup
    worker.log.info(f"Worker {worker.pid}: Initialized")