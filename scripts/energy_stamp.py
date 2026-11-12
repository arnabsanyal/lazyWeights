"""stdin lines (tegrastats) -> '<epoch seconds> <line>' on stdout, flushed per line (Python 3.6+, runs on the Jetsons).
Used by scripts/device/energy_run.sh; when this process exits, tegrastats dies on its next write (SIGPIPE)."""
import sys
import time

for line in sys.stdin:
    sys.stdout.write("%.3f %s" % (time.time(), line))
    sys.stdout.flush()
