"""Refresh existing status views when the fixed MRI preparation aggregate changes."""
import subprocess
import sys
import time
from registration_qc_pilot import PRIVATE, ROOT


def main():
    run = PRIVATE / 'registration-qc-mri-validation-20261103-v1'
    aggregate = run / 'aggregate.json'
    last = None
    while True:
        current = aggregate.stat().st_mtime_ns if aggregate.exists() else None
        if current != last:
            # The producer writes small JSON files; retry rather than publish partial reads.
            status = subprocess.run([sys.executable,str(ROOT/'src/reg_status.py')],
                cwd=ROOT,capture_output=True,text=True)
            if status.returncode == 0:
                board = subprocess.run([sys.executable,str(ROOT/'src/metrics_board.py')],
                    cwd=ROOT,capture_output=True,text=True)
                if board.returncode == 0:
                    last = current
        if (run/'preparation_complete.json').exists():
            break
        if not (run/'runner.lock').exists():
            break
        time.sleep(10)


if __name__ == '__main__':
    main()
