#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watchdog.py
===========
Restart-on-failure wrapper around ``run.py``.

If training crashes (OOM, kernel kill, transient hardware error), this
script waits ``RETRY_DELAY`` seconds and relaunches the trainer. The
trainer itself resumes from its latest checkpoint, so progress is not
lost. Stops after ``MAX_RETRIES`` consecutive failures.

Usage:
    python watchdog.py                       # default: 10 retries, 30 s delay
    MAX_RETRIES=20 RETRY_DELAY=60 python watchdog.py
"""

import os, sys, subprocess, time, signal

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, 'run.py')
PYTHON       = sys.executable

MAX_RETRIES  = int(os.environ.get('MAX_RETRIES', 10))
RETRY_DELAY  = int(os.environ.get('RETRY_DELAY', 30))

_child_proc = None
def _sigint_handler(sig, frame):
    print("\n\n⛔  watchdog received interrupt signal, stopping training process...")
    if _child_proc and _child_proc.poll() is None:
        _child_proc.terminate()
        try:
            _child_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _child_proc.kill()
    print("Exited. Next run will resume from checkpoint.")
    sys.exit(0)

signal.signal(signal.SIGINT,  _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)

def main():
    global _child_proc

    env = os.environ.copy()
    attempt = 0

    print("=" * 60)
    print("🐕  Training watchdog started")
    print(f"   Training script: {TRAIN_SCRIPT}")
    print(f"   Max retries: {MAX_RETRIES} times")
    print(f"   Retry delay: {RETRY_DELAY} ")
    print("=" * 60)

    while attempt <= MAX_RETRIES:
        start_time = time.time()
        print(f"\n🚀  [Attempt {attempt + 1}/{MAX_RETRIES + 1}]  "
              f"Starting training @ {time.strftime('%H:%M:%S')}")

        _child_proc = subprocess.Popen(
            [PYTHON, TRAIN_SCRIPT],
            env=env,
        )

        exit_code = _child_proc.wait()
        elapsed   = time.time() - start_time

        if exit_code == 0:
            print(f"\n✅  Training completed successfully! Time elapsed: {elapsed/3600:.1f} hours")
            break
        else:
            attempt += 1
            if attempt > MAX_RETRIES:
                print(f"\n❌  Max retries reached ({MAX_RETRIES})")
                break

            print(f"\n⚠️  Training exited abnormally (exit code={exit_code}，"
                  f" {elapsed/60:.1f} minutes)")
            print(f"   {RETRY_DELAY} seconds before auto-restart (resume)...")
            time.sleep(RETRY_DELAY)

if __name__ == '__main__':
    main()
