#!/bin/sh
cd "$(dirname "$0")/.."
python3 scripts/run_all.py all > experiments/run_all.log 2>&1 &
echo $! > experiments/run_all.pid
