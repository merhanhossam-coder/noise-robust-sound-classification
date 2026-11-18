#!/bin/sh
cd "$(dirname "$0")/.."
python3 scripts/export_demo_checkpoint.py > experiments/demo_export.log 2>&1 &
