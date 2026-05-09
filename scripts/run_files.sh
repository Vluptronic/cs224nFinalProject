#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting pipeline..."

# 1. Run baseline.py
echo "Running baseline.py..."
python baseline.py --dtype float16

# 2. Run lora.py
echo "Running lora.py..."
python lora.py --dtype float16

# 3. Run baseline_rag.py with rebuild_index
echo "Running baseline_rag.py..."
python baseline_rag.py --dtype float16 --rebuild_index

# 4. Run lora_rag.py
echo "Running lora_rag.py..."
python lora_rag.py --dtype float16

echo "All scripts completed successfully."
