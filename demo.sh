#!/bin/bash

set -e  # Exit immediately if a command fails
echo "Running FFM training script..."
python3 Code/ForwardForecastModel.py

echo "Running FFM evaluation script..."
python3 Code/EvaluateFFM.py

echo "Running BRM training script..."
python3 Code/BackwardReconstructionModel.py

echo "Running BRM evaluation script..."
python3 Code/EvaluateBRM.py

