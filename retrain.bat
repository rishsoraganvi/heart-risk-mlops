@echo off
REM retrain.bat — Full retraining pipeline
REM Run this whenever drift is detected or on a schedule
REM Usage: retrain.bat

echo ============================================
echo  Heart Risk MLOps — Retraining Pipeline
echo ============================================

echo [1/4] Checking for data drift...
python src/data_drift.py
if %errorlevel% neq 0 (
    echo Drift check failed. Aborting.
    exit /b 1
)

echo [2/4] Running training...
python src/train.py
if %errorlevel% neq 0 (
    echo Training failed. Aborting.
    exit /b 1
)

echo [3/4] Comparing models...
python src/compare_models.py
if %errorlevel% neq 0 (
    echo Model comparison failed. Aborting.
    exit /b 1
)

echo [4/4] Running batch inference with latest model...
python src/batch_infer.py
if %errorlevel% neq 0 (
    echo Batch inference failed. Aborting.
    exit /b 1
)

echo ============================================
echo  Retraining pipeline complete successfully
echo ============================================