@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =============== User-configurable defaults ===============
set "PYTHON=python"
set "NUM_EPOCH=200"
set "BATCH_SIZE=128"
set "LR=0.0001"
set "DROPOUT=0.3"
set "TRAIN_DATA=.\data\archive\avg_train.csv"
set "VAL_DATA=.\data\archive\avg_val.csv"
set "BASE_SAVE_DIR=.\train_log\grid_search"

REM =============== Parameter grids (edit as needed) ===============
set "INTENTION_LIST=0.04 0.06 0.08 0.1"
set "SENTIMENT_LIST=0.04 0.06 0.08 0.1"
set "OFFENSE_LIST=0.04 0.06 0.08 0.1"

if not exist "%BASE_SAVE_DIR%" mkdir "%BASE_SAVE_DIR%"

for %%I in (%INTENTION_LIST%) do (
  for %%S in (%SENTIMENT_LIST%) do (
    for %%O in (%OFFENSE_LIST%) do (
      set "RUN_SAVE_DIR=%BASE_SAVE_DIR%\ic-%%I_sc-%%S_oc-%%O"
      echo ==========================================================
      echo Starting run: ic=%%I sc=%%S oc=%%O
      echo Save dir: !RUN_SAVE_DIR!
      if not exist "!RUN_SAVE_DIR!" mkdir "!RUN_SAVE_DIR!"
      D:/AnaConda/envs/pytorch/python.exe e:/Documents/metaphor_detector/run_contrast_train.py ^
        --intention_coefficient %%I ^
        --sentiment_coefficient %%S ^
        --offensiveness_coefficient %%O ^
        --dropout %DROPOUT% ^
        --batch_size %BATCH_SIZE% ^
        --num_epoch %NUM_EPOCH% ^
        --lr %LR% ^
        --train_data "%TRAIN_DATA%" ^
        --val_data "%VAL_DATA%" ^
        --save_dir "!RUN_SAVE_DIR!"
      echo Finished run: ic=%%I sc=%%S oc=%%O
      echo.
    )
  )
)

echo All runs completed.
endlocal


