#!/bin/bash
# bash scripts/channelformer.sh

export CUDA_VISIBLE_DEVICES=0
id="001"
train_id=$(date "+%Y-%m-%d-")$id
model_name=ChannelFormer
e_layers=3
batch_size=16

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/ArticularyWordRecognition/ \
    --problem ArticularyWordRecognition \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 64 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 256 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/AtrialFibrillation/ \
    --problem AtrialFibrillation \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 64 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/BasicMotions/ \
    --problem BasicMotions \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 64 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/CharacterTrajectories/ \
    --problem CharacterTrajectories \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 64 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 1024 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/Cricket/ \
    --problem Cricket \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 256 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/DuckDuckGeese/ \
    --problem DuckDuckGeese \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 32 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 32 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/EigenWorms/ \
    --problem EigenWorms \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 32 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 1536 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/Epilepsy/ \
    --problem Epilepsy \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 128 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 64 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/ERing/ \
    --problem ERing \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 32 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 512 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/EthanolConcentration/ \
    --problem EthanolConcentration \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 64 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/FaceDetection/ \
    --problem FaceDetection \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 128 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 32 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/FingerMovements/ \
    --problem FingerMovements \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 256 \
    --d_ff 128 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 128 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/HandMovementDirection/ \
    --problem HandMovementDirection \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 128 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/Handwriting/ \
    --problem Handwriting \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 256 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 512 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/Heartbeat/ \
    --problem Heartbeat \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 32 \
    --d_ff 128 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 2048 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/JapaneseVowels/ \
    --problem JapaneseVowels \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 512 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 32 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/Libras/ \
    --problem Libras \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 512 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 1024 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/LSST/ \
    --problem LSST \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 128 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 128 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/MotorImagery/ \
    --problem MotorImagery \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 256 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/NATOPS/ \
    --problem NATOPS \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 32 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 1024 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/PEMS-SF/ \
    --problem PEMS-SF \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 64 \
    --d_ff 256 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 512 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/PenDigits/ \
    --problem PenDigits \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 256 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/PhonemeSpectra/ \
    --problem PhonemeSpectra \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 128 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 512 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/RacketSports/ \
    --problem RacketSports \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 512 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/SelfRegulationSCP1/ \
    --problem SelfRegulationSCP1 \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 32 \
    --d_ff 128 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 64 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id\
    --is_training 1 \
    --data_path ./dataset/SelfRegulationSCP2/ \
    --problem SelfRegulationSCP2 \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 32 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 32 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/SpokenArabicDigits/ \
    --problem SpokenArabicDigits \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 32 \
    --d_ff 128 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 64 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/StandWalkJump/ \
    --problem StandWalkJump \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 128 \
    --d_ff 64 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 32 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/UWaveGestureLibrary/ \
    --problem UWaveGestureLibrary \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 128 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 4 \
    --c_d_model 32 \
    --val_ratio 0.2 \
    --dropout 0.1 \

python -u run.py \
    --train_id $train_id \
    --is_training 1 \
    --data_path ./dataset/InsectWingbeat/ \
    --problem InsectWingbeat \
    --model $model_name \
    --dataset UEA \
    --e_layers $e_layers \
    --batch_size $batch_size\
    --d_model 16 \
    --d_ff 128 \
    --learning_rate 0.001 \
    --train_epochs 200 \
    --patience 20  \
    --c_n_heads 8 \
    --c_d_model 1536 \
    --val_ratio 0.2 \
    --dropout 0.1 \