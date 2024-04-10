#!/usr/bin/env bash

GPU=$1  # 1

CLIENT_NUM=$2  # 宗的1000人  2

WORKER_NUM=$3 # 1000人挑出5人  3

BATCH_SIZE=$4  # 每批数据量的大小  4

DATASET=$5  # 5

DATA_PATH=$6

MODEL=$7

DISTRIBUTION=$8

ROUND=$9

EPOCH=${10}

LR=${11}

OPT=${12}

CI=${13}

B=${14}

M=${15}

METHOD_NAME=${16}

ATTACK_TYPE=${17}

#T=${15}

python ./main_fedavg.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--client_optimizer $OPT \
--lr $LR \
--ci $CI \
--malicious_m $M \
--method_name $METHOD_NAME \
--attack_type $ATTACK_TYPE \
#--task_array $T