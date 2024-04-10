#!/usr/bin/env bash

methods=("HierHFL" "JSHFL")
attacks=("label" "sign" "gaussian" "krum_attack" "full_trim" )
for attack in "${attacks[@]}"; do
    for method in "${methods[@]}"; do
        echo "Execute method = ${method}，attack=${attack}"
        for i in 5; do  # control malicious clients number
            echo " m=${i} is starting"
            echo "Execute now-"
            sh run_fedavg_standalone_pytorch.sh 0 100 10 8 mnist "mnist" cnn hetero 1 1 0.001 sgd 0 500 ${i} ${method} ${attack}
            # if you want to run on cifar10, please change relevant parameters
            #sh run_fedavg_standalone_pytorch.sh 0 100 10 10 cifar10 "cifar10" resnet20 hetero 1 1 0.001 sgd 0 500 ${i} ${method} ${attack}
        done
    done
done
