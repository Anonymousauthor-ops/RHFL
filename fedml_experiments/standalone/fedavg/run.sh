#!/usr/bin/env bash

for i in 10
do
  echo "m=${i} for this programming"
  echo "start programming"

    sh run_fedavg_standalone_pytorch.sh 0 100 10 16 mnist FedML-master/data/MNIST/mnist cnn hetero 1 1 0.001 sgd 0 500 ${i} JSHFL gaussian
  echo "end programming"
done

