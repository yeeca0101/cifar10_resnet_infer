#!/bin/bash
# cifar100 110: 512 , others 1024
# cifar10 128
# resnet32 resnet44 resnet56 resnet110 resnet20 lightvit
for model in resnet20 resnet32 resnet44 resnet56 resnet110 
do
    echo "python3 -u trainer.py  --epochs=200 -n 10 --arch=$model  --save-dir=save_$model -b=128 |& tee -a ./t_log/log_cifar100_$model"
    python3 -u trainer.py --epochs=200 -n 100 --arch=$model  --save_folder "checkpoint/dev_cifar100/save_$model" -b=128 --repeat 5
done

# for model in lightvit
# do
#     echo "python3 -u trainer.py  --epochs=200 -n 100 --arch=$model  --save-dir=save_$model -b=128 |& tee -a ./t_log/log_cifar10_$model"
#     python3 -u trainer.py --epochs=200 -n 100 --arch=$model  --save-dir=save_$model -b=128 
# done