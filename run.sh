export PYTHONPATH=$PYTHONPATH:./
prefix_cmd="srun -p aigc-video --nodes 1 --ntasks-per-node 8 --cpus-per-task 8 --gres=gpu:8 --quotatype=auto"
test_cmd="srun -p aigc-video --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --gres=gpu:1 --quotatype=auto"
submit_cmd="srun -p aigc-video --nodes 1 --ntasks-per-node 8 --cpus-per-task 8 --gres=gpu:8 --async"

function train_syncdreamer(){
    export CUDA_LAUNCH_BLOCKING=1
    $test_cmd python train_random_sync_dreamer.py -b configs/random-syncdreamer-train.yaml \
                           --finetune_from "ckpt/random-syncdreamer-train/zero123-xl.ckpt" \
                           -l log  \
                           -c ckpt \
                           --gpus 0
}

function test_sync_dataset(){
    export PYTHONPATH=$PYTHONPATH:.
    $test_cmd python ldm/data/sync_dreamer.py
}
$1