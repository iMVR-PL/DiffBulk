export CUDA_VISIBLE_DEVICES=1

start_time=$(date +%s)

echo "Starting experiments..."

echo "Running experiment:"
start_exp_time=$(date +%s)

python train.py --config ./config.yaml

end_exp_time=$(date +%s)
echo "Experiment finished in $(($end_exp_time - $start_exp_time)) seconds."

end_time=$(date +%s)
echo "All experiments finished in $(($end_time - $start_time)) seconds!"
