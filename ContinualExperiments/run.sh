#Hopper
CUDA_VISIBLE_DEVICES=0 parallel nohup python ./experiments/train_ddpg.py --env Hopper-v2 --data_dir ./data/Hopper --num_Q 1 --seed {1} ::: $(seq 1 5) > ddpg.log &
CUDA_VISIBLE_DEVICES=1 parallel nohup python ./experiments/train_averageQ.py --env Hopper-v2 --data_dir ./data/Hopper --num_Q 8 --seed {1} ::: $(seq 1 5) > average.log &
CUDA_VISIBLE_DEVICES=2 parallel nohup python ./experiments/train_maxminQ.py --env Hopper-v2 --data_dir ./data/Hopper --num_Q 8 --seed {1} ::: $(seq 1 5) > maxmin.log &
CUDA_VISIBLE_DEVICES=0 parallel nohup python ./experiments/train_redq.py --env Hopper-v2 --data_dir ./data/Hopper --num_Q 8  --num_min 3 --seed {1} ::: $(seq 1 5) > redq.log
CUDA_VISIBLE_DEVICES=1 parallel nohup python ./experiments/train_adaeq.py --env Hopper-v2 --data_dir ./data/Hopper --num_Q 8  --num_min 3 --parameterc 0.3 --seed {1} ::: $(seq 1 5) > adaeq.log &
CUDA_VISIBLE_DEVICES=2 parallel nohup python ./experiments/train_order.py --env Hopper-v2 --data_dir ./data/Hopper --num_Q 8  --order_m 2,4,8 --seed {1} ::: $(seq 1 5) > order.log &
CUDA_VISIBLE_DEVICES=0 parallel nohup python ./experiments/train_adaorder.py --env Hopper-v2 --data_dir ./data/Hopper --num_C 4,8,16,32  --seed {1} ::: $(seq 1 5) > adaorder.log &
CUDA_VISIBLE_DEVICES=1 parallel nohup python ./experiments/train_adaaverage.py --env Hopper-v2 --data_dir ./data/Hopper --num_C 8  --seed {1} ::: $(seq 1 5) > adaaverage.log &
CUDA_VISIBLE_DEVICES=2 parallel nohup python ./experiments/train_adamin.py --env Hopper-v2 --data_dir ./data/Hopper --num_C 8  --seed {1} ::: $(seq 1 5) > adamin.log &


#Ant
CUDA_VISIBLE_DEVICES=0 parallel nohup python ./experiments/train_ddpg.py --env Ant-v2 --data_dir ./data/Ant --num_Q 1 --seed {1} ::: $(seq 1 5) > ddpg.log &
CUDA_VISIBLE_DEVICES=1 parallel nohup python ./experiments/train_averageQ.py --env Ant-v2 --data_dir ./data/Ant --num_Q 8 --seed {1} ::: $(seq 1 5) > average.log &
CUDA_VISIBLE_DEVICES=2 parallel nohup python ./experiments/train_maxminQ.py --env Ant-v2 --data_dir ./data/Ant --num_Q 8 --seed {1} ::: $(seq 1 5) > maxmin.log &
CUDA_VISIBLE_DEVICES=0 parallel nohup python ./experiments/train_redq.py --env Ant-v2 --data_dir ./data/Ant --num_Q 8  --num_min 3 --seed {1} ::: $(seq 1 5) > redq.log &
CUDA_VISIBLE_DEVICES=1 parallel nohup python ./experiments/train_adaeq.py --env Ant-v2 --data_dir ./data/Ant --num_Q 8  --num_min 3 --parameterc 0.3 --seed {1} ::: $(seq 1 5) > adaeq.log &
CUDA_VISIBLE_DEVICES=2 parallel nohup python ./experiments/train_order.py --env Ant-v2 --data_dir ./data/Ant --num_Q 8  --order_m 2,4,8 --seed {1} ::: $(seq 1 5) > order.log &
CUDA_VISIBLE_DEVICES=0 parallel nohup python ./experiments/train_adaorder.py --env Ant-v2 --data_dir ./data/Ant --num_C 4,8,16,32  --seed {1} ::: $(seq 1 5) > adaorder.log &
CUDA_VISIBLE_DEVICES=1 parallel nohup python ./experiments/train_adaaverage.py --env Ant-v2 --data_dir ./data/Ant --num_C 8  --seed {1} ::: $(seq 1 5) > adaaverage.log &
CUDA_VISIBLE_DEVICES=2 parallel nohup python ./experiments/train_adamin.py --env Ant-v2 --data_dir ./data/Ant --num_C 8  --seed {1} ::: $(seq 1 5) > adamin.log &



#Walk
CUDA_VISIBLE_DEVICES=0 parallel nohup python ./experiments/train_ddpg.py --env Walker2d-v2 --data_dir ./data/Walker2d --num_Q 1 --seed {1} ::: $(seq 1 5) > ddpg.log &
CUDA_VISIBLE_DEVICES=1 parallel nohup python ./experiments/train_averageQ.py --env Walker2d-v2 --data_dir ./data/Walker2d --num_Q 8 --seed {1} ::: $(seq 1 5) > average.log &
CUDA_VISIBLE_DEVICES=2 parallel nohup python ./experiments/train_maxminQ.py --env Walker2d-v2 --data_dir ./data/Walker2d --num_Q 8 --seed {1} ::: $(seq 1 5) > maxmin.log &
CUDA_VISIBLE_DEVICES=0 parallel nohup python ./experiments/train_redq.py --env Walker2d-v2 --data_dir ./data/Walker2d --num_Q 8  --num_min 3 --seed {1} ::: $(seq 1 5) > redq.log &
CUDA_VISIBLE_DEVICES=1 parallel nohup python ./experiments/train_adaeq.py --env Walker2d-v2 --data_dir ./data/Walker2d --num_Q 8  --num_min 3 --parameterc 0.3 --seed {1} ::: $(seq 1 5) > adaeq.log &
CUDA_VISIBLE_DEVICES=2 parallel nohup python ./experiments/train_order.py --env Walker2d-v2 --data_dir ./data/Walker2d --num_Q 8  --order_m 2,4,8 --seed {1} ::: $(seq 1 5) > order.log &
CUDA_VISIBLE_DEVICES=0 parallel nohup python ./experiments/train_adaorder.py --env Walker2d-v2 --data_dir ./data/Walker2d --num_C 4,8,16,32  --seed {1} ::: $(seq 1 5) > adaorder.log &
CUDA_VISIBLE_DEVICES=1 parallel nohup python ./experiments/train_adaaverage.py --env Walker2d-v2 --data_dir ./data/Walker2d --num_C 8  --seed {1} ::: $(seq 1 5) > adaaverage.log &
CUDA_VISIBLE_DEVICES=2 parallel nohup python ./experiments/train_adamin.py --env Walker2d-v2 --data_dir ./data/Walker2d --num_C 8  --seed {1} ::: $(seq 1 5) > adamin.log &