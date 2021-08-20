CUDA_VISIBLE_DEVICES=1 python ../../../learn.py\
    --train_data_num 500\
    --dataset_prefix ../../../../data/human/\
    --context_prefix ../../../../checkpoints/prototype_selector/human/few-shot-500/rerank-context/\
    --model_name facebook/bart-base\
    --optimizer_name adam\
    --learning_rate 3e-5\
    --dropout 0.2\
    --total_steps 6000\
    --warmup_steps 100\
    --print_every 10\
    --eval_every 30\
    --batch_size_per_gpu 8\
    --number_of_gpu 1\
    --gradient_accumulation_steps 1\
    --ckpt_save_path ../../../../checkpoints/generator/human/few-shot-500/rerank-context/