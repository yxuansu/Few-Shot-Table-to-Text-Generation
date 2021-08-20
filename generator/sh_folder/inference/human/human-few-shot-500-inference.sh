CUDA_VISIBLE_DEVICES=2 python ../../../inference.py\
    --dataset_prefix ../../../../data/human/\
    --context_prefix ../../../../checkpoints/prototype_selector/human/few-shot-500/rerank-context/\
    --model_name facebook/bart-base\
    --ckpt_path ../../../../checkpoints/generator/human/few-shot-500/rerank-context/epoch_54_validation_bleu_50.93/\
    --batch_size 64
