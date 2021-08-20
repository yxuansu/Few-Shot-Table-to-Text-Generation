CUDA_VISIBLE_DEVICES=2 python ../../../inference.py\
    --dataset_prefix ../../../../data/human/\
    --context_prefix ../../../../checkpoints/prototype_selector/human/few-shot-200/rerank-context/\
    --model_name facebook/bart-base\
    --ckpt_path ../../../../checkpoints/generator/human/few-shot-200/rerank-context/epoch_131_validation_bleu_47.91/\
    --batch_size 64
