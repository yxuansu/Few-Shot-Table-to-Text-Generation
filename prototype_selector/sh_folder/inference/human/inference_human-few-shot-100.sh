CUDA_VISIBLE_DEVICES=4 python3 ../../../inference.py\
    --dataset_name human\
    --model_name bert-base-uncased\
    --dataset_prefix ../../../../data/human\
    --ckpt_path ../../../../checkpoints/prototype_selector/human/few-shot-100/epoch_18_batch_480_test_bleu_9.180\
    --save_prefix ../../../../checkpoints/prototype_selector/human/few-shot-100/rerank-context/\
    --batch_size 16
