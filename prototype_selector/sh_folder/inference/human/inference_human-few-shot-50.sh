CUDA_VISIBLE_DEVICES=2 python3 ../../../inference.py\
    --dataset_name human\
    --model_name bert-base-uncased\
    --dataset_prefix ../../../../data/human\
    --ckpt_path ../../../../checkpoints/prototype_selector/human/few-shot-50/epoch_56_batch_740_test_bleu_9.059\
    --save_prefix ../../../../checkpoints/prototype_selector/human/few-shot-50/rerank-context/\
    --batch_size 16
