CUDA_VISIBLE_DEVICES=7 python3 ../../../inference.py\
    --dataset_name human\
    --model_name bert-base-uncased\
    --dataset_prefix ../../../../data/human\
    --ckpt_path ../../../../checkpoints/prototype_selector/human/few-shot-500/epoch_23_batch_2900_test_bleu_9.430\
    --save_prefix ../../../../checkpoints/prototype_selector/human/few-shot-500/rerank-context/\
    --batch_size 16
