CUDA_VISIBLE_DEVICES=5 python3 ../../../inference.py\
    --dataset_name human\
    --model_name bert-base-uncased\
    --dataset_prefix ../../../../data/human\
    --ckpt_path ../../../../checkpoints/prototype_selector/human/few-shot-200/epoch_17_batch_880_test_bleu_9.471\
    --save_prefix ../../../../checkpoints/prototype_selector/human/few-shot-200/rerank-context/\
    --batch_size 16
