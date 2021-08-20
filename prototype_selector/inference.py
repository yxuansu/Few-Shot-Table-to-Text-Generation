import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import argparse
import logging
from transformers import AdamW, get_linear_schedule_with_warmup
from retriever import trainer

def extract_test_pred_text(test_batch_score, test_batch_candidate_text_list):
    pred_text_list = []
    test_batch_score_list = test_batch_score.detach().cpu().numpy()
    bsz = len(test_batch_score_list)
    for idx in range(bsz):
        one_score_list = test_batch_score_list[idx]
        one_select_idx_list = np.argsort(one_score_list)[::-1]
        one_text_list = test_batch_candidate_text_list[idx]
        one_pred_text_list = [one_text_list[s_idx] for s_idx in one_select_idx_list]
        pred_text_list.append('\t'.join(one_pred_text_list).strip('\t'))
    return pred_text_list

def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset_name', type=str, help="human or song or book")
    parser.add_argument('--model_name', type=str, help="e.g. bert-base...")
    parser.add_argument('--dataset_prefix', type=str, help="the path that stores the data")
    parser.add_argument('--train_candi_pool_size', type=int, default=50, 
        help="Randomly selecting negative examples from the top-k retrieved candidates provided by the IR system.")
    parser.add_argument('--train_negative_num', type=int, default=5, 
        help="number of randomly selected negatives from the retrieved candidates from the IR system.")
    parser.add_argument('--test_candi_span', type=int, default=20, 
        help="reranking the best response from top-n candidates from the IR system.")
    parser.add_argument('--max_table_len', type=int, default=250)
    parser.add_argument('--max_tgt_len', type=int, default=80)
    # training configuration
    parser.add_argument('--ckpt_path', type=str, help="path of pre-trained prototype selector.")
    parser.add_argument('--save_prefix', type=str, help="path prefix to save the reranked context.")
    parser.add_argument('--batch_size', type=int)
    # learning configuration
    parser.add_argument('--gpu_id', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    device = args.gpu_id

    import os
    if os.path.exists(args.save_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(args.save_prefix, exist_ok=True)

    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print ('Cuda is available.')
    else:
        print ('Cuda is not available.')

    SEP, EOS = '<sep>', '<eos>'
    print ('Loading tokenizer...')
    model_name = args.model_name
    if model_name.startswith('bert'):
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    elif model_name.startswith('roberta'):
        from transformers import RobertaTokenizerFast
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    else:
        raise Exception('Wrong Tokenization Mode!')
    special_token_list = [SEP, EOS]
    print ('original vocabulary Size %d' % len(tokenizer))
    tokenizer.add_tokens(special_token_list)
    print ('vocabulary size after extension is %d' % len(tokenizer))

    print ('Initializing Model...')
    from retriever import Model
    model = Model(args.model_name, tokenizer)
    if cuda_available:
        model = model.cuda(device)
    print ('Model Loaded.')

    if cuda_available:
        model_ckpt = torch.load(args.ckpt_path)
    else:
        model_ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model_parameters = model_ckpt['model']
    model.load_state_dict(model_parameters)
    model.eval()
    print ('Model loaded.')

    from dataclass import Data
    train_table_path, train_summary_path, train_context_path = args.dataset_prefix + '/train_table.txt', args.dataset_prefix + '/train_reference_summary.txt', \
    args.dataset_prefix + '/train_summary_top_100.txt'
    dev_table_path, dev_summary_path, dev_context_path = args.dataset_prefix + '/dev_table.txt', args.dataset_prefix + '/dev_reference_summary.txt', \
    args.dataset_prefix + '/dev_summary_top_100.txt'
    test_table_path, test_summary_path, test_context_path = args.dataset_prefix + '/test_table.txt', args.dataset_prefix + '/test_reference_summary.txt', \
    args.dataset_prefix + '/test_summary_top_100.txt'

    assert args.dataset_name in ["human", "book", "song"]
    if args.dataset_name == 'human':
        remove_slot_key_list = ['caption', 'death date', 'name', 'article title', 
                                'image', 'fullname', 'full name', 'birthname', 'birth name', 'alias', 
                                'othername', 'imdb', '|name', '|imagesize', 'othername',
                                'image caption', 'image size']
    elif args.dataset_name == 'book':
        remove_slot_key_list = ['name', 'author', 'publisher', 'publication date', 'written by', 'country']
    else: # song
        remove_slot_key_list = ['name']

    model_name = args.model_name
    remove_key_set = set(remove_slot_key_list)
    for header in ['train', 'dev', 'test']:
        print ('------------------------------------------')
        print ('Start inference {} data...'.format(header))
        if header == 'train':
            val_table_path, val_summary_path, val_context_path = train_table_path, train_summary_path, train_context_path
        elif header == 'dev':
            val_table_path, val_summary_path, val_context_path = dev_table_path, dev_summary_path, dev_context_path
        else:
            val_table_path, val_summary_path, val_context_path = test_table_path, test_summary_path, test_context_path

        data = Data(args.model_name, train_table_path, train_summary_path, train_context_path, 
            val_table_path, val_summary_path, val_context_path, 500, args.train_candi_pool_size, 
            args.train_negative_num, args.test_candi_span, args.max_table_len, args.max_tgt_len, remove_key_set)

        test_num = data.test_num
        batch_size = args.batch_size
        test_step_num = int(test_num / batch_size) + 1

        test_ref_text_list, test_pred_text_list = [], []
        with torch.no_grad():
            import progressbar
            print ('Test Evaluation...')
            p = progressbar.ProgressBar(test_step_num)
            p.start()
            for test_step in range(test_step_num):
                p.update(test_step)
                test_all_batch_token_list, test_all_batch_mask_list, test_all_batch_seg_list, \
                test_batch_summary_text_list, test_batch_candidate_summary_list = data.get_next_test_batch(batch_size)
                test_batch_score = trainer(model, test_all_batch_token_list, test_all_batch_mask_list, 
                    test_all_batch_seg_list, cuda_available, device)
                test_batch_select_text = extract_test_pred_text(test_batch_score, test_batch_candidate_summary_list)
                test_pred_text_list += test_batch_select_text
                test_ref_text_list += test_batch_summary_text_list
            p.finish()
            test_ref_text_list = test_ref_text_list[:test_num]
            test_pred_text_list = test_pred_text_list[:test_num]

        if header == 'train':
            test_pred_path = args.save_prefix + '/train_top_{}_rerank_context.txt'.format(args.test_candi_span)
        elif header == 'dev':
            test_pred_path = args.save_prefix + '/dev_top_{}_rerank_context.txt'.format(args.test_candi_span)
        else:
            test_pred_path = args.save_prefix + '/test_top_{}_rerank_context.txt'.format(args.test_candi_span)

        with open(test_pred_path, 'w', encoding = 'utf8') as o:
            for text in test_pred_text_list:
                o.writelines(text + '\n')