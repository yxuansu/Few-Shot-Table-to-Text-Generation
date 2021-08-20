import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
#from dataclass import Data
import numpy as np
import argparse
import logging
import nltk
from transformers import AdamW, get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu
from retriever import trainer

def eval_bleu_score(pred_text_list, ref_text_list):
    score_list = []
    assert len(pred_text_list) == len(ref_text_list)
    for idx in range(len(pred_text_list)):
        one_pred_text_list = pred_text_list[idx].split()
        one_ref_text_list = ref_text_list[idx].split()
        one_score = sentence_bleu([one_ref_text_list], one_pred_text_list)
        score_list.append(one_score)
    return np.mean(score_list) * 100 

def hinge_loss(scores, margin):
    # y_pred: bsz x candi_num
    loss = torch.nn.functional.relu(margin - (torch.unsqueeze(scores[:, 0], -1) - scores[:, 1:]))
    return torch.mean(loss)

def extract_test_pred_text(test_batch_score, test_batch_candidate_text_list):
    pred_text_list = []
    test_batch_score_list = test_batch_score.detach().cpu().numpy()
    bsz = len(test_batch_score_list)
    for idx in range(bsz):
        one_score_list = test_batch_score_list[idx]
        one_select_idx = np.argsort(one_score_list)[::-1][0]
        one_select_text = test_batch_candidate_text_list[idx][one_select_idx]
        pred_text_list.append(one_select_text)
    return pred_text_list

def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset_name', type=str, help="human or song or book")
    parser.add_argument('--model_name', type=str, help="e.g. bert-base...")
    parser.add_argument('--dataset_prefix', type=str, help="the path that stores the data")
    parser.add_argument('--train_data_num', type=int)
    parser.add_argument('--train_candi_pool_size', type=int, default=50, 
        help="Randomly selecting negative examples from the top-k retrieved candidates provided by the IR system.")
    parser.add_argument('--train_negative_num', type=int, default=5, 
        help="number of randomly selected negatives from the retrieved candidates from the IR system.")
    parser.add_argument('--test_candi_span', type=int, default=20, 
        help="reranking the best response from top-n candidates from the IR system.")
    parser.add_argument('--max_table_len', type=int, default=250)
    parser.add_argument('--max_tgt_len', type=int, default=80)
    # training configuration
    parser.add_argument('--loss_margin', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--total_steps', type=int)
    parser.add_argument('--update_steps', type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--eval_every', type=int)
    # learning configuration
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_output_dir', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print ('Cuda is available.')
    else:
        print ('Cuda is not available.')

    args = parse_config()
    device = args.gpu_id
    from dataclass import Data
    train_table_path, train_summary_path, train_context_path = args.dataset_prefix + '/train_table.txt', args.dataset_prefix + '/train_reference_summary.txt', \
    args.dataset_prefix + '/train_summary_top_100.txt'
    dev_table_path, dev_summary_path, dev_context_path = args.dataset_prefix + '/dev_table.txt', args.dataset_prefix + '/dev_reference_summary.txt', \
    args.dataset_prefix + '/dev_summary_top_100.txt'
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
    remove_key_set = set(remove_slot_key_list)
    model_name = args.model_name
    data = Data(args.model_name, train_table_path, train_summary_path, train_context_path, 
        dev_table_path, dev_summary_path, dev_context_path, args.train_data_num, 
        args.train_candi_pool_size, args.train_negative_num, args.test_candi_span, args.max_table_len, 
        args.max_tgt_len, remove_key_set)

    print ('Initializing Model...')
    from retriever import Model
    model = Model(args.model_name, data.tokenizer)
    if cuda_available:
        model = model.cuda(device)
    print ('Model Loaded.')

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, 
            num_training_steps=args.total_steps)
    optimizer.zero_grad()

    train_num, test_num = data.train_num, data.test_num
    batch_size = args.batch_size
    train_step_num = int(train_num / batch_size) + 1
    test_step_num = int(test_num / batch_size) + 1

    batches_processed = 0
    total_steps = args.total_steps
    print_every, eval_every = args.print_every, args.eval_every
    train_loss_accum, max_test_bleu = 0, 0.
    model.train()
    for one_step in range(total_steps):
        epoch = one_step // train_step_num
        batches_processed += 1

        train_all_batch_token_list, train_all_batch_mask_list, train_all_batch_seg_list = \
        data.get_next_train_batch(batch_size)
        train_batch_score = trainer(model, train_all_batch_token_list, train_all_batch_mask_list, 
            train_all_batch_seg_list, cuda_available, device)

        train_loss = hinge_loss(train_batch_score, args.loss_margin)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if batches_processed % args.update_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        train_loss_accum += train_loss.item()
        if batches_processed % print_every == 0:
            curr_train_loss = train_loss_accum / print_every
            print ('At epoch %d, batch %d, train loss %.5f, max score is %.5f' % 
                (epoch, batches_processed, curr_train_loss, max_test_bleu))
            train_loss_accum = 0.

        test_output_dir = args.test_output_dir
        if batches_processed % eval_every == 0:
            model.eval()
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

                one_test_bleu = eval_bleu_score(test_pred_text_list, test_ref_text_list)
                print ('----------------------------------------------------------------')
                print ('At epoch %d, batch %d, test bleu %5f' \
                    % (epoch, batches_processed, one_test_bleu))
                save_name = '/epoch_%d_batch_%d_test_bleu_%.3f' \
                        % (epoch, batches_processed, one_test_bleu)
                print ('----------------------------------------------------------------')

                if one_test_bleu > max_test_bleu:
                    # keep track of model's best result
                    print ('Saving Model...')
                    model_save_path = test_output_dir + save_name
                    import os
                    if os.path.exists(test_output_dir):
                        pass
                    else: # recursively construct directory
                        os.makedirs(test_output_dir, exist_ok=True)

                    torch.save({'model':model.state_dict()}, model_save_path)

                    max_test_bleu = one_test_bleu
                    fileData = {}
                    for fname in os.listdir(test_output_dir):
                        if fname.startswith('epoch'):
                            fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                        else:
                            pass
                    sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                    
                    if len(sortedFiles) < 1:
                        pass
                    else:
                        delete = len(sortedFiles) - 1
                        for x in range(0, delete):
                            os.remove(test_output_dir + '/' + sortedFiles[x][0])
            model.train()
