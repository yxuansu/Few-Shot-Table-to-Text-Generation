import os
import sys
import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import operator
from operator import itemgetter
from transformers import AdamW, get_linear_schedule_with_warmup
import yaml
import subprocess
from subprocess import call
import argparse
from dataclass import Data
from evaluation import eval_multi_ref_bleu
import progressbar

def get_adafactor_optimizer(model, args):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    from transformers.optimization import Adafactor, AdafactorSchedule
    print ('Use Adafactor Optimizer for Training.')
    optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
            )
    scheduler = None
    return optimizer, scheduler

def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--train_data_num', type=int)
    parser.add_argument('--dataset_prefix', type=str, help="the path that stores the data")
    parser.add_argument('--context_prefix', type=str, help="where the reranked context are.")
    parser.add_argument('--max_table_len', type=int, default=300, help="maximum table length")
    parser.add_argument('--max_tgt_len', type=int, default=90, help="maximum context length")
    parser.add_argument('--max_context_len', default=50, type=int)
    parser.add_argument('--context_num', type=int, default=3, help="number of prototypes used by the generator")
    parser.add_argument('--add_context', type=str, default='True', help="whether include context in the generator input.")
    # model configuration
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--max_decode_len', type=int, default=90)

    # training configuration
    parser.add_argument('--optimizer_name', default='adafactor', type=str, help='which optimizer to use during training, adam or adafactor')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--warmup_steps", type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--total_steps", type=int, help="Total training steps.")
    parser.add_argument("--print_every", type=int, help="For how many steps to print the result.")
    parser.add_argument("--eval_every", type=int, help="For how many steps to evaluate the result.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")  
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation step.")
    parser.add_argument("--ckpt_save_path", type=str, help="directory to save the model parameters.")
    return parser.parse_args()

import argparse
if __name__ == '__main__':
    args = parse_config()

    import os
    if os.path.exists(args.ckpt_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(args.ckpt_save_path, exist_ok=True)

    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU training.')
    else:
        pass
    device = torch.device('cuda')

    train_table_path, train_summary_path = args.dataset_prefix + '/train_table.txt', args.dataset_prefix + '/train_reference_summary.txt'
    dev_table_path, dev_summary_path = args.dataset_prefix + '/dev_table.txt', args.dataset_prefix + '/dev_reference_summary.txt'

    train_context_path = args.context_prefix + '/train_top_20_rerank_context.txt'
    dev_context_path = args.context_prefix + '/dev_top_20_rerank_context.txt'

    print ('Start loading data...')
    assert args.add_context in ['True', 'False']
    if args.add_context == 'True':
        add_context = True
    else:
        add_context = False
    data = Data(train_table_path, train_summary_path, train_context_path, args.train_data_num, dev_table_path, dev_summary_path, dev_context_path, 
        args.max_table_len, args.max_tgt_len, args.max_context_len, args.context_num, args.model_name, add_context=add_context)
    print ('Data loaded.')

    print ('Loading Model.')
    if args.model_name.startswith('t5'): 
        print ('Use T5 Model...')
        from modelling.T5Model import T5Gen_Model
        model = T5Gen_Model(model_name=args.model_name, tokenizer=data.decode_tokenizer, 
            max_decode_len=args.max_decode_len, dropout=args.dropout)
    elif args.model_name.startswith('facebook/bart'):
        print ('Use Bart Model...') 
        from modelling.BARTModel import BARTGen_Model
        model = BARTGen_Model(model_name=args.model_name, tokenizer=data.decode_tokenizer, 
            max_decode_len=args.max_decode_len, dropout=args.dropout)
    else:
        raise Exception('Wrong Model Mode!!!')

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print ('Model loaded')

    if args.optimizer_name == 'adam':
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, 
                num_training_steps=args.total_steps)
    elif args.optimizer_name == 'adafactor':
        optimizer, scheduler = get_adafactor_optimizer(model, args)
    else:
        raise Exception('Wrong Optimizer Configuration!')
    optimizer.zero_grad()

    train_num, dev_num = data.train_num, data.dev_num
    batch_size = args.number_of_gpu * args.batch_size_per_gpu
    train_step_num = int(train_num / batch_size) + 1
    dev_step_num = int(dev_num / batch_size) + 1

    batches_processed = 0
    total_steps = args.total_steps
    print_every, eval_every = args.print_every, args.eval_every

    train_loss, max_dev_bleu = 0., 0.

    model.train()
    for one_step in range(total_steps):
        epoch = one_step // train_step_num

        train_batch_src_tensor, train_batch_src_mask, train_batch_tgt_in_tensor, train_batch_tgt_out_tensor, _ = \
        data.get_next_train_batch(args.number_of_gpu * args.batch_size_per_gpu)
        if cuda_available:
            train_batch_src_tensor = train_batch_src_tensor.cuda(device)
            train_batch_src_mask = train_batch_src_mask.cuda(device)
            train_batch_tgt_in_tensor = train_batch_tgt_in_tensor.cuda(device)
            train_batch_tgt_out_tensor = train_batch_tgt_out_tensor.cuda(device)

        if args.model_name.startswith('t5'):
            loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_tgt_out_tensor)
        elif args.model_name.startswith('facebook/bart'):
            loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_tgt_in_tensor, train_batch_tgt_out_tensor)
        else:
            raise Exception('Wrong Model Mode!')
        loss = loss.mean()
        loss.backward()
        train_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if (one_step+1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            if args.optimizer_name == 'adam':
                scheduler.step() # only update learning rate when using adam
            optimizer.zero_grad()

        if (one_step+1) % args.print_every == 0:
            train_loss = round(train_loss / args.print_every, 2)
            print ('At epoch {}, total training steps {}, train mle loss is {}, max dev score is {}'.format(
                epoch, one_step, train_loss, max_dev_bleu))
            train_loss = 0.

        if (one_step+1) % args.eval_every == 0:
            model.eval()
            dev_reference_text_list, dev_output_text_list = [], []
            with torch.no_grad():
                print ('Perform Evaluation...')
                import progressbar
                dev_step_num = (data.dev_num // (args.number_of_gpu * args.batch_size_per_gpu)) + 1
                p = progressbar.ProgressBar(dev_step_num)
                p.start()
                for dev_step in range(dev_step_num):
                    p.update(dev_step)
                    dev_batch_src_tensor, dev_batch_src_mask, dev_batch_tgt_in_tensor, dev_batch_tgt_out_tensor, \
                    dev_batch_reference_text_list = data.get_next_dev_batch(args.number_of_gpu * args.batch_size_per_gpu)

                    if cuda_available:
                        dev_batch_src_tensor = dev_batch_src_tensor.cuda(device)
                        dev_batch_src_mask = dev_batch_src_mask.cuda(device)
                        dev_batch_tgt_in_tensor = dev_batch_tgt_in_tensor.cuda(device)
                        dev_batch_tgt_out_tensor = dev_batch_tgt_out_tensor.cuda(device)
                    if multi_gpu_training:
                        decoded_result = model.module.generate(dev_batch_src_tensor, dev_batch_src_mask, tokenized_data=True)
                    else:
                        decoded_result = model.generate(dev_batch_src_tensor, dev_batch_src_mask, tokenized_data=True)
                    dev_output_text_list += decoded_result
                    dev_reference_text_list += dev_batch_reference_text_list
                p.finish()
                dev_output_text_list = dev_output_text_list[:data.dev_num]
                dev_reference_text_list = dev_reference_text_list[:data.dev_num]

                dev_bleu = eval_multi_ref_bleu([dev_reference_text_list], dev_output_text_list, r'./', 'dev_')
                if dev_bleu > max_dev_bleu:
                    max_dev_bleu = dev_bleu
                    model_save_path = args.ckpt_save_path + '/epoch_{}_validation_bleu_{}'.format(epoch, round(dev_bleu, 2))
                    import os
                    if os.path.exists(model_save_path):
                        pass
                    else: # recursively construct directory
                        os.makedirs(model_save_path, exist_ok=True)
                    if multi_gpu_training:
                        model.module.save_model(model_save_path)
                    else:
                        model.save_model(model_save_path)

                    with open(model_save_path + '/dev_reference.txt', 'w', encoding = 'utf8') as o:
                        for text in dev_reference_text_list:
                            o.writelines(text + '\n')

                    with open(model_save_path + '/dev_prediction.txt', 'w', encoding = 'utf8') as o:
                        for text in dev_output_text_list:
                            o.writelines(text + '\n')

                    import os
                    from operator import itemgetter
                    fileData = {}
                    test_output_dir = args.ckpt_save_path
                    for fname in os.listdir(test_output_dir):
                        if fname.startswith('epoch'):
                            fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                        else:
                            pass
                    sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                    max_save_num = 1
                    if len(sortedFiles) < max_save_num:
                        pass
                    else:
                        delete = len(sortedFiles) - max_save_num
                        for x in range(0, delete):
                            one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                            #print (one_folder_name)
                            os.system('rm -r ' + one_folder_name)
                print ('----------------------------------------------------------------')
                print ('At epoch {}, current test bleu is {}, maximum test bleu is {}'.format(epoch, dev_bleu, max_dev_bleu))
                print ('----------------------------------------------------------------')
            model.train()
