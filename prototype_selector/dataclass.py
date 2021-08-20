import sys
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import progressbar

SEP, EOS = '<sep>', '<eos>'
class Data:
    def __init__(self, model_name, train_table_path, train_summary_path, train_context_path, test_table_path, 
        test_summary_path, test_context_path, train_data_num, train_candi_pool_size, train_negative_num, test_candi_span, 
        max_table_len, max_tgt_len, remove_key_set):
        '''
            train_candi_pool_size: number of possible negative candidates for each table
            train_negative_num: number of negatives during training
            test_candi_span: number of candidates considered during testing
            remove_key_set: the key to remove in the input table
        '''
        print ('Loading tokenizer...')
        if model_name.startswith('bert'):
            from transformers import BertTokenizerFast
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        elif model_name.startswith('roberta'):
            from transformers import RobertaTokenizerFast
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        else:
            raise Exception('Wrong Tokenization Mode!')
        self.special_token_list = [SEP, EOS]
        print ('original vocabulary Size %d' % len(self.tokenizer))
        self.tokenizer.add_tokens(self.special_token_list)
        print ('vocabulary size after extension is %d' % len(self.tokenizer))

        self.cls_token, self.cls_token_id, self.sep_token, self.sep_token_id = self.tokenizer.cls_token, \
        self.tokenizer.cls_token_id, self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.max_table_len, self.max_tgt_len = max_table_len, max_tgt_len
        self.train_candi_pool_size, self.train_negative_num, self.test_candi_span = \
        train_candi_pool_size, train_negative_num, test_candi_span
        self.remove_key_set = remove_key_set
        self.train_data_num = train_data_num

        print ('Loading training data...')
        self.train_table_id_list, self.train_summary_id_list, self.train_all_candidate_id_list, \
        self.train_summary_text_list, self.train_all_candidate_text_list = \
        self.load_data(train_table_path, train_summary_path, train_context_path, mode='train')

        print ('Loading test data...')
        self.test_table_id_list, self.test_summary_id_list, self.test_all_candidate_id_list, \
        self.test_summary_text_list, self.test_all_candidate_text_list = \
        self.load_data(test_table_path, test_summary_path, test_context_path, mode='test')

        self.train_num, self.test_num = len(self.train_table_id_list), len(self.test_table_id_list)
        print ('train number is %d, test number is %d' % (self.train_num, self.test_num))
        self.train_idx_list = [i for i in range(self.train_num)]
        self.test_idx_list, self.test_current_idx = [i for i in range(self.test_num)], 0
        
    def load_one_text_id(self, text, max_len):
        text_id_list = self.tokenizer.encode(text, max_length=512, truncation=True, add_special_tokens=False)[:max_len]
        return text_id_list
    
    def parse_table_text(self, table_text):
        flag = True
        res_str = ''
        item_list = table_text.strip('\n').split('\t')
        for item in item_list:
            one_item_list = item.split(':')
            assert len(one_item_list) == 2
            slot_key, slot_value = one_item_list[0].strip(), one_item_list[1].strip()
            if slot_key in self.remove_key_set:
                continue
            else:
                one_res_str = slot_key + ' ' + SEP + ' ' + slot_value + ' ' + EOS
                res_str += one_res_str + ' '
        res_str = res_str.strip()
        if len(res_str) == 0: # no valid string
            flag = False
        return res_str, flag

    def load_table(self, table_path):
        res_id_list = []
        with open(table_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            print ('Loading Table...')
            p = progressbar.ProgressBar(len(lines))
            p.start()
            idx = 0
            for l in lines:
                p.update(idx)
                one_table_text, one_table_flag = self.parse_table_text(l.strip('\n'))
                if one_table_flag:
                    one_res_id_list = self.load_one_text_id(one_table_text.strip('\n'), self.max_table_len)
                else:
                    one_res_id_list = []
                res_id_list.append(one_res_id_list)
            p.finish()
        return res_id_list

    def load_summary(self, summary_path):
        res_id_list = []
        with open(summary_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            print ('Loading Summary...')
            p = progressbar.ProgressBar(len(lines))
            p.start()
            idx = 0
            for l in lines:
                p.update(idx)
                one_res_id_list = self.load_one_text_id(l.strip('\n'), self.max_tgt_len) 
                res_id_list.append(one_res_id_list)
            p.finish()
        return res_id_list

    def load_candidates(self, candidate_text, mode):
        if mode == 'train':
            select_num = self.train_candi_pool_size
        elif mode == 'test':
            select_num = self.test_candi_span
        else:
            raise Exception('Wrong Mode!!!')
        candidate_text_list = candidate_text.split('\t')[:select_num]
        candidate_id_list = []
        for text in candidate_text_list:
            one_text_id = self.load_one_text_id(text, self.max_tgt_len)
            assert len(one_text_id) > 0
            candidate_id_list.append(one_text_id)
        assert len(candidate_id_list) == select_num
        return candidate_id_list, candidate_text_list

    def load_data(self, table_path, summary_path, candidate_summary_path, mode):
        tmp_table_id_list = self.load_table(table_path)
        tmp_summary_id_list = self.load_summary(summary_path)
        with open(summary_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            tmp_summary_text_list = [l.strip('\n') for l in lines]

        with open(candidate_summary_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            tmp_all_candidate_text_list = [l.strip('\n') for l in lines]
        assert len(tmp_all_candidate_text_list) == len(tmp_table_id_list)

        data_num = len(tmp_table_id_list)
        table_id_list, summary_id_list, all_candidate_id_list, summary_text_list, \
        all_candidate_text_list = [], [], [], [], []
        print ('Loading candidates...')
        p = progressbar.ProgressBar(data_num)
        p.start()
        for idx in range(data_num):
            p.update(idx)
            one_table_id_list = tmp_table_id_list[idx]
            if len(one_table_id_list) == 0:
                print (idx)
                continue
            else:
                table_id_list.append(tmp_table_id_list[idx])
                summary_id_list.append(tmp_summary_id_list[idx])
                summary_text_list.append(tmp_summary_text_list[idx])
                one_candidate_text = tmp_all_candidate_text_list[idx]
                one_candidate_text_id_list, one_candidate_text_list = self.load_candidates(one_candidate_text, mode)
                all_candidate_id_list.append(one_candidate_text_id_list)
                all_candidate_text_list.append(one_candidate_text_list)
        p.finish()
        if mode == 'train':
            return table_id_list[:self.train_data_num], summary_id_list[:self.train_data_num], \
            all_candidate_id_list[:self.train_data_num], summary_text_list[:self.train_data_num], \
            all_candidate_text_list[:self.train_data_num]
        elif mode == 'test':
            return table_id_list, summary_id_list, all_candidate_id_list, summary_text_list, all_candidate_text_list
        else:
            raise Exception('Wrong Mode!!!')

    def padding(self, batch_id_list):
        batch_tensor_list = [torch.LongTensor(one_id_list) for one_id_list in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_tensor_list, batch_first=True, padding_value=self.pad_token_id)
        batch_mask = torch.ones_like(batch_tensor)
        batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_tensor, batch_mask

    def padding_segment(self, batch_seg_id_list):
        res_seg_id_list = []
        max_len = max([len(item) for item in batch_seg_id_list])
        for one_seg_id_list in batch_seg_id_list:
            one_len_diff = max_len - len(one_seg_id_list)
            res_seg_id_list.append(one_seg_id_list + [1 for _ in range(one_len_diff)])
        seg_id_tensor = torch.LongTensor(res_seg_id_list)
        assert seg_id_tensor.size() == torch.Size([len(batch_seg_id_list), max_len])
        return seg_id_tensor

    def process_batch_data(self, batch_table_id_list, batch_context_id_list):
        batch_token_id_list, batch_seg_id_list = [], []
        assert len(batch_table_id_list) == len(batch_context_id_list)
        bsz = len(batch_context_id_list)
        for idx in range(bsz):
            one_table_id_list = [self.cls_token_id] + batch_table_id_list[idx] + [self.sep_token_id]
            one_table_seg_id_list = [0 for _ in one_table_id_list]
            one_context_id_list = batch_context_id_list[idx] + [self.sep_token_id]
            one_context_seg_id_list = [1 for _ in one_context_id_list]
            one_token_id_list = one_table_id_list + one_context_id_list
            batch_token_id_list.append(one_token_id_list)
            one_seg_id_list = one_table_seg_id_list + one_context_seg_id_list
            batch_seg_id_list.append(one_seg_id_list)
        batch_token_tensor, batch_token_mask = self.padding(batch_token_id_list)
        batch_seg_tensor = self.padding_segment(batch_seg_id_list)
        assert batch_token_tensor.size() == batch_token_mask.size()
        assert batch_seg_tensor.size() == batch_token_mask.size()
        return batch_token_tensor, batch_token_mask, batch_seg_tensor

    def get_one_train_negative_id_list(self, table_id):
        candi_id_list = self.train_all_candidate_id_list[table_id]
        candi_num = len(candi_id_list)
        neg_idx_list = random.sample([i for i in range(candi_num)], self.train_negative_num)
        neg_candi_id_list = [candi_id_list[idx] for idx in neg_idx_list]
        return neg_candi_id_list

    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_table_id_list, batch_summary_id_list, all_batch_neg_candi_list \
        = [], [], [[] for _ in range(self.train_negative_num)]
        for idx in batch_idx_list:
            one_table_id_list = self.train_table_id_list[idx]
            batch_table_id_list.append(one_table_id_list)
            one_summary_id_list = self.train_summary_id_list[idx]
            batch_summary_id_list.append(one_summary_id_list)
            one_candi_neg_id_list = self.get_one_train_negative_id_list(idx)
            for k in range(self.train_negative_num):
                all_batch_neg_candi_list[k].append(one_candi_neg_id_list[k])
        all_batch_token_list, all_batch_mask_list, all_batch_seg_list = [], [], []
        # first process reference data
        batch_ref_token_tensor, batch_ref_mask_tensor, batch_ref_seg_tensor = \
        self.process_batch_data(batch_table_id_list, batch_summary_id_list)
        all_batch_token_list.append(batch_ref_token_tensor)
        all_batch_mask_list.append(batch_ref_mask_tensor)
        all_batch_seg_list.append(batch_ref_seg_tensor)
        for neg_k in range(self.train_negative_num):
            batch_neg_token_tensor, batch_neg_mask_tensor, batch_neg_seg_tensor = \
            self.process_batch_data(batch_table_id_list, all_batch_neg_candi_list[neg_k])
            all_batch_token_list.append(batch_neg_token_tensor)
            all_batch_mask_list.append(batch_neg_mask_tensor)
            all_batch_seg_list.append(batch_neg_seg_tensor)
        return all_batch_token_list, all_batch_mask_list, all_batch_seg_list

    def get_next_test_batch(self, batch_size):
        batch_table_id_list, all_candi_id_list = [], [[] for _ in range(self.test_candi_span)]
        batch_summary_text_list, batch_candidate_summary_list = [], []
        if self.test_current_idx + batch_size < self.test_num - 1:
            for i in range(batch_size):
                curr_idx = self.test_current_idx + i

                one_table_id_list = self.test_table_id_list[curr_idx]
                batch_table_id_list.append(one_table_id_list)
                one_candi_id_list = self.test_all_candidate_id_list[curr_idx]
                for k in range(self.test_candi_span):
                    all_candi_id_list[k].append(one_candi_id_list[k])
                batch_summary_text_list.append(self.test_summary_text_list[curr_idx])
                batch_candidate_summary_list.append(self.test_all_candidate_text_list[curr_idx][:self.test_candi_span])
            self.test_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = self.test_current_idx + i
                if curr_idx > self.test_num - 1: 
                    curr_idx = 0
                    self.test_current_idx = 0
                else:
                    pass
                one_table_id_list = self.test_table_id_list[curr_idx]
                batch_table_id_list.append(one_table_id_list)
                one_candi_id_list = self.test_all_candidate_id_list[curr_idx]
                for k in range(self.test_candi_span):
                    all_candi_id_list[k].append(one_candi_id_list[k])
                batch_summary_text_list.append(self.test_summary_text_list[curr_idx])
                batch_candidate_summary_list.append(self.test_all_candidate_text_list[curr_idx][:self.test_candi_span])
            self.test_current_idx = 0

        all_batch_token_list, all_batch_mask_list, all_batch_seg_list = [], [], []
        for k in range(self.test_candi_span):
            batch_token_tensor, batch_mask_tensor, batch_seg_tensor = \
            self.process_batch_data(batch_table_id_list, all_candi_id_list[k])
            all_batch_token_list.append(batch_token_tensor)
            all_batch_mask_list.append(batch_mask_tensor)
            all_batch_seg_list.append(batch_seg_tensor)
        return all_batch_token_list, all_batch_mask_list, all_batch_seg_list, batch_summary_text_list, batch_candidate_summary_list
