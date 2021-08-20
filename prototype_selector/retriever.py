import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig

def trainer(model, all_batch_token, all_batch_mask, all_batch_seg, is_cuda, device):
    assert len(all_batch_token) == len(all_batch_mask)
    assert len(all_batch_mask) == len(all_batch_seg)
    bsz = all_batch_token[0].size()[0]
    batch_score_list = []
    item_num = len(all_batch_token)
    for k in range(item_num):
        batch_token_inp = all_batch_token[k]
        batch_mask_inp = all_batch_mask[k]
        batch_seg_inp = all_batch_seg[k]
        if is_cuda:
            batch_token_inp = batch_token_inp.cuda(device)
            batch_mask_inp = batch_mask_inp.cuda(device)
            batch_seg_inp = batch_seg_inp.cuda(device)
        one_score = model(batch_token_inp, batch_mask_inp, batch_seg_inp)
        batch_score_list.append(one_score)            
    batch_scores = torch.cat(batch_score_list, dim = -1)
    assert batch_scores.size() == torch.Size([bsz, item_num])
    return batch_scores

class Model(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(Model, self).__init__()
        self.model_name = model_name
        if model_name.startswith('bert'):
            self.plm = BertModel.from_pretrained(model_name)
            self.config = BertConfig.from_pretrained(model_name)
        elif model_name.startswith('roberta'):
            self.plm = RobertaModel.from_pretrained(model_name)
            self.config = RobertaConfig.from_pretrained(model_name)
            self.plm.config.type_vocab_size = 2 
            self.plm.embeddings.token_type_embeddings = nn.Embedding(2, self.plm.config.hidden_size)
            self.plm.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
        else:
            raise Exception('Wrong Model Configuration!')
        self.embed_dim = self.config.hidden_size
        self.plm.resize_token_embeddings(len(tokenizer)) 
        self.final_linear = nn.Linear(self.embed_dim, 1)
        nn.init.xavier_uniform_(self.final_linear.weight)

    def forward(self, batch_input, batch_mask, batch_seg):
        bsz, input_len = batch_input.size()
        outputs = self.plm(input_ids=batch_input, attention_mask=batch_mask, token_type_ids=batch_seg)
        representation = outputs.last_hidden_state
        assert representation.size() == torch.Size([bsz, input_len, self.embed_dim])
        cls_vec = representation.transpose(0,1)[0]
        assert cls_vec.size() == torch.Size([bsz, self.embed_dim])
        logits = self.final_linear(cls_vec)
        assert logits.size() == torch.Size([bsz, 1])
        return logits