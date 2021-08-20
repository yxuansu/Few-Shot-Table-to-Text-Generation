import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config

def get_special_token_list(tokenizer):
    res_list = []
    for key in tokenizer.special_tokens_map:
        one_token = tokenizer.special_tokens_map[key]
        if type(one_token) == str:
            res_list.append(one_token)
    return res_list

class T5Gen_Model(nn.Module):
	def __init__(self, model_name, tokenizer, max_decode_len, dropout):
		super().__init__()
		self.tokenizer = tokenizer # tokenizer with extended vocabulary
		self.max_decode_len = max_decode_len
		self.tokenizer_special_token_set = set(get_special_token_list(self.tokenizer))

		print ('Initializing Huggingface T5 model...')
		t5_config = T5Config.from_pretrained(model_name)
		t5_config.__dict__["dropout"] = dropout
		self.model = T5ForConditionalGeneration.from_pretrained(model_name, config = t5_config)
		print ('Resizing Token Embeddings...')
		self.model.resize_token_embeddings(len(self.tokenizer)) 

	def forward(self, src_input, src_mask, tgt_output):
		#outputs = self.model(input_ids=src_input, attention_mask=src_mask,
		#					 decoder_input_ids=tgt_input, labels=tgt_output)
		outputs = self.model(input_ids=src_input, attention_mask=src_mask, labels=tgt_output)
		loss = outputs[0]#.mean()
		return loss

	def generate(self, src_input, src_mask, tokenized_data=False):
		result_list = []
		outputs = self.model.generate(input_ids=src_input, attention_mask=src_mask, max_length=self.max_decode_len)
		for predicted_ids in outputs:
			if tokenized_data:
				one_token_list = self.tokenizer.convert_ids_to_tokens(predicted_ids)
				pred_tokens = []
				for token in one_token_list:
					if token in self.tokenizer_special_token_set:
						continue
					else:
						pred_tokens.append(token)
				one_result = self.tokenizer.convert_tokens_to_string(pred_tokens)
			else:
				one_result = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
			result_list.append(one_result)
		return result_list
		
	def save_model(self, ckpt_save_path):
		import os
		if not os.path.exists(ckpt_save_path):
			os.mkdir(ckpt_save_path)
		self.model.save_pretrained(ckpt_save_path)
		self.tokenizer.save_pretrained(ckpt_save_path)