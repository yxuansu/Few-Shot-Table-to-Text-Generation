import torch
import numpy as np
from subprocess import call
import subprocess
import os

def eval_multi_ref_bleu(ref_text_list, prediction_text_list, test_out_dir, header):
    ref_text_path_list = []
    for i in range(len(ref_text_list)):
        assert len(ref_text_list[i]) == len(prediction_text_list)
        ref_path = test_out_dir + '/' + header + 'ref_path_' + str(i) + '.txt'
        ref_text_path_list.append(ref_path)
        with open(ref_path, 'w', encoding = 'utf8') as o:
            for text in ref_text_list[i]:
                o.writelines(text + '\n')
                
    pred_path = test_out_dir + '/' + header + 'prediction_path.txt'
    with open(pred_path, 'w', encoding = 'utf8') as o:
        for text in prediction_text_list:
            o.writelines(text + '\n')
    res = compute_multi_reference_bleu(ref_text_path_list, pred_path)
    import os
    os.remove(ref_path)
    os.remove(pred_path)
    return res

def compute_multi_reference_bleu(reference_path_list, predictions_file_path):
    command = 'perl ../../../multi-bleu.perl ' 
    for file in reference_path_list:
        command += file + ' '
    command += '<' + ' ' + predictions_file_path
    result = subprocess.run(command,
        check=True,
        shell=True,
        stdout=subprocess.PIPE,)
    res = result.stdout.decode("utf-8") 
    return float(res.split(',')[0].split('=')[1].strip())

def eval_bleu(reference_file_path, predictions_file_path):
    command = 'perl ./multi-bleu.perl ' + reference_file_path + ' ' + '<' + ' ' + predictions_file_path
    result = subprocess.run(command,
        check=True,
        shell=True,
        stdout=subprocess.PIPE,)
    res = result.stdout.decode("utf-8") 
    return float(res.split()[2].strip(','))

def map_subword_data(subword_data_list, out_f):
    tmp_f = r'./tmp_f.txt'
    with open(tmp_f, 'w', encoding = 'utf8') as o:
        for data in subword_data_list:
            one_text = data.strip()
            o.writelines(one_text + '\n')
    command = r"sed -r 's/(@@ )|(@@ ?$)//g' < " + tmp_f + " > " + out_f
    call([command], shell=True)
    os.remove(tmp_f)

def eval_result(ref_subword_data_list, pred_subword_data_list):
    ref_f = r'./eva_ref_file.txt'
    map_subword_data(ref_subword_data_list, ref_f)
    pred_f = r'./eva_pred_file.txt'
    map_subword_data(pred_subword_data_list, pred_f)
    result = eval_bleu(ref_f, pred_f)
    return result

def map_text(batch_greedy_result, vocab):
    padding_idx = vocab.padding_idx
    sos_idx = vocab.sos_idx
    eos_idx = vocab.eos_idx
    unk_idx = vocab.unk_idx

    batch_result = batch_greedy_result.cpu().detach().numpy()

    result = []
    for one_result in batch_result:
        one_res = []
        for one_idx in one_result:
            one_idx = int(one_idx)
            if one_idx == padding_idx or one_idx == sos_idx or one_idx == eos_idx or one_idx == unk_idx:
                continue
            else:
                one_token = vocab.idx_token_dict[one_idx]
                one_res.append(one_token)
        one_res_text = ' '.join(one_res)
        result.append(one_res_text)
    return result

def bleu_evaluation(ref_text_list, pred_text_list):
    # ref_text_list : list of reference text
    # pred_text_list : list of prediction text
    result = eval_result(ref_text_list, pred_text_list)
    return result
