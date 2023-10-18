import random
import os
from torch.utils.data import Dataset
import torch
import csv
from toolz.sandbox import unzip
from cytoolz import concat
import json
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from .data import (get_ids_and_lens, pad_tensors,
                   get_gather_index)
from utils.tsv_file import TSVFile
import base64
import pickle
from tqdm import tqdm
import clip

class PMR_ChunkAlign_Dataset_align_ensemble_T(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, roberta_tokenizer, VCR_example_file, VCR_chunk_mask_file,
                 flickr_feat_file,
                 roberta_example_file,
                 preprocess,
                 clip_example_file,
                 device,
                 max_img_seq_length=50,
                 is_train=True, heat_index=None):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.roberta_toker = roberta_tokenizer
        self.chunk_mask_dict = pickle.load(open(VCR_chunk_mask_file, 'rb'))
        self.VCR_annot_dict = pickle.load(open(VCR_example_file, 'rb'))
        self.roberta_annot_dict = pickle.load((open(roberta_example_file, 'rb')))
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
        self.cls = self.bert_toker.cls_token
        self.sep = self.bert_toker.sep_token
        self.rcls = self.roberta_toker.bos_token
        self.rsep = self.roberta_toker.eos_token
        self.preprocess = preprocess
        self.device = device

        if heat_index is not None:
            self.VCR_annot_dict = self.VCR_annot_dict[heat_index:heat_index + 1]


    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.VCR_annot_dict)
    
    def p_random(self, arr1, arr2):
        assert len(arr1) == len(arr2), "Length does not match."
        assert sum(arr2) == 1 , "Total rate is not 1."

        sup_list = [len(str(i).split(".")[-1]) for i in arr2]
        top = 10 ** max(sup_list)
        new_rate = [int(i*top) for i in arr2]
        rate_arr = []
        for i in range(1,len(new_rate)+1):
            rate_arr.append(sum(new_rate[:i]))
        rand = random.randint(1,top)
        data = None
        for i in range(len(rate_arr)):
            if rand <= rate_arr[i]:
                data = arr1[i]
                break
        return data


    def __getitem__(self, i):

        image_ori_path = "pmr_data/images/"


        example = self.VCR_annot_dict[i]
        roberta_example = self.roberta_annot_dict[i]

        image_input = None

        # --------------CALeC---------------
        id = example['image_id']
        num_id = id.split("-")[1]
        str_id = "img-" + str(num_id)
        try:
            answer_label = example['answer_label']
        except:
            answer_label = 0


        image_feat = self.image_feat_dict[str_id]
        img_feat = image_feat['features'].cuda(self.device)
        img_mask = image_feat['img_mask'].cuda(self.device)
        que_tokens = example['sent'].lower()
        premise_tokens = self.bert_toker.tokenize(que_tokens)
        que_tokens = image_feat['objects']

        outputs = []

        prompt_text1 = 'Given an image with feature <mask>, the alignment feature between text and image is <mask>, objects identified as '
        prompt_text2 = 'And the premise text is '
        
        r_que_tokens = roberta_example['sent'].lower()

        r_image_tokens = [self.rcls] + self.roberta_toker.tokenize(prompt_text1)
        r_que_tokens = self.roberta_toker.tokenize(prompt_text2 + r_que_tokens)



        for ans_idx in range(len(example['answer_choices'])):

            clip_ans_tokens = None

            r_ans = roberta_example['answer_choices'][ans_idx]
            r_ans_tokens = self.roberta_toker.tokenize('Is the following answer correct? ' + ' '.join(r_ans.split(' , ')))
            r_text_tokens = [self.rsep] + r_que_tokens + [self.rsep] + r_ans_tokens
            objects_begin = len(r_image_tokens)
            object_num = []
            swap_position = [-1] * len(r_text_tokens)
            for token_idx, token in enumerate(r_text_tokens):
                if '<|det' in token:
                    index = token[5:].split('|')[0]
                    swap_position[token_idx] = int(index)
                    if int(index) not in object_num:
                        object_num.append(int(index))
            object_num.sort()
            # object_num=list(range(len(que_tokens))) #no select
            
            swap_position = [-1] * (len(r_image_tokens) + len(object_num)) + swap_position
            swap_position = torch.tensor(swap_position).cuda(self.device)
            
            # swap_opt = self.p_random([0,1,2,3],[0.25,0.25,0.25,0.25])
            swap_opt = 3
            r_input_tokens = r_image_tokens + ['<mask>'] * len(object_num) + r_text_tokens


            r_input_ids = self.roberta_toker.convert_tokens_to_ids(r_input_tokens)
            r_input_ids = torch.tensor(r_input_ids).cuda(self.device)
            
            r_mask_len = r_input_ids.size(0)
            r_input_mask = torch.ones(r_mask_len).cuda(self.device)
            all_objects_mask = torch.ones(len(que_tokens)).cuda(self.device)


            r_segment_ids = torch.zeros(len(r_input_tokens), dtype=torch.int64).cuda(self.device)

            ans = example['answer_choices'][ans_idx]
            ans_tokens = self.bert_toker.tokenize(ans)
            input_tokens = [self.cls] + premise_tokens + [self.sep] + ans_tokens + [self.sep]
            region_tokens = [0] * len(input_tokens)


            for token_idx, token in enumerate(input_tokens):
                if '<|det' in token:
                    index = token[5:].split('|')[0]
                    region_tokens[token_idx] = int(index)
            region_tokens = torch.tensor(region_tokens).cuda(self.device)
            total_label = region_tokens
            align_pos = torch.where(total_label != 0, torch.ones_like(total_label).cuda(self.device), total_label)
            input_ids = self.bert_toker.convert_tokens_to_ids(input_tokens)
            input_ids = torch.tensor(input_ids).cuda(self.device)
            mask_len = input_ids.size(0)
            input_mask = torch.ones(mask_len).cuda(self.device)
            segment_ids_ques = torch.zeros(len(premise_tokens) + 2, dtype=torch.int64).cuda(self.device)
            segment_ids_ans = torch.ones(len(ans_tokens) + 1, dtype=torch.int64).cuda(self.device)
            segment_ids = torch.cat((segment_ids_ques, segment_ids_ans), 0)


            offsets = self.chunk_mask_dict[i][ans_idx]['offsets']
            chunk_mask = self.chunk_mask_dict[i][ans_idx]['mask'].cuda(self.device)
            gather_index = []
            for idx, set in enumerate(offsets):
                set = torch.tensor(set).cuda(self.device)
                gather_index.extend([idx] * set.size(0))
            gather_index = torch.tensor(gather_index).cuda(self.device)

            if isinstance(answer_label, list):
                if ans_idx in example['answer_label']:
                    target = torch.tensor(1).cuda(self.device)
                else:
                    target = torch.tensor(0).cuda(self.device)
            else:
                if ans_idx == example['answer_label']:
                    target = torch.tensor(1).cuda(self.device)
                else:
                    target = torch.tensor(0).cuda(self.device)

            outputs.append((example['annot_id'], image_input, clip_ans_tokens,
                            r_input_ids, r_segment_ids, r_input_mask, input_ids, segment_ids, input_mask, img_feat, img_mask, target,
                            chunk_mask, gather_index, offsets, example['sent'],
                            example['answer_choices'][ans_idx], total_label, align_pos,
                            object_num, objects_begin, swap_position, swap_opt, all_objects_mask))
            
        return tuple(outputs)

    def SNLIGPT_gen_collate(self, inputs):

        (img_id, image, text, r_input_ids, r_segment_ids, r_input_mask, input_ids, segment_ids, input_mask, img_feat, img_mask, target, chunk_mask,
         gather_index, offsets, ques, ans, total_label, align_pos, object_num, objects_begin, swap_position, swap_opt, all_objects_mask) = map(list, unzip(concat(inputs)))
        

        r_input_ids = pad_sequence(r_input_ids, batch_first=True, padding_value=0)
        r_input_mask = pad_sequence(r_input_mask, batch_first=True, padding_value=0)
        r_segment_ids = pad_sequence(r_segment_ids, batch_first=True, padding_value=0)
        swap_position = pad_sequence(swap_position, batch_first=True, padding_value=-1)

        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)

        all_objects_mask = pad_sequence(all_objects_mask, batch_first=True, padding_value=0)

        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        target = torch.stack(target, dim=0)
        target = target.type(torch.FloatTensor)
        target = target.cuda(self.device)
        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]
        
        input_mask = torch.cat((input_mask, img_mask), -1)
        max_hypo = input_ids.size(1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda(self.device)
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda(self.device)
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)

        batch = {'img_id': img_id, "image" : image, "text":text ,'r_input_ids':r_input_ids, "r_token_type_ids":r_segment_ids,
                 "r_attention_mask":r_input_mask,
                 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'ques_str': ques, 'ans_str': ans,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                 'total_label': total_label, 'align_pos': align_pos, 'object_num': object_num, 'objects_begin': objects_begin, 
                 'swap_position':swap_position, 'swap_opt': swap_opt, 'all_objects_mask': all_objects_mask
                 }

        return batch