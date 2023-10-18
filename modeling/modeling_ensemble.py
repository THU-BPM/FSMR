
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from a_transformers.modeling_bert import (BertEmbeddings,
                                              BertSelfAttention, BertAttention, BertEncoder, BertLayer,
                                              BertSelfOutput, BertIntermediate, BertOutput,
                                              BertPooler, BertPreTrainedModel)

from torch.nn.utils.rnn import pad_sequence
from collections import UserDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessorList, MinLengthLogitsProcessor,
                          BeamScorer)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from modeling.modeling_vcr_chunkalign_v10 import ChunkAlign_CLS_enc4_align_ensemble
from local_transformers.adapter_transformers.models.roberta import RobertaModel

import sys
import os
sys.path.append(os.path.dirname(__file__))
from attention import (MultiheadAttention, AttentionPool)

class Abstract_Specific(nn.Module):
    def __init__(self, roberta_model, calec_model, clip_model=None, num_labels=4):
        super(Abstract_Specific, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.roberta = roberta_model
        # attention layer
        if clip_model is not None:
            self.clip_model = clip_model
            self.classifier = nn.Linear(1024 + 768 + 512, 1)
        else:
            self.classifier = nn.Linear(768 + 768, 1)

        self.abst_confidence_scorer = nn.Linear(1024, 1)
        self.confidence_scorer = nn.Linear(768, 1)
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.214),
            torch.nn.Linear(768, 768 * 5, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.214),
            torch.nn.Linear(768 * 5, 1024 * 5, bias=True)
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.214),
            torch.nn.Linear(768, 768 * 3, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.214),
            torch.nn.Linear(768 * 3, 1024 * 3, bias=True)
        )
        self.promptfuse = torch.nn.Embedding(2, 1024)
        self.img_feat_adapt = nn.Sequential(
            nn.Dropout(p=0.237),
            torch.nn.Linear(2054, 1024, bias=True),
            torch.nn.Sigmoid()
        )
        self.attn1=MultiheadAttention(1024,16,0.214)
        self.attn2=MultiheadAttention(1024,16,0.214)
        self.fc = nn.Sequential(
            nn.Linear(2*1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.214))
        self.attn_pool = AttentionPool(1024,0.214)
        self.final_output = nn.Linear(2*1024, 1)


    def forward(self, img_id, image, text, roberta_input_ids, roberta_token_type_ids, roberta_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None, object_num=None, objects_begin=None, swap_position=None, swap_opt=None, all_objects_mask=None):


        # vision representations
        with torch.no_grad():
            img_attention_mask = torch.cat([input_mask[:, :1], input_mask[:, -img_feat.size(1):]], dim=-1)
            image_features_ = self.calec.global_enc(input_ids[:, :1], img_feats=img_feat, attention_mask=img_attention_mask,
                                      position_ids=None, token_type_ids=None,
                                      head_mask=None,
                                      encoder_history_states=None)
            
        prefix_vision = self.mapping_network_vision(image_features_[0][:, 0, :])
        prefix_vision = prefix_vision.reshape(input_ids.size(0), 3, 1024)
        vision_mask = input_mask[:, :1].repeat(1, 3)


        # alignment representations
        CALeC_encoder_output, align_loss, specific_alignment = self.calec(input_ids=input_ids, img_feat=img_feat,
                                                                          input_mask=input_mask, token_type_ids=token_type_ids,
                                                                          position_ids=position_ids, head_mask=head_mask,
                                                                          encoder_history_states=encoder_history_states, offsets=offsets,
                                                                          chunk_attention_mask=chunk_attention_mask, gather_index=gather_index,
                                                                          align_pos=align_pos, total_label=total_label,
                                                                          abstract_hidden_states=None)


        Alignment_prompt = self.mapping_network_alignment(CALeC_encoder_output).unsqueeze(1).view(input_ids.size(0), 5, 1024)
        align_mask = input_mask[:, :1].repeat(1, 5)
        
        ##### visual + align
        prefix_emb = torch.cat([prefix_vision, Alignment_prompt], dim=1)
        prompt_mask = torch.cat([vision_mask, align_mask], dim=1)

        ##### image features
        adapt_img_feat=self.img_feat_adapt(img_feat)

        roberta_encoder_outputs = self.roberta(input_ids=roberta_input_ids, token_type_ids=roberta_token_type_ids,
                                              attention_mask=roberta_attention_mask, prompt_embeddings=prefix_emb, input_mask=prompt_mask,
                                              object_num=object_num, objects_begin=objects_begin, img_feat=adapt_img_feat,
                                              swap_position=swap_position, swap_opt=swap_opt)
        roberta_encoder_output = roberta_encoder_outputs[1]

        sequence_output = roberta_encoder_outputs[0]

        wrc_loss = None

        ##### Multi-head attention --- attn_logit
        left_mask = torch.cat([prompt_mask, roberta_attention_mask[:, 2:]], dim=1) == 0
        right_mask = all_objects_mask == 0
        left_out = sequence_output.transpose(0, 1)
        right_out = adapt_img_feat.transpose(0, 1)
        l2r_attn, _ = self.attn1(left_out, right_out, right_out, key_padding_mask=right_mask)
        r2l_attn, _ = self.attn2(right_out, left_out, left_out, key_padding_mask=left_mask)
        left_out = self.fc(torch.cat([l2r_attn, left_out], dim=-1)).transpose(0, 1)
        right_out = self.fc(torch.cat([r2l_attn, right_out], dim=-1)).transpose(0, 1)
        left_out = self.attn_pool(left_out, left_mask)
        right_out = self.attn_pool(right_out, right_mask)
        attn_logit = self.final_output(torch.cat([left_out, right_out], dim=-1))

        ##### abst_logit
        abstract_level = roberta_encoder_output
        abst_logit = self.abst_confidence_scorer(abstract_level)

        reshaped_abst_logits = abst_logit.view(-1, self.num_labels)
        reshaped_attn_logits = attn_logit.view(-1, self.num_labels)

        loss = None
        loss_specific = None
        loss_abstract = None
        loss_attention = None
        align_f_loss = None
        logits = None
        if label is not None and total_label is not None:
            loss_fct = CrossEntropyLoss()
            
            label = label.view(reshaped_abst_logits.size())
            loss_abstract = loss_fct(reshaped_abst_logits, label)
            loss_attention = loss_fct(reshaped_attn_logits, label)
            
            align_f_loss = align_loss
            loss = loss_abstract * 0.5 + loss_attention * 0.5

        reshaped_attn_logits=F.softmax(reshaped_attn_logits,dim=1)
        reshaped_abst_logits=F.softmax(reshaped_abst_logits,dim=1)
        logits = reshaped_attn_logits * 0.5 + reshaped_abst_logits * 0.5
        return loss, (loss_abstract, loss_attention, loss_specific, align_f_loss, wrc_loss), logits