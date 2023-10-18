# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import sys
import argparse
import base64
import numpy as np
import os
import os.path as op
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda")
import random, time, json
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from utils.logger import setup_logger
from utils.misc import (mkdir, load_from_yaml_file, find_file_path_in_yaml)
from modeling.modeling_transfomres import BertImgModel
from modeling.modeling_vcr_chunkalign_v10 import ChunkAlign_CLS_enc4_align, SeqBertImgModel, ChunkAlign_CLS_enc4_align_ensemble
from transformers import BertTokenizerFast, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from local_transformers.adapter_transformers.models.roberta import RobertaConfig, RobertaTokenizer, RobertaModel
from modeling.modeling_ensemble import Abstract_Specific
import clip
import pickle
from data_processing.VCRChunkAlign import PMR_ChunkAlign_Dataset_align_ensemble_T
from progressbar import ProgressBar
from transformers import GPT2Tokenizer, GPT2Config
import xlwt
import yaml

def build_dataloader(dataset, is_train, opts):
    if is_train:
        dataloader = DataLoader(dataset, drop_last=True, batch_size=opts.per_gpu_train_batch_size * opts.num_gpus,
                                num_workers=0,
                                shuffle=False, collate_fn=dataset.SNLIGPT_gen_collate)
    else:
        dataloader = DataLoader(dataset, batch_size=opts.per_gpu_eval_batch_size,
                                num_workers=0, shuffle=False, collate_fn=dataset.SNLIGPT_gen_collate)
    return dataloader


def train(args, train_dataloader, val_dataloader, model, tokenizer):

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) //
                                                   args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                  * args.num_train_epochs
        
    seq_align = ['seq_enc']
    grouped_parameters = [

        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in seq_align) and p.requires_grad],
         'lr': args.learning_rate},
        # seqalign
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in seq_align)],
         'lr': args.learning_rate * 0.1},
    ]
    global_step = args.global_step
    optimizer = torch.optim.RMSprop(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    if args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))
    if global_step > 0:
        model_file = os.path.join(args.eval_model_dir, 'model.pth')
        model.load_state_dict(torch.load(model_file))
        optimizer.load_state_dict(torch.load(op.join(args.eval_model_dir, 'optimizer.pth')))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(device)
        scheduler.load_state_dict(torch.load(op.join(args.eval_model_dir, 'scheduler.pth')))
        logger.info("  Resume from %s", args.eval_model_dir)

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                args.per_gpu_train_batch_size * args.num_gpus * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Total examples = %d", len(train_dataloader) * args.per_gpu_train_batch_size * args.num_gpus)

    n_correct_qa_0 = 0
    global_cls_loss = 0.0
    global_align_loss = 0.0
    model.zero_grad()
    model.train()
    n_examples = 0
    total_num = 0
    correct_num = 0
    best_acc = 0
    pbar_len = len(train_dataloader) // args.gradient_accumulation_steps
    for epoch in range(int(args.num_train_epochs)):
        global_loss = 0.0
        global_loss_abstract = 0.0
        global_loss_attention = 0.0
        new_step = 0
        for step, batch in enumerate(train_dataloader):
            inputs = {'img_id': batch['img_id'], 
                      'input_ids': batch['input_ids'],
                      'image': batch['image'],
                      'text': batch['text'],
                      'roberta_input_ids': batch['r_input_ids'],
                      'roberta_token_type_ids': batch['r_token_type_ids'],
                      'roberta_attention_mask': batch['r_attention_mask'],
                      'token_type_ids': batch['token_type_ids'],
                      'input_mask': batch['input_mask'], 'img_feat': batch['img_feat'], 'label': batch['label'],
                      'gather_index': batch['gather_index'],
                      'offsets': batch['offsets'], 'chunk_attention_mask': batch['chunk_attention_mask'],
                      'align_pos': batch['align_pos'], 'total_label': batch['total_label'],
                      'object_num': batch['object_num'], 'objects_begin': batch['objects_begin'],
                      'swap_position': batch['swap_position'], 'swap_opt': batch['swap_opt'],
                      'all_objects_mask': batch['all_objects_mask']
                      }
            outputs = model(**inputs)
            cls_loss = outputs[0]
            loss_abstract = outputs[1][0]
            loss_attention = outputs[1][1]
            loss = cls_loss
            if args.gradient_accumulation_steps > 1:

                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            global_loss += loss.item()
            global_loss_abstract += loss_abstract.item()
            global_loss_attention += loss_attention.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                new_step += 1
                global_step += 1
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                print('Epoch'+str(epoch + 1)+" "+str(new_step)+"/"+str(pbar_len)+', loss'+str(global_loss / new_step))
                print('     loss_abstract'+str(global_loss_abstract / new_step)+', loss_attention'+str(global_loss_attention / new_step))
                if epoch >= args.epoch_begin - 1 and global_step % args.valid_steps == 0:
                    acc = eval(args, val_dataloader, model)
                    logger.info("when epoch {}, the accuracy is {}".format(epoch + 1, acc))
                    if acc > best_acc:
                        best_acc = acc
                        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                        torch.save(state, args.output_dir + "/" + "FSMR" + "-" + str(
                            epoch + 1) + "-" + str(acc) + "-" + str(global_step) + ".pth",
                                   _use_new_zipfile_serialization=False)
                    model.train()


def eval(args, test_dataloader, model):
    time_meter = 0
    result_dict = {}
    n_examples = 0
    n_correct_qa_0 = 0
    model.eval()
    acc = 0
    cnt = 0
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {'img_id': batch['img_id'], 
                      'input_ids': batch['input_ids'],
                      'image': batch['image'],
                      'text': batch['text'],
                      'roberta_input_ids': batch['r_input_ids'],
                      'roberta_token_type_ids': batch['r_token_type_ids'],
                      'roberta_attention_mask': batch['r_attention_mask'],
                      'token_type_ids': batch['token_type_ids'],
                      'input_mask': batch['input_mask'], 'img_feat': batch['img_feat'], 'label': batch['label'],
                      'gather_index': batch['gather_index'],
                      'offsets': batch['offsets'], 'chunk_attention_mask': batch['chunk_attention_mask'],
                      'object_num': batch['object_num'], 'objects_begin': batch['objects_begin'],
                      'swap_position': batch['swap_position'], 'swap_opt': batch['swap_opt'],
                      'all_objects_mask': batch['all_objects_mask']
                      }
            outputs = model(**inputs)
            logits = outputs[2]

            logits = torch.argmax(logits, -1)
            label = batch['label'].reshape(-1, 4)
            label = torch.argmax(label, -1)
            for idx in range(len(logits)):
                if logits[idx] == label[idx]:
                    acc += 1
                cnt += 1
            print(str(step)+" acc_0:"+str(acc / cnt))
        print(str(step)+" acc_0:"+str(acc / cnt))

        return acc / cnt


def test(args, test_dataloader, model, test_label=None):
    model.eval()
    acc = 0
    cnt = 0
    results = []
    logits_list= []
    logit_results = None
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {'img_id': batch['img_id'], 
                      'input_ids': batch['input_ids'],
                      'image': batch['image'],
                      'text': batch['text'],
                      'roberta_input_ids': batch['r_input_ids'],
                      'roberta_token_type_ids': batch['r_token_type_ids'],
                      'roberta_attention_mask': batch['r_attention_mask'],
                      'token_type_ids': batch['token_type_ids'],
                      'input_mask': batch['input_mask'], 'img_feat': batch['img_feat'],
                      'gather_index': batch['gather_index'],
                      'offsets': batch['offsets'], 'chunk_attention_mask': batch['chunk_attention_mask'],
                      'object_num': batch['object_num'], 'objects_begin': batch['objects_begin'],
                      'swap_position': batch['swap_position'], 'swap_opt': batch['swap_opt'],
                      'all_objects_mask': batch['all_objects_mask']
                      }
            outputs = model(**inputs)
            logits = outputs[2]
            logits_copy=logits.clone()

            if logit_results is None:
                logit_results = logits
            else:
                logit_results = torch.cat([logit_results, logits],dim=0)

            logits = torch.argmax(logits, -1)
            label = batch['label'].reshape(-1, 4)
            label = torch.argmax(label, -1)
            for idx in range(len(logits)):
                if logits[idx] == label[idx]:
                    acc += 1
                cnt += 1
            preds = logits.cpu().numpy().tolist()
            results = results + preds
            logits_list = logits_list + logits_copy.cpu().numpy().tolist()

            print(str(step)+" acc_0:"+str(acc / cnt))
        print(str(step)+" acc_0:"+str(acc / cnt))
        print(acc / cnt)
        output_path = args.result_dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        other_info=args.eval_model_dir.split('.')[-2].split('-')[-1]
        if other_info is not None:
            output_file = output_path + str(acc/cnt) + "-" + other_info + ".json"
        else:
            output_file = output_path + str(acc/cnt) + ".json"

        test_data = test_label
        answer = []
        for idx in range(len(test_data)):
            total_id = test_data[idx]['total_id']
            img_id = test_data[idx]['img_id']
            prediction = results[idx]
            answer_type = test_data[idx]['answer_types'][prediction]
            logits_item = logits_list[idx]
            tmp = {"total_id": total_id,
                   "img_id": img_id,
                   "prediction": prediction,
                   "answer_type": answer_type,
                   "logits":logits_item}
            answer.append(json.dumps(tmp, ensure_ascii=False))

        with open(output_file, 'w', encoding='utf-8') as file:
            for line in answer:
                file.write(line+"\n")
                file.flush()

        return acc / cnt


def restore_training_settings(args):
    if args.do_train:
        if not args.scst:
            return args
        checkpoint = args.model_name_or_path
    else:
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(checkpoint, map_location=device)
    if hasattr(train_args, 'max_seq_a_length'):
        if hasattr(train_args, 'scst') and train_args.scst:
            max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
        else:
            max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
            max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
                       'max_img_seq_length']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                                                                              test_v, train_v))
                setattr(args, param, train_v)
    return args


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://'
        )
    return local_rank

def set_seed(sd):
    print("Random seed is set to "+str(sd))
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)

clip_model, preprocess = clip.load('ViT-B/16', device=device, jit=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roberta_file_train",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/train_CALeC.pkl",
                        type=str)
    parser.add_argument("--roberta_file_dev",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/val_CALeC.pkl",
                        type=str)
    parser.add_argument("--roberta_file_test",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/test_CALeC.pkl",
                        type=str)
    parser.add_argument("--clip_file_train",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/clip_data/train_p_ori-clip.jsonl",
                        type=str)
    parser.add_argument("--clip_file_dev",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/clip_data/val_p_ori-clip.jsonl",
                        type=str)
    parser.add_argument("--clip_file_test",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/clip_data/test_p_ori-clip.jsonl",
                        type=str)
    parser.add_argument("--vcr_example_file_train",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/train_CALeC_ori-o.pkl",
                        type=str)
    parser.add_argument("--vcr_example_file_dev",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/val_CALeC_ori-o.pkl",
                        type=str)
    parser.add_argument("--vcr_example_file_test",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/test_CALeC_ori-o.pkl",
                        type=str)
    parser.add_argument("--vcr_feat_file_train",
                        default="Checkpoints_and_Data/Data_feature/pmr_image_feature/train_feat_m.pkl",
                        type=str)
    parser.add_argument("--vcr_feat_file_dev",
                        default="Checkpoints_and_Data/Data_feature/pmr_image_feature/val_feat_m.pkl",
                        type=str)
    parser.add_argument("--vcr_feat_file_test",
                        default="Checkpoints_and_Data/Data_feature/pmr_image_feature/test_feat_m.pkl",
                        type=str)

    parser.add_argument("--vcr_chunk_mask_train",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/ChunkMaskTrain_v4_without_premise.pkl",
                        type=str)
    parser.add_argument("--vcr_chunk_mask_dev",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/ChunkMaskVal_v4_without_premise.pkl",
                        type=str)
    parser.add_argument("--vcr_chunk_mask_test",
                        default="Checkpoints_and_Data/Data_feature/pmr_data/ChunkMaskTest_v4_without_premise.pkl",
                        type=str)

    parser.add_argument("--num_gpus", default=1, type=int, help="Workers in dataloader.")

    parser.add_argument("--train_yaml", default='train.yaml', type=str, required=False,
                        help="yaml file for training.")
    parser.add_argument("--test_yaml", default='test.yaml', type=str, required=False,
                        help="yaml file for testing.")
    parser.add_argument("--val_yaml", default='val.yaml', type=str, required=False,
                        help="yaml file used for validation during training.")
    parser.add_argument("--gpt_model_name_or_path", default='./GPT2', type=str,
                        required=False,
                        help="Path to GPT model.")
    # alignment pretraining checkpoint, containing oscar and phrase-level alignment
    parser.add_argument("--model_name_or_path",
                        default='Checkpoints_and_Data/checkpoint-2000000/',
                        type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--seq_model_name_or_path",
                        default='Checkpoints_and_Data/checkpoint-2000000/',
                        type=str, required=False,
                        help="Path to pre-trained model or model type.")
    #phrase-level alignment
    parser.add_argument("--seq_pretrain_model_dir", type=str,
                        default='Checkpoints_and_Data/checkpoint-6-2625-acc-0.8164/checkpoint-6-2625-acc-0.8164/',
                        help="Model directory for evaluation.")
    parser.add_argument("--output_dir", default='./output/checkpoint/output1/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str,
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=140, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--max_hypo_len", default=50, type=int,
                        help="The maximum sequence length for hypothesis.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--add_residual", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--add_local_residual", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--wo_gate", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mask_prob", default=0.0, type=float,
                        help="Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=3,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--add_od_labels", default=False, action='store_true',
                        help="Whether to add object detection labels or not")
    parser.add_argument("--drop_out", default=0.3, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=150, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--tie_weights", default=False, action='store_true',
                        help="Whether to tie decoding weights to that of encoding")
    parser.add_argument("--freeze_embedding", default=False, action='store_true',
                        help="Whether to freeze word embeddings in Bert")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_ratio", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_after", default=0, type=int,
                        help=".")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-5, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=30, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    # for self-critical sequence training
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')
    parser.add_argument('--sc_train_sample_n', type=int, default=2,
                        help="number of sampled captions for sc training")
    parser.add_argument('--sc_baseline_type', type=str, default='greedy',
                        help="baseline tyep of REINFORCE algorithm")
    parser.add_argument('--beam_size', type=int, default=5,
                        help="beam size for scst training")
    parser.add_argument('--cider_cached_tokens', type=str, default='coco-train-words.p',
                        help="path to cached cPickle file used to calculate CIDEr scores")
    # for generation
    parser.add_argument("--eval_model_dir", type=str,
                        default='./output/checkpoint/output1/FSMR-2-0.8159947984395318-2400.pth',
                        help="Model directory for evaluation.")

    parser.add_argument('--max_gen_length', type=int, default=40,
                        help="max length of generated sentences")
    parser.add_argument('--output_hidden_states', action='store_true',
                        help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    parser.add_argument('--num_beams', type=int, default=1, help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")
    # for Constrained Beam Search
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument('--use_cbs', action='store_true',
                        help='Use constrained beam search for decoding')
    parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                        help="minimum number of constraints to satisfy")
    parser.add_argument("--epoch_begin", default=2, type=int)
    parser.add_argument("--valid_steps", default=400, type=int,
                        help="Run validation begin")
    parser.add_argument("--result_dir", default="./output/results/output1/", type=str)
    parser.add_argument(
        "--global_step", default=0, type=int,
        help="")
    parser.add_argument(
        "--example_index", default=None, type=int,
        help="")

    if os.path.exists("config.yaml"):
        print("#############################   loading cofig from 'config.yaml'   #############################")
        with open('config.yaml', 'r') as f:
            default_arg = yaml.load(f,Loader=yaml.FullLoader)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()

    if args.do_train:
        args.do_test=False
    else:
        args.do_test=True

    set_seed(args.seed)
    global logger
    gpt_tokenizer = None
    # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.distributed = False
    assert args.valid_steps % args.gradient_accumulation_steps == 0
    args.device = device
    synchronize()

    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("vlpretrain", output_dir, args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    args = restore_training_settings(args)

    # Load pretrained model and tokenizer

    assert args.model_name_or_path is not None
    config_class, model_class, tokenizer_class = BertConfig, BertImgModel, BertTokenizerFast
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=args.num_labels,
                                          finetuning_task='image_captioning')

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                                                    else args.model_name_or_path, do_lower_case=args.do_lower_case)
    det_tokens = ["<|det%d|>" % i for i in range(45)]
    tokenizer.add_special_tokens({"additional_special_tokens": det_tokens})
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.tie_weights = args.tie_weights
    config.freeze_embedding = args.freeze_embedding
    config.label_smoothing = args.label_smoothing
    config.drop_worst_ratio = args.drop_worst_ratio
    config.drop_worst_after = args.drop_worst_after
    config.output_attentions = True
    oscar_model = model_class.from_pretrained(args.model_name_or_path,
                                              from_tf=False, config=config)

    oscar_model.resize_token_embeddings(len(tokenizer))

    seq_config_class, seq_model_class = BertConfig, SeqBertImgModel
    seq_config = seq_config_class.from_pretrained(args.seq_model_name_or_path, num_labels=args.num_labels,
                                                  finetuning_task='image_captioning')

    seq_config.img_feature_dim = args.img_feature_dim
    seq_config.img_feature_type = args.img_feature_type
    seq_config.hidden_dropout_prob = args.drop_out
    seq_config.loss_type = args.loss_type
    seq_config.tie_weights = args.tie_weights
    seq_config.freeze_embedding = args.freeze_embedding
    seq_config.label_smoothing = args.label_smoothing
    seq_config.drop_worst_ratio = args.drop_worst_ratio
    seq_config.drop_worst_after = args.drop_worst_after
    seq_config.max_hypo = args.max_hypo_len
    seq_config.output_attentions = True
    seq_config.add_residual = args.add_residual
    seq_config.add_local_residual = args.add_local_residual
    seq_model = seq_model_class.from_pretrained(args.seq_model_name_or_path,
                                                from_tf=False, config=seq_config)
    # loading seqAlign
    model_file = os.path.join(args.seq_pretrain_model_dir, 'model.pth')

    a = torch.cuda.is_available()

    pretrained_dict = torch.load(model_file, map_location='cpu')
    renamed_dict = {}
    for k, v in pretrained_dict.items():
        if 'seq_enc' in k:
            k = '.'.join(k.split('.')[1:])
            renamed_dict[k] = v
    seq_model.load_state_dict(renamed_dict)
    logger.info("load pretrained ChunkAlign from %s", args.seq_pretrain_model_dir)
    seq_model.resize_token_embeddings(len(tokenizer))
    calec_model = ChunkAlign_CLS_enc4_align_ensemble(oscar_model, seq_model, num_labels=4)


    # ------------------- additional_roberta----------------------
    roberta_train = args.roberta_file_train
    roberta_val = args.roberta_file_dev
    roberta_test = args.roberta_file_test
    R_MODEL_PATH = "local_transformers/roberta-large/"
    roberta_tokenizer = RobertaTokenizer.from_pretrained(R_MODEL_PATH)
    roberta_tokenizer.add_special_tokens({"additional_special_tokens": det_tokens})
    config = RobertaConfig.from_pretrained(R_MODEL_PATH + "config.json")
    roberta_model = RobertaModel.from_pretrained(R_MODEL_PATH, config=config)
    roberta_model.resize_token_embeddings(len(roberta_tokenizer))
    roberta_model.config.type_vocab_size = 2
    roberta_model.embeddings.token_type_embeddings = nn.Embedding(2, roberta_model.config.hidden_size)
    roberta_model.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0,
                                                                       std=roberta_model.config.initializer_range)

    clip_train = None
    clip_val = None
    clip_test = None

    clip.model.convert_weights(clip_model)
    for name, parameter in clip_model.named_parameters():
        parameter.requires_grad = False


    model = Abstract_Specific(calec_model=calec_model, clip_model=clip_model, roberta_model=roberta_model, num_labels=4)

    if args.do_test:
        model_file = args.eval_model_dir
        ck = torch.load(model_file, map_location='cpu')
        params = ck['net']
        model.load_state_dict(params)
    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_file = 'Multi-View-Reasoning-cold-start-1.pth'
        params = torch.load(model_file, map_location='cpu')['net']
        params_c = params.copy()
        for n, p in params_c.items():
            if 'mapping_network_vision.' in n:
                del params[n]
            elif 'mapping_network_alignment.' in n:
                del params[n]
            else:
                continue
        model.load_state_dict(params, False)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = PMR_ChunkAlign_Dataset_align_ensemble_T(tokenizer, gpt_tokenizer, roberta_tokenizer,
                                                                args.vcr_example_file_train,
                                                                args.vcr_chunk_mask_train,
                                                                args.vcr_feat_file_train,
                                                                roberta_train,
                                                                preprocess,
                                                                clip_train,
                                                                device)
        train_dataloader = build_dataloader(train_dataset, True, args)
        val_dataset = PMR_ChunkAlign_Dataset_align_ensemble_T(tokenizer, gpt_tokenizer, roberta_tokenizer,
                                                              args.vcr_example_file_dev,
                                                              args.vcr_chunk_mask_dev,
                                                              args.vcr_feat_file_dev,
                                                              roberta_val,
                                                              preprocess,
                                                              clip_val,
                                                              device)
        val_dataloader = build_dataloader(val_dataset, False, args)
        last_checkpoint = train(args, train_dataloader, val_dataloader, model, tokenizer)
    else:
        test_dataset = PMR_ChunkAlign_Dataset_align_ensemble_T(tokenizer, gpt_tokenizer, roberta_tokenizer,
                                                              args.vcr_example_file_test,
                                                              args.vcr_chunk_mask_test,
                                                              args.vcr_feat_file_test,
                                                              roberta_test,
                                                              preprocess,
                                                              clip_test,
                                                              device)
        test_dataloader = build_dataloader(test_dataset, False, args)

        test_data_f = open('./Checkpoints_and_Data/meta_data/test-ori.jsonl', 'rb')
        test_label = []
        for f in test_data_f:
           test_label.append(json.loads(f))
        test(args, test_dataloader, model, test_label=test_label)


if __name__ == "__main__":
    main()