import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
#from torch.utils.data import DataLoader

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from BertMaxPdataset import BertMaxPDataset
from longformerdataset import longformerMaxpDataset
from BertMaxP import BertMaxP
from dataloader import DataLoader
from Longformer import LongformerMaxp

def dev(args, model, dev_loader, device):
    rst_dict = {}
    for dev_batch in dev_loader:
        query_id, doc_id, label, retrieval_score = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label'], dev_batch['retrieval_score']
        with torch.no_grad():
            if args.model == 'bert':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device), dev_batch['segment_ids'].to(device))
            elif args.model == 'longformer':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device), dev_batch['segment_ids'].to(device), dev_batch['global_attention_mask'].to(device))
            else:
                batch_score, _ = model(dev_batch['query_idx'].to(device), dev_batch['query_mask'].to(device),
                                       dev_batch['doc_idx'].to(device), dev_batch['doc_mask'].to(device))
            if args.task == 'classification':
                ##對[batch_size,2]做softmax取[:,1]label的機率
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            batch_score = [batch_score.detach().cpu().tolist()]
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]
    return rst_dict

def train(args, model, loss_fn, m_optim, m_scheduler, train_loader, dev_loader, device):
    best_F1 = 0.0
    for epoch in range(args.epoch):
        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):
            print(step, train_batch)
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device), train_batch['segment_ids_neg'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'longformer':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device), train_batch['segment_ids_neg'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device), train_batch['global_attention_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_pos_idx'].to(device), train_batch['doc_pos_mask'].to(device))
                    batch_score_neg, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_neg_idx'].to(device), train_batch['doc_neg_mask'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                           train_batch['doc_idx'].to(device), train_batch['doc_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            if args.task == 'ranking':
                batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(device))
            elif args.task == 'classification':
                batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
            if torch.cuda.device_count() > 1:
                batch_loss = batch_loss.mean()

            avg_loss += batch_loss.item()
            batch_loss = batch_loss / args.accumulation_steps
            batch_loss.backward()
            if (step+1) % args.accumulation_steps == 0:
                m_optim.step()
                m_scheduler.step()
                m_optim.zero_grad()    
            
            if (step+1) % args.eval_every == 0:
                with torch.no_grad():
                    rst_dict = dev(args, model, dev_loader, device)
                with open("dev_result.tmp", 'w') as writer:
                    F1score=0
                    TP=0
                    FP=0
                    FN=0
                    for q_id, scores in rst_dict.items():
                        res = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
                        for rank, value in enumerate(res):
                            if rank<10:
                                if value[1][1] == '1':
                                    TP +=1
                                else:
                                    FP +=1
                            else:
                                if value[1][1] == '1':
                                    FN +=1    
                            writer.write(q_id+' '+str(value[0])+' '+str(rank+1)+' '+ str(value[1][0])+' '+str(args.model)+'\n')
                print(TP,FP,FN)
                Precision = TP/(TP+FP)
                Recall = TP/(TP+FN)
                if Precision==0 and Recall==0:
                    print("ZERO")
                else:
                    F1score = 2*Precision/(Precision+Recall)
                with open("result.tmp","a") as writer:
                    writer.write(str(Precision)+' '+str(Recall)+' '+str(F1score)+' '+str(best_F1)+'\n')
                print('save_model...')
                torch.save(model.state_dict(), args.save+str(epoch)+str(step))           
                avg_loss = 0.0
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-reinfoselect', action='store_true', default=False)
    parser.add_argument('-reset', action='store_true', default=False)
    parser.add_argument('-train', type=str, default='./data/train_toy.jsonl')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-dev', type=str, default='./data/dev_toy.jsonl')
    parser.add_argument('-qrels', type=str, default='./data/qrels_toy')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-metric', type=str, default='ndcg_cut_10')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=20)
    parser.add_argument('-max_doc_len', type=int, default=150)
    parser.add_argument('-maxp', action='store_true', default=False)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-accumulation_steps', type=int, default=8)
    parser.add_argument('-eval_every', type=int, default=1000)
    args = parser.parse_args()

    args.model = args.model.lower()

    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading training data...')
        if args.maxp:
            train_set = BertMaxPDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        '''
        else:
            train_set = om.data.datasets.BertDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        '''
        print('reading dev data...')
        if args.maxp:
            dev_set = BertMaxPDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        '''
        else:
            dev_set = om.data.datasets.BertDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
               task=args.task
            )
        '''
    elif args.model == 'longformer':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading training data...')
        if args.maxp:
            train_set = longformerMaxpDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        print('reading dev data...')
        if args.maxp:
            dev_set = longformerMaxpDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    
    dev_loader = DataLoader(
        dataset=dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    if args.model == 'bert' or args.model == 'roberta':
        if args.maxp:
            model = BertMaxP(
                pretrained=args.pretrain,
                max_query_len=args.max_query_len,
                max_doc_len=args.max_doc_len,
                mode=args.mode,
                task=args.task
            )
        '''
        else:
            model = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task=args.task
            )
        
        if args.reinfoselect:
            policy = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task='classification'
            )
        '''
    elif args.model == "longformer":
        if args.maxp:
            model = LongformerMaxp(
                pretrained=args.pretrain,
                max_query_len=args.max_query_len,
                max_doc_len=args.max_doc_len,
                mode=args.mode,
                task=args.task
            )
    else:
        raise ValueError('model name error.')

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)
        if args.model == 'bert':
            st = {}
            for k in state_dict:
                if k.startswith('bert'):
                    st['_model'+k[len('bert'):]] = state_dict[k]
                elif k.startswith('classifier'):
                    st['_dense'+k[len('classifier'):]] = state_dict[k]
                else:
                    st[k] = state_dict[k]
            model.load_state_dict(st)
        else:
            model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    if args.reinfoselect:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError('Task must be `ranking` or `classification`.')
    else:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1)
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//args.batch_size)
    #if args.reinfoselect:
    #    p_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=args.lr)
    #metric = om.metrics.Metric()
    model.to(device)
    #if args.reinfoselect:
    #    policy.to(device)
    loss_fn.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        loss_fn = nn.DataParallel(loss_fn)

    #if args.reinfoselect:
    #    train_reinfoselect(args, model, policy, loss_fn, m_optim, m_scheduler, p_optim, metric, train_loader, dev_loader, device)
    #else:
    train(args, model, loss_fn, m_optim, m_scheduler, train_loader, dev_loader, device)

if __name__ == "__main__":
    main()